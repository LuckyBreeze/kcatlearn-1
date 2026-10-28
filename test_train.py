"""短 CPU 合成验收；不读取真实数据，不调用正式训练 CLI，不提交任务。"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import SequentialSampler

import train


def batch(size, protein_length=5, molt5_length=4):
    generator = torch.Generator().manual_seed(17 + size)
    pm = torch.ones(size, protein_length, dtype=torch.bool)
    sm = torch.ones(size, molt5_length, dtype=torch.bool)
    pm[0, -1] = False
    sm[1, -1] = False
    return dict(protein_tokens=torch.randn(size, protein_length, 1024, generator=generator),
                protein_mask=pm,
                molt5_tokens=torch.randn(size, molt5_length, 768, generator=generator),
                molt5_mask=sm,
                maccs=torch.randint(0, 2, (size, 167), generator=generator).float(),
                label=torch.linspace(-1, 1, size), row_uid=[f"synthetic-{i}" for i in range(size)])


class TrainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_preflight_updates_discarded_and_model_optimizer_roundtrip(self):
        train.set_seed()
        model = train.make_model("cpu")
        initial = train.state_sha(model)
        with tempfile.TemporaryDirectory() as directory:
            result = train.preflight_check(model, train.make_optimizer(model),
                [("batch8", batch(8)), ("batch9", batch(9, 6, 5))], "cpu", directory)
        self.assertEqual(result["initial_sha256"], initial)
        self.assertTrue(result["adam_roundtrip"])
        self.assertTrue(result["eval_roundtrip_exact"])
        self.assertNotEqual(initial, train.state_sha(model))
        self.assertEqual([x["bn_updates"] for x in result["batches"]],
                         [{"readout.decoder.1": 1}, {"readout.decoder.1": 1}])
        train.set_seed()
        self.assertEqual(train.state_sha(train.make_model("cpu")), initial)

    def test_terminal_singleton_is_merged_without_dropping_any_row(self):
        indices = list(train.MergeSingletonBatchSampler(SequentialSampler(range(17)), 8, False))
        self.assertEqual([len(x) for x in indices], [8, 9])
        self.assertEqual(sum(indices, []), list(range(17)))
        self.assertEqual(len(train.MergeSingletonBatchSampler(SequentialSampler(range(21585)), 8, False)), 2698)

    def test_eval_identity_and_batch_mean_loss(self):
        train.set_seed()
        model = train.make_model("cpu")
        batches = [batch(8), batch(3)]
        batches[1]["row_uid"] = ["tail-0", "tail-1", "tail-2"]
        actual = train.run_epoch(model, batches, "cpu")
        with torch.no_grad():
            expected_losses = [float(train.forward(model, item)[1]) for item in batches]
        self.assertAlmostEqual(actual["metrics"]["batch_average_RMSELoss"], np.mean(expected_losses), places=7)
        self.assertEqual(len(actual["labels"]), 11)
        class Dataset:
            rows = [dict(row_uid=str(uid), label=float(label))
                    for uid, label in zip(actual["row_uid"], actual["labels"])]
        train.check_eval_identity(actual, Dataset())
        Dataset.rows[0]["row_uid"] = "wrong-id"
        with self.assertRaisesRegex(RuntimeError, "identity"):
            train.check_eval_identity(actual, Dataset())

    def test_loss_rejects_broadcast_shape(self):
        model = train.make_model("cpu").eval()
        values = batch(2)
        values["label"] = values["label"][:, None]
        with self.assertRaisesRegex(RuntimeError, "广播"):
            train.forward(model, values)

    def test_real_collator_preflight_retains_both_length_maxima(self):
        class Dataset:
            protein_lengths = [12, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            molt5_lengths = [2, 14, 3, 4, 5, 6, 7, 8, 9, 10]

            def __len__(self):
                return 10

            def __getitem__(self, i):
                return dict(protein_tokens=torch.zeros(self.protein_lengths[i], 1024),
                            molt5_tokens=torch.zeros(self.molt5_lengths[i], 768),
                            maccs=torch.zeros(167), label=torch.tensor(float(i)), row_uid=str(i))
        batches = dict(train.preflight_batches(Dataset()))
        worst = batches["worst_padding_real_train_9"]
        self.assertEqual(worst["protein_tokens"].shape, (9, 12, 1024))
        self.assertEqual(worst["molt5_tokens"].shape, (9, 14, 768))
        self.assertEqual(len(set(worst["row_uid"])), 9)
        self.assertIn("0", worst["row_uid"])
        self.assertIn("1", worst["row_uid"])

    def test_author_selection_initial_counter_and_min_delta(self):
        stop = train.author.EarlyStopping(20, 0.001)
        self.assertEqual(stop.check(0, 1.0), (True, False))
        self.assertEqual(stop.count_epoch, 1)
        self.assertEqual(stop.check(1, 0.9995), (False, False))
        self.assertEqual(stop.count_epoch, 2)
        self.assertEqual(stop.check(2, 0.998), (True, False))
        self.assertEqual(stop.count_epoch, 0)
        for epoch in range(3, 23):
            best, stopping = stop.check(epoch, 1.0)
            self.assertFalse(best)
            self.assertEqual(stopping, epoch == 22)

    def test_selection_lock_replay_rejects_incomplete_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # 只有第一轮时还不满足自然早停，不能提前test。
            history = [dict(epoch_zero_based=0, epoch=1,
                            valid=dict(batch_average_RMSELoss=1.0),
                            is_best=True, early_stop=False, counter=1)]
            train.atomic_json(output / "history.json", history)
            train.atomic_json(output / "protocol.json", {})
            (output / "best.pt").write_bytes(b"synthetic checkpoint")
            (output / "valid_best.npz").write_bytes(b"synthetic predictions")
            lock = dict(best_epoch_zero_based=0, completed_epochs=1)
            for filename, key in (("best.pt", "checkpoint_sha256"),
                                  ("valid_best.npz", "valid_predictions_sha256"),
                                  ("history.json", "history_sha256"),
                                  ("protocol.json", "protocol_sha256")):
                lock[key] = train.sha256(output / filename)
            train.atomic_json(output / "selection_lock.json", lock)
            with self.assertRaisesRegex(RuntimeError, "naturally finished"):
                train.verify_selection(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
