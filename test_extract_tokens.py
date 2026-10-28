"""只用合成字符串、微型假编码器和临时文件检查提取器；不加载预训练模型。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import extract_tokens as ex


def item(text, **extra):
    return {"key": hashlib.sha256(text.encode()).hexdigest(), "input": text, **extra}


class FakeTokenizer:
    eos_token_id, pad_token_id, unk_token_id = 1, 0, 2

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return [2 if c == "?" else ord(c) + 3 for c in text if c != " "]

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False and kwargs["padding"] is False
        content = self.encode(text, add_special_tokens=False, truncation=False)
        # 故意加 padding，验证提取器也能严格剔除 PAD 而保留 UNK。
        return {"input_ids": content + [1, 0, 0],
                "attention_mask": [1] * (len(content) + 1) + [0, 0]}

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


class TinyEncoder(torch.nn.Module):
    calls = []
    loads = 0

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(d_model=768)

    def forward(self, input_ids, attention_mask, return_dict):
        assert not self.training and not self.weight.requires_grad and not torch.is_grad_enabled()
        type(self).calls.append(int(attention_mask.sum()))
        values = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 768).clone()
        return SimpleNamespace(last_hidden_state=values)

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        assert kwargs["use_safetensors"] is False and kwargs["torch_dtype"] == torch.float32
        cls.loads += 1
        return cls(), {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}


class ExtractionTests(unittest.TestCase):
    def test_unknown_is_kept_and_eos_padding_are_removed(self):
        rec = ex.tokenize_item(FakeTokenizer(), item("C?O"), "molt5")
        self.assertEqual(rec["positions"], [0, 1, 2])
        self.assertEqual(rec["metadata"]["unknown_token_count"], 1)
        model = TinyEncoder().requires_grad_(False).eval()
        arr = ex.extract_one(model, torch, torch.device("cpu"), rec)
        self.assertEqual(arr.shape, (3, 768))
        self.assertEqual(float(arr[1, 0]), FakeTokenizer.unk_token_id)

    def test_silent_token_truncation_is_rejected(self):
        class TruncatingTokenizer(FakeTokenizer):
            def __call__(self, text, **kwargs):
                rec = super().__call__(text, **kwargs)
                rec["input_ids"].pop(0)
                rec["attention_mask"].pop(0)
                return rec
        with self.assertRaisesRegex(ValueError, "完整内容"):
            ex.tokenize_item(TruncatingTokenizer(), item("CCO"), "molt5")

    def test_protein_position_count_is_checked(self):
        class SplittingTokenizer(FakeTokenizer):
            def encode(self, text, **kwargs):
                return super().encode(text, **kwargs) + [99]
        with self.assertRaisesRegex(ValueError, "残基数"):
            ex.tokenize_item(SplittingTokenizer(), item("ACX"), "protein")

    def test_committed_cache_and_changed_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = ex.tokenize_item(FakeTokenizer(), item("CO"), "molt5")["metadata"]
            metadata, provenance = {meta["key"]: meta}, {"kind": "molt5", "test": 1}
            manifest = ex.open_manifest(root, provenance, metadata)
            ex.commit_array(root, manifest, np.ones((2, 768), dtype=np.float32), meta)
            reopened = ex.open_manifest(root, provenance, metadata)
            self.assertEqual(len(reopened["records"]), 1)
            with self.assertRaisesRegex(ValueError, "不完全一致"):
                ex.open_manifest(root, {**provenance, "test": 2}, metadata)
            np.save(root / meta["file"], np.zeros((2, 768), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "SHA256 不一致"):
                ex.open_manifest(root, provenance, metadata)

    def test_interruption_between_array_and_manifest_commit_recovers(self):
        for final_already_renamed in (False, True):
            with self.subTest(final_already_renamed=final_already_renamed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                meta = ex.tokenize_item(FakeTokenizer(), item("N"), "molt5")["metadata"]
                metadata, provenance = {meta["key"]: meta}, {"kind": "molt5"}
                manifest = ex.open_manifest(root, provenance, metadata)
                temporary = root / ("." + meta["key"] + "." + "a" * 32 + ".tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, np.ones((1, 768), dtype=np.float32))
                pending = {"provenance_sha256": ex.canonical_sha(provenance),
                           "record": {**meta, "sha256": ex.file_sha(temporary)},
                           "temporary_file": temporary.name}
                ex.atomic_json(root / ".receipts" / (meta["key"] + ".json"), pending)
                if final_already_renamed:
                    temporary.rename(root / meta["file"])
                recovered = ex.open_manifest(root, provenance, metadata)
                self.assertIsNone(recovered["pending"])
                self.assertIn(meta["key"], recovered["records"])
                self.assertTrue((root / meta["file"]).is_file())

    def test_orphan_and_nonfinite_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = ex.tokenize_item(FakeTokenizer(), item("N"), "molt5")["metadata"]
            np.save(root / meta["file"], np.full((1, 768), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "无 manifest"):
                ex.open_manifest(root, {"kind": "molt5"}, {meta["key"]: meta})
            record = {**meta, "sha256": ex.file_sha(root / meta["file"])}
            with self.assertRaisesRegex(ValueError, "有限性"):
                ex.verify_array(root / meta["file"], record, meta)

    def test_checkpoint_scope_and_untracked_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ex.REQUIRED_CHECKPOINT_FILES:
                (root / name).write_bytes(name.encode())
            (root / "model.safetensors").write_bytes(b"unused alternative weight")
            first = ex.checkpoint_fingerprint(root)
            self.assertEqual(set(first), set(ex.REQUIRED_CHECKPOINT_FILES))
            (root / "added_tokens.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "added_tokens"):
                ex.checkpoint_fingerprint(root)

    def test_complete_flow_longest_first_and_verified_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, checkpoint = Path(tmp), Path(tmp) / "weights"
            checkpoint.mkdir()
            for name in ex.REQUIRED_CHECKPOINT_FILES:
                (checkpoint / name).write_bytes(name.encode())
            expected = {k: v["sha256"] for k, v in ex.checkpoint_fingerprint(checkpoint).items()}
            items = root / "items.json"
            items.write_text(json.dumps({"kind": "molt5", "expected_checkpoint_files": expected,
                                         "items": [item("C"), item("C?O")]}), encoding="utf-8")
            args = argparse.Namespace(kind="molt5", items=items, checkpoint=checkpoint,
                                      output=root / "out", device="cpu")
            fake_transformers = SimpleNamespace(T5Tokenizer=FakeTokenizer, T5EncoderModel=TinyEncoder)
            TinyEncoder.calls, TinyEncoder.loads = [], 0
            with patch.dict("sys.modules", {"transformers": fake_transformers}), patch.object(
                    ex, "configure_runtime", return_value=(torch, torch.device("cpu"), {"test": True})), redirect_stdout(io.StringIO()):
                ex.run(args)
                ex.run(args)
            self.assertEqual(TinyEncoder.calls, [4, 2])  # 含 EOS：最长内容 3 个，较短内容 1 个。
            self.assertEqual(TinyEncoder.loads, 1)     # 完整缓存重新逐项验证后无需重做推断。
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["records"]), 2)
            self.assertTrue(manifest["longest_preflight"]["passed"])

    def test_invalid_input_key_and_unapproved_crop_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "items.json"
            for value in ({"key": "0" * 64, "input": "AC"}, item("A" * 1001)):
                path.write_text(json.dumps({"expected_checkpoint_files": {"config.json": "0" * 64},
                                            "items": [value]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ex.read_items(path, "protein")


if __name__ == "__main__":
    unittest.main()
