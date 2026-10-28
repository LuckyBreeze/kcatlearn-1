#!/usr/bin/env python3
"""已批准模型二的单次试跑：token 预检、训练、锁定后的独立 test。

正式运行只接受 CUDA。特征来自冻结编码器；本文件只训练下游模型。
author_utils.py 原样保存作者损失、早停和指标实现，不修改其广播形状。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, RandomSampler

import author_utils as author
from model import (AttentionConfig, CataProTokenRegressor, MaskedMeanKcatHead,
                   ResidueSubstrateCrossAttention)
from tail_batch import MergeSingletonBatchSampler

CONFIG = dict(seed=668, num_heads=8, attention_dropout=0.0, output_dropout=0.0,
              head_dropout=0.0, linear_bias=True, optimizer="Adam", lr=1e-5,
              betas=[0.9, 0.999], weight_decay=0.01, batch_size=8,
              train_tail="merge_terminal_8_plus_1_to_9", max_epochs=150,
              patience=20, min_delta=0.001, eps=1e-6, num_workers=8,
              selection="author_batch_average_RMSELoss_raw", amp=False,
              scheduler=None, auxiliary_loss=None, ema=False)
AUTHOR_SHA = "1c5dfb08259a90eee896ec9ca1b3fcbb8dfe641b53e85f750b64f7ddaeaa762d"
EXPECTED_ROWS = {"train": 21585, "valid": 3037, "test": 3036}
FEATURE_KEYS = ("protein_tokens", "protein_mask", "molt5_tokens", "molt5_mask", "maccs")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path, value):
    tmp = Path(path).with_suffix(".pt.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = Path(path).with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def set_seed():
    random.seed(668)
    np.random.seed(668)
    torch.manual_seed(668)
    torch.cuda.manual_seed_all(668)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # FP32 权重与运算；不引入 AMP 或 TF32 的隐式精度变化。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_model(device):
    model = CataProTokenRegressor(
        ResidueSubstrateCrossAttention(AttentionConfig(8, 0.0, 0.0, True)),
        MaskedMeanKcatHead(dropout=0.0, linear_bias=True),
    )
    return model.to(device=device, dtype=torch.float32)


def make_optimizer(model):
    # 所有下游参数参与 Adam，包括 bias 和 BN；这不是 AdamW 分组配方。
    return torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)


def make_loader(dataset, split, workers=8):
    from data import collate_tokens
    generator = torch.Generator().manual_seed(668)
    common = dict(num_workers=workers, collate_fn=collate_tokens,
                  generator=generator, worker_init_fn=seed_worker, pin_memory=True)
    if split == "train":
        sampler = MergeSingletonBatchSampler(RandomSampler(dataset, generator=generator),
                                             batch_size=8, drop_last=False)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False, **common)


def device_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def forward(model, batch):
    pred = model(**{key: batch[key] for key in FEATURE_KEYS})["pred"]
    label = batch["label"]
    if pred.shape != label.shape or label.ndim != 1:
        raise RuntimeError("损失输入必须都是 [B]，禁止广播成 [B,B]")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(label).all()):
        raise RuntimeError("Nonfinite prediction/label")
    # 与作者调用完全同形：label[B] 和 pred.ravel()[B]。
    loss = author.rmse_loss(label, pred.ravel())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite RMSELoss")
    return pred, loss


def metrics(labels, predictions, batch_average_loss):
    pcc, scc, r2, rmse = author.evaluate(labels, predictions)
    result = dict(PCC=float(pcc), SCC=float(scc), R2=float(r2), RMSE=float(rmse),
                  MAE=float(np.mean(np.abs(predictions - labels))),
                  batch_average_RMSELoss=float(batch_average_loss))
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite aggregate metric")
    return result


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    all_ids, labels, predictions = [], [], []
    # 保持作者 total_loss += loss.cpu().numpy() 的逐批累计方式。
    total_loss = 0
    with torch.set_grad_enabled(training):
        for cpu_batch in loader:
            batch = device_batch(cpu_batch, device)
            pred, loss = forward(model, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            total_loss += loss.detach().cpu().numpy()
            labels.append(batch["label"].detach().cpu().numpy().ravel())
            predictions.append(pred.detach().cpu().numpy().ravel())
            all_ids.extend(str(x) for x in cpu_batch["row_uid"])
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Repeated row IDs in an epoch")
    return dict(row_uid=np.asarray(all_ids), labels=labels, predictions=predictions,
                metrics=metrics(labels, predictions, total_loss / len(loader)))


def check_eval_identity(result, dataset):
    ids = np.asarray([str(row["row_uid"]) for row in dataset.rows])
    labels = np.asarray([row["label"] for row in dataset.rows], dtype=np.float32)
    if not np.array_equal(result["row_uid"], ids) or not np.array_equal(result["labels"], labels):
        raise RuntimeError("Evaluation row identity/order/labels changed")


def save_predictions(path, result):
    atomic_npz(path, **{name: result[name] for name in ("row_uid", "labels", "predictions")})


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def preflight_batches(dataset):
    from data import collate_tokens
    protein = np.asarray(dataset.protein_lengths)
    smiles = np.asarray(dataset.molt5_lengths)
    if len(dataset) < 9 or len(protein) != len(dataset) or len(smiles) != len(dataset):
        raise RuntimeError("Invalid train lengths for preflight")
    longest_p = np.argsort(-protein, kind="stable")[:8].tolist()
    longest_s = np.argsort(-smiles, kind="stable")[:8].tolist()
    # 9 条真实 train 行中同时包含最大 L 和最大 M，使 padding 张量达到训练上界。
    mixed = list(dict.fromkeys([int(protein.argmax()), int(smiles.argmax())] +
                              np.argsort(-(protein * smiles), kind="stable").tolist()))[:9]
    for name, indices in (("longest_protein_8", longest_p), ("longest_molt5_8", longest_s),
                          ("worst_padding_real_train_9", mixed)):
        yield name, collate_tokens([dataset[i] for i in indices])


def preflight_check(model, optimizer, batches, device, directory):
    """预检会更新临时模型；正式训练必须另建模型并重置 seed，不能接着训练。"""
    initial_sha = state_sha(model)
    checks = []
    last_batch = None
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for name, cpu_batch in batches:
        batch = device_batch(cpu_batch, device)
        model.train()
        before = {n: int(m.num_batches_tracked) for n, m in model.named_modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        pred, loss = forward(model, batch)
        after = {n: int(m.num_batches_tracked) for n, m in model.named_modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        if not before or any(after[n] - count != 1 for n, count in before.items()):
            raise RuntimeError("BN must update exactly once per train batch")
        optimizer.zero_grad()
        loss.backward()
        for n, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("Missing/nonfinite gradient: " + n)
        optimizer.step()
        if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise RuntimeError("Nonfinite updated state")
        checks.append(dict(name=name, shape_protein=list(batch["protein_tokens"].shape),
                           shape_molt5=list(batch["molt5_tokens"].shape),
                           rows=list(cpu_batch["row_uid"]), loss=float(loss.detach()),
                           bn_updates={n: after[n] - count for n, count in before.items()}))
        last_batch = batch
    if last_batch is None:
        raise RuntimeError("No preflight batches")
    model.eval()
    with torch.no_grad():
        expected, _ = forward(model, last_batch)
    checkpoint = Path(directory) / "roundtrip.pt"
    atomic_torch(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    restored = make_model(device)
    restored_optimizer = make_optimizer(restored)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"])
    if not tree_equal(model.state_dict(), restored.state_dict()):
        raise RuntimeError("Model state roundtrip mismatch")
    if not tree_equal(optimizer.state_dict(), restored_optimizer.state_dict()):
        raise RuntimeError("Adam state roundtrip mismatch")
    restored.eval()
    with torch.no_grad():
        actual, _ = forward(restored, last_batch)
    if not torch.equal(expected, actual):
        raise RuntimeError("Eval prediction roundtrip mismatch")
    result = dict(state="preflight_complete", initial_sha256=initial_sha, batches=checks,
                  model_roundtrip=True, adam_roundtrip=True, eval_roundtrip_exact=True,
                  discarded_optimizer_updates=True, test_forward=False)
    if str(device).startswith("cuda"):
        result["peak_reserved_MiB"] = torch.cuda.max_memory_reserved(device) / 2**20
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def source_identity():
    root = Path(__file__).resolve().parent
    names = ("train.py", "model.py", "author_utils.py", "tail_batch.py", "data.py",
             "prepare.py", "extract_tokens.py")
    result = {name: sha256(root / name) for name in names}
    if result["author_utils.py"] != AUTHOR_SHA:
        raise RuntimeError("Author utils bytes changed")
    return result


def input_identity(args):
    return {"prepared_manifest": sha256(args.prepared / "manifest.json"),
            "prepared_rows": sha256(args.prepared / "rows.json"),
            "prepared_maccs": sha256(args.prepared / "maccs.npy"),
            "protein_manifest": sha256(args.protein_cache / "manifest.json"),
            "molt5_manifest": sha256(args.molt5_cache / "manifest.json")}


def protocol(args):
    return dict(config=CONFIG, source_sha256=source_identity(), inputs=input_identity(args),
                torch_version=torch.__version__, numpy_version=np.__version__,
                scipy_version=scipy.__version__, python_version=sys.version,
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                device=args.device, test_policy="locked raw model; full valid replay before one test")


def dataset_for(args, split):
    from data import TokenDataset
    dataset = TokenDataset(args.prepared, args.protein_cache, args.molt5_cache, split)
    if len(dataset) != EXPECTED_ROWS[split]:
        raise RuntimeError("Wrong complete split count: " + split)
    return dataset


def run_preflight(args):
    output = args.output / "preflight"
    output.mkdir(parents=True, exist_ok=False)
    audit_protocol = protocol(args)
    atomic_json(output / "protocol.json", audit_protocol)
    dataset = dataset_for(args, "train")
    set_seed()
    model = make_model(args.device)
    result = preflight_check(model, make_optimizer(model), preflight_batches(dataset), args.device, output)
    result["protocol"] = audit_protocol
    atomic_json(output / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def verify_preflight(args, current):
    result = json.loads((args.output / "preflight" / "preflight.json").read_text(encoding="utf-8"))
    if result["state"] != "preflight_complete" or result["protocol"] != current:
        raise RuntimeError("Preflight is missing, failed, or no longer matches code/inputs/config/runtime")
    return result


def training(args):
    if any((args.output / name).exists() for name in ("protocol.json", "history.json", "best.pt", "status.json")):
        raise FileExistsError("Refusing to overwrite or resume an existing trial")
    current = protocol(args)
    passed = verify_preflight(args, current)
    train = dataset_for(args, "train")
    valid = dataset_for(args, "valid")
    # 预检进程产生的参数和优化器不复用；正式初始化重新 seed668。
    set_seed()
    model = make_model(args.device)
    initial = state_sha(model)
    if initial != passed["initial_sha256"]:
        raise RuntimeError("Formal initial state differs from preflight initial state")
    current["initial_state_sha256"] = initial
    current["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    atomic_json(args.output / "protocol.json", current)
    atomic_json(args.output / "status.json", dict(state="training", completed_epochs=0))
    optimizer = make_optimizer(model)
    loaders = {"train": make_loader(train, "train"), "valid": make_loader(valid, "valid")}
    stopper = author.EarlyStopping(20, 0.001)
    history, best_epoch = [], None
    for epoch in range(150):
        started = time.time()
        tr = run_epoch(model, loaders["train"], args.device, optimizer)
        # 只检查数值健康，不裁剪梯度、不改变优化器或训练配方。
        if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
            raise RuntimeError("Nonfinite model state after training epoch")
        for state in optimizer.state.values():
            if any(isinstance(t, torch.Tensor) and not bool(torch.isfinite(t).all()) for t in state.values()):
                raise RuntimeError("Nonfinite Adam state after training epoch")
        va = run_epoch(model, loaders["valid"], args.device)
        check_eval_identity(va, valid)
        if set(tr["row_uid"]) != {str(row["row_uid"]) for row in train.rows}:
            raise RuntimeError("Training epoch did not cover every original row exactly once")
        is_best, stop = stopper.check(epoch, va["metrics"]["batch_average_RMSELoss"])
        if is_best:
            best_epoch = epoch
            atomic_torch(args.output / "best.pt", dict(model=model.state_dict(),
                         optimizer=optimizer.state_dict(), epoch_zero_based=epoch,
                         initial_state_sha256=initial))
            save_predictions(args.output / "valid_best.npz", va)
        record = dict(epoch_zero_based=epoch, epoch=epoch + 1, train=tr["metrics"],
                      valid=va["metrics"], is_best=bool(is_best), early_stop=bool(stop),
                      counter=stopper.count_epoch, elapsed_seconds=time.time() - started)
        history.append(record)
        atomic_json(args.output / "history.json", history)
        atomic_json(args.output / "status.json", dict(state="training", completed_epochs=len(history),
                    best_epoch=best_epoch + 1, counter=stopper.count_epoch))
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if stop:
            break
    if best_epoch is None:
        raise RuntimeError("No selected checkpoint")
    lock = dict(best_epoch_zero_based=best_epoch, best_epoch=best_epoch + 1,
                checkpoint_sha256=sha256(args.output / "best.pt"),
                valid_predictions_sha256=sha256(args.output / "valid_best.npz"),
                history_sha256=sha256(args.output / "history.json"),
                protocol_sha256=sha256(args.output / "protocol.json"),
                completed_epochs=len(history), test_evaluated=False,
                selection=CONFIG["selection"])
    atomic_json(args.output / "selection_lock.json", lock)
    atomic_json(args.output / "status.json", dict(state="complete_locked", **lock))
    print(json.dumps(lock), flush=True)


def verify_selection(output):
    lock = json.loads((output / "selection_lock.json").read_text(encoding="utf-8"))
    for filename, key in (("best.pt", "checkpoint_sha256"), ("valid_best.npz", "valid_predictions_sha256"),
                          ("history.json", "history_sha256"), ("protocol.json", "protocol_sha256")):
        if sha256(output / filename) != lock[key]:
            raise RuntimeError("Selected trial changed: " + filename)
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    stopper, selected = author.EarlyStopping(20, 0.001), None
    for epoch, record in enumerate(history):
        if record["epoch_zero_based"] != epoch or record["epoch"] != epoch + 1:
            raise RuntimeError("Nonconsecutive history")
        best, stop = stopper.check(epoch, record["valid"]["batch_average_RMSELoss"])
        if (best, stop, stopper.count_epoch) != (record["is_best"], record["early_stop"], record["counter"]):
            raise RuntimeError("Selection history replay mismatch")
        if best:
            selected = epoch
        if stop and epoch != len(history) - 1:
            raise RuntimeError("History continued after stopping")
    if selected != lock["best_epoch_zero_based"] or (not stopper.stop and len(history) != 150):
        raise RuntimeError("Training has not naturally finished under the approved rule")
    if len(history) != lock["completed_epochs"]:
        raise RuntimeError("Wrong locked epoch count")
    return lock, history


def evaluate_test(args):
    lock, history = verify_selection(args.output)
    frozen = json.loads((args.output / "protocol.json").read_text(encoding="utf-8"))
    current = protocol(args)
    if any(current[key] != frozen[key] for key in current):
        raise RuntimeError("Evaluation code/inputs/config/runtime differ from locked training")
    destination = args.output / "test"
    destination.mkdir(exist_ok=False)  # 不覆盖、不滚动重复 test；失败也留下证据。
    atomic_json(destination / "status.json", dict(state="validating_before_test", test_forward=False))
    set_seed()
    model = make_model(args.device)
    payload = torch.load(args.output / "best.pt", map_location=args.device, weights_only=False)
    if payload["epoch_zero_based"] != lock["best_epoch_zero_based"]:
        raise RuntimeError("Checkpoint epoch mismatch")
    model.load_state_dict(payload["model"], strict=True)
    valid = dataset_for(args, "valid")
    replay = run_epoch(model, make_loader(valid, "valid"), args.device)
    check_eval_identity(replay, valid)
    with np.load(args.output / "valid_best.npz", allow_pickle=False) as reference:
        for key in ("row_uid", "labels"):
            if not np.array_equal(replay[key], reference[key]):
                raise RuntimeError("Valid replay identity mismatch")
        difference = float(np.max(np.abs(replay["predictions"] - reference["predictions"])))
        # 同设备/精度/批量的完整逐值重放；仅接受浮点容差，不用汇总指标掩盖逐行问题。
        if not np.allclose(replay["predictions"], reference["predictions"], rtol=1e-6, atol=1e-6):
            raise RuntimeError("Full valid per-value replay mismatch; test prohibited")
    expected = history[lock["best_epoch_zero_based"]]["valid"]
    if any(not np.isclose(replay["metrics"][key], expected[key], rtol=1e-6, atol=1e-7) for key in expected):
        raise RuntimeError("Valid metric replay mismatch; test prohibited")
    atomic_json(destination / "valid_replay.json", dict(passed=True, rows=len(valid),
                max_prediction_abs_difference=difference, metrics=replay["metrics"]))
    save_predictions(destination / "valid_replay.npz", replay)
    if sha256(args.output / "best.pt") != lock["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during replay")
    test = dataset_for(args, "test")
    atomic_json(destination / "status.json", dict(state="test_running", test_forward=True))
    result = run_epoch(model, make_loader(test, "test"), args.device)
    check_eval_identity(result, test)
    save_predictions(destination / "test_predictions.npz", result)
    verify_selection(args.output)
    summary = dict(state="complete", test_rows=len(test), test_metrics=result["metrics"],
                   selected_epoch=lock["best_epoch"], checkpoint_sha256=lock["checkpoint_sha256"],
                   valid_replay_passed=True, test_evaluations=1,
                   prediction_sha256=sha256(destination / "test_predictions.npz"))
    atomic_json(destination / "summary.json", summary)
    atomic_json(destination / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--protein-cache", type=Path, required=True)
    parser.add_argument("--molt5-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "train", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Formal trial requires an allocated CUDA GPU")
    if os.environ.get("PYTHONHASHSEED") != "668":
        raise RuntimeError("Set PYTHONHASHSEED=668 before starting the Python process")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Formal preflight/training/test must run in sbatch")
    args.output.mkdir(parents=True, exist_ok=True)
    {"preflight": run_preflight, "train": training, "test": evaluate_test}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
