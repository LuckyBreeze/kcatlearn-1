"""冻结编码器的逐 token 特征提取；本脚本不训练，也不自行提交作业。

输入 JSON::

    {"kind": "protein", "expected_checkpoint_files": {"config.json": "<sha256>", ...},
     "items": [{"key": "<sha256(input.encode('utf-8'))>", "input": "ACDX",
                "original_lengths": [4]}]}

protein 的 input 必须已由上游按作者规则截断和替换 U/Z/O/B；此处不再修改。
MolT5 的 input 是原始 SMILES。original_lengths 可保留裁剪前的多个原序列长度。
输出为 <key>.npy 和 manifest.json；只有 status=complete 才能进入训练。
恢复时必须逐项核验来源、代码、权重、运行环境以及已有数组，不接纳来历不明的缓存。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import uuid


REQUIRED_CHECKPOINT_FILES = (
    "pytorch_model.bin", "config.json", "tokenizer_config.json",
    "special_tokens_map.json", "spiece.model",
)
OPTIONAL_CHECKPOINT_FILES = ("tokenizer.json",)
DIMENSIONS = {"protein": 1024, "molt5": 768}
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"校验过程中源文件变化：{path}")
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: Path) -> dict:
    """仅读取批准的必要文件；不递归扫描目录中的其它权重、缓存或说明书。"""
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint 不是目录：{checkpoint}")
    # T5Tokenizer 会读取此可选文件；未经纳入审批的额外 token 表不能悄悄生效。
    if (checkpoint / "added_tokens.json").exists():
        raise ValueError("发现未纳入批准指纹的 added_tokens.json，请先核实 tokenizer 来源")
    names = list(REQUIRED_CHECKPOINT_FILES)
    names.extend(name for name in OPTIONAL_CHECKPOINT_FILES if (checkpoint / name).exists())
    result = {}
    for name in sorted(names):
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要 checkpoint 文件：{path}")
        result[name] = {"sha256": file_sha(path), "bytes": path.stat().st_size}
    return result


def read_items(path: Path, kind: str) -> tuple[list[dict], str, dict]:
    source = path.read_bytes()
    data = json.loads(source.decode("utf-8"))
    if not isinstance(data, dict) or data.get("kind", kind) != kind:
        raise ValueError("items 顶层必须是对象，kind 必须与命令一致")
    expected = data.get("expected_checkpoint_files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("items 必须包含事先核准的 expected_checkpoint_files 文件名→SHA256")
    if not all(isinstance(v, str) and SHA_PATTERN.fullmatch(v) for v in expected.values()):
        raise ValueError("expected_checkpoint_files 中存在非法 SHA256")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items 必须为非空列表")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("input"), str):
            raise ValueError("每条输入必须含字符串 input")
        value, key = item["input"], item.get("key")
        if not value or key != hashlib.sha256(value.encode("utf-8")).hexdigest():
            raise ValueError("输入为空或 key 不是精确 UTF-8 input 的 SHA256")
        if key in seen:
            raise ValueError(f"重复 key；请在上游显式去重：{key}")
        seen.add(key)
        if kind == "protein":
            if not re.fullmatch(r"[A-Z]+", value) or any(c in value for c in "UZOB"):
                raise ValueError(f"蛋白须为已替换 U/Z/O/B 的无空格大写序列：{key}")
            if len(value) > 1000:
                raise ValueError(f"蛋白超过批准的 1000 残基；本脚本不会自动裁剪：{key}")
        lengths = item.get("original_lengths")
        if lengths is not None:
            if (not isinstance(lengths, list) or not lengths
                    or any(type(n) is not int or n < len(value) for n in lengths)):
                raise ValueError(f"original_lengths 必须为非空整数列表且不小于输入长度：{key}")
            if kind == "protein" and any(min(n, 1000) != len(value) for n in lengths):
                raise ValueError(f"original_lengths 与批准的首尾 500 规则不符：{key}")
            if kind == "molt5" and any(n != len(value) for n in lengths):
                raise ValueError(f"MolT5 不得截断或替换原始 SMILES：{key}")
            item["original_lengths"] = sorted(set(lengths))
    return items, hashlib.sha256(source).hexdigest(), expected


def tokenize_item(tokenizer, item: dict, kind: str) -> dict:
    """只排除末尾 EOS 和 padding。UNK 属于实际内容，绝不能按特殊 ID 总表过滤。"""
    text = " ".join(item["input"]) if kind == "protein" else item["input"]
    content = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    encoded = tokenizer(text, add_special_tokens=True, padding=False, truncation=False,
                        return_attention_mask=True)
    ids, mask = list(encoded["input_ids"]), list(encoded["attention_mask"])
    if len(ids) != len(mask) or any(flag not in (0, 1) for flag in mask):
        raise ValueError(f"tokenizer 返回非法 attention_mask：{item['key']}")
    positions = [i for i, flag in enumerate(mask) if flag]
    if not content or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError(f"无内容 token 或未定义 EOS/PAD：{item['key']}")
    if any(t in (tokenizer.eos_token_id, tokenizer.pad_token_id) for t in content):
        raise ValueError(f"原始内容中出现 EOS/PAD token，不能静默删除：{item['key']}")
    # 独立编码两次以证实没有截断：有效编码必须恰为全部内容 + 一个 EOS。
    if [ids[i] for i in positions] != list(content) + [tokenizer.eos_token_id]:
        raise ValueError(f"实际编码不等于完整内容加单个末尾 EOS：{item['key']}")
    if positions != list(range(len(positions))):
        raise ValueError(f"非右侧 padding 不符合本次提取约定：{item['key']}")
    if any(ids[i] != tokenizer.pad_token_id for i, flag in enumerate(mask) if not flag):
        raise ValueError(f"被 mask 排除的位置不是 PAD：{item['key']}")
    if kind == "protein" and len(content) != len(item["input"]):
        raise ValueError(f"蛋白 token 数不等于残基数：{item['key']}")
    return {
        "ids": ids, "mask": mask, "positions": positions[:-1],
        "metadata": {
            "key": item["key"], "file": item["key"] + ".npy",
            "shape": [len(content), DIMENSIONS[kind]], "dtype": "float32",
            "input_length": len(item["input"]),
            "original_lengths": item.get("original_lengths"),
            "content_token_length": len(content), "token_ids_sha256": canonical_sha(content),
            "unknown_token_count": sum(t == tokenizer.unk_token_id for t in content),
        },
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(output: Path):
    """OS 文件锁会随进程退出释放；被 sbatch 终止后无需猜测或删除陈旧 PID 锁。"""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extract.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_array(path: Path, record: dict, metadata: dict) -> None:
    import numpy as np
    if any(record.get(k) != v for k, v in metadata.items()):
        raise ValueError(f"缓存记录与当前输入/token 对不上：{path.name}")
    if (not isinstance(record.get("sha256"), str)
            or not SHA_PATTERN.fullmatch(record["sha256"]) or not path.is_file()):
        raise ValueError(f"缓存文件或 SHA256 缺失：{path}")
    if file_sha(path) != record["sha256"]:
        raise ValueError(f"缓存 SHA256 不一致：{path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if (list(array.shape) != metadata["shape"] or array.dtype != np.dtype("float32")
            or not np.isfinite(array).all()):
        raise ValueError(f"缓存 shape/dtype/有限性检查失败：{path}")


def save_manifest(output: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now()
    atomic_json(output / "manifest.json", manifest)


def open_manifest(output: Path, provenance: dict, metadata: dict[str, dict]) -> dict:
    """逐条 receipt 核验后恢复；避免每个样本都重写全量 manifest 的二次方开销。"""
    path = output / "manifest.json"
    receipts = output / ".receipts"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("provenance") != provenance
                or manifest.get("kind") != provenance["kind"]):
            raise ValueError("缓存来源/权重/代码/运行环境不完全一致，拒绝续写")
        if not isinstance(manifest.get("records"), dict):
            raise ValueError("manifest.records 格式错误")
    else:
        if list(output.glob("*.npy")) or (receipts.exists() and list(receipts.glob("*.json"))):
            raise ValueError("存在无 manifest 的 npy/receipt，拒绝接纳来历不明的缓存")
        manifest = {"schema_version": 1, "kind": provenance["kind"], "status": "running",
                    "provenance": provenance, "records": {}, "pending": None,
                    "created_at": utc_now()}

    receipts.mkdir(exist_ok=True)
    provenance_sha = canonical_sha(provenance)
    recovered = {}
    for receipt_path in sorted(receipts.glob("*.json")):
        key = receipt_path.stem
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if key not in metadata or receipt.get("provenance_sha256") != provenance_sha:
            raise ValueError(f"receipt 的输入或来源指纹不匹配：{receipt_path}")
        record = receipt["record"]
        final = output / (key + ".npy")
        temporary_name = receipt.get("temporary_file")
        if temporary_name is not None:
            if not re.fullmatch(r"\." + key + r"\.[0-9a-f]{32}\.tmp", temporary_name):
                raise ValueError("receipt 的临时文件名非法")
            temporary = output / temporary_name
            if final.exists() == temporary.exists():
                raise ValueError("未完成 receipt 必须恰有一个可核验的临时或正式数组")
            candidate = final if final.exists() else temporary
            verify_array(candidate, record, metadata[key])
            if candidate == temporary:
                os.replace(temporary, final)
            atomic_json(receipt_path, {"provenance_sha256": provenance_sha, "record": record})
        else:
            verify_array(final, record, metadata[key])
        recovered[key] = record
    if any(recovered.get(key) != record for key, record in manifest["records"].items()):
        raise ValueError("manifest 已提交记录缺少匹配 receipt，拒绝恢复")
    manifest["records"] = recovered
    manifest["pending"] = None
    actual = {p.name for p in output.glob("*.npy")}
    expected = {key + ".npy" for key in manifest["records"]}
    if actual != expected:
        raise ValueError("目录含未纳入 manifest 的 npy 或缺失已提交数组，拒绝恢复")
    manifest["status"] = "running"
    manifest.pop("error", None)
    save_manifest(output, manifest)
    return manifest


def commit_array(output: Path, manifest: dict, array, metadata: dict) -> None:
    import numpy as np
    key = metadata["key"]
    receipt_path = output / ".receipts" / (key + ".json")
    if (key in manifest["records"] or (output / metadata["file"]).exists()
            or receipt_path.exists()):
        raise ValueError(f"禁止覆盖已存在的特征：{key}")
    temporary = output / ("." + key + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    record = {**metadata, "sha256": file_sha(temporary)}
    verify_array(temporary, record, metadata)
    receipt = {"provenance_sha256": canonical_sha(manifest["provenance"]), "record": record}
    atomic_json(receipt_path, {**receipt, "temporary_file": temporary.name})
    os.replace(temporary, output / metadata["file"])
    atomic_json(receipt_path, receipt)
    manifest["records"][key] = record
    if len(manifest["records"]) % 50 == 0:
        save_manifest(output, manifest)


def configure_runtime(device_name: str) -> tuple[object, object, dict]:
    import numpy as np
    import sentencepiece
    import torch
    import transformers
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("仅支持显式 cpu 或 cuda 设备；不自动改变计算后端")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但 CUDA 不可用；不会退回 CPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = {
        "python": platform.python_version(), "platform": platform.system(),
        "numpy": np.__version__, "torch": str(torch.__version__),
        "transformers": transformers.__version__, "sentencepiece": sentencepiece.__version__,
        "cuda_runtime": torch.version.cuda, "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "dtype": "float32", "tf32": False, "batch_size": 1,
    }
    return torch, device, runtime


def extract_one(model, torch, device, tokenized: dict):
    import numpy as np
    ids = torch.tensor([tokenized["ids"]], dtype=torch.long, device=device)
    mask = torch.tensor([tokenized["mask"]], dtype=torch.long, device=device)
    positions = torch.tensor(tokenized["positions"], dtype=torch.long, device=device)
    # 不启用 autocast；输入和模型均固定 FP32；不计算梯度、不修改参数。
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        if hidden.dtype != torch.float32 or tuple(hidden.shape[:2]) != tuple(ids.shape):
            raise ValueError("编码器输出精度或序列形状不符合约定")
        array = hidden.index_select(1, positions)[0].cpu().numpy().copy()
    expected = tokenized["metadata"]["shape"]
    if list(array.shape) != expected or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"编码器输出 shape/dtype/有限性失败：{tokenized['metadata']['key']}")
    return array


def run(args) -> None:
    import numpy as np
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if checkpoint == output or checkpoint in output.parents or output in checkpoint.parents:
        raise ValueError("checkpoint 与 output 目录不能相互包含")
    items, source_sha, expected_files = read_items(args.items, args.kind)
    fingerprint = checkpoint_fingerprint(checkpoint)
    if {name: rec["sha256"] for name, rec in fingerprint.items()} != expected_files:
        raise ValueError("checkpoint 实际文件清单/SHA256 与预先批准的指纹不一致")
    code_sha = file_sha(Path(__file__))
    torch, device, runtime = configure_runtime(args.device)
    from transformers import T5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                            **({"do_lower_case": False} if args.kind == "protein" else {}))
    tokenized = {item["key"]: tokenize_item(tokenizer, item, args.kind) for item in items}
    metadata = {key: rec["metadata"] for key, rec in tokenized.items()}
    order = sorted(tokenized, key=lambda key: (-metadata[key]["content_token_length"], key))
    provenance = {
        "kind": args.kind, "source_items_sha256": source_sha, "extract_code_sha256": code_sha,
        "checkpoint_files": fingerprint, "runtime": runtime, "items_count": len(items),
        "token_metadata_sha256": canonical_sha(metadata),
        "token_policy": "all_content_including_UNK; remove_only_trailing_EOS_and_padding; no_truncation",
    }
    with output_lock(output):
        manifest = open_manifest(output, provenance, metadata)
        try:
            preflight = manifest.get("longest_preflight", {})
            preflight_valid = (preflight.get("passed") is True and preflight.get("key") == order[0]
                               and preflight.get("shape") == metadata[order[0]]["shape"])
            if len(manifest["records"]) < len(items) or not preflight_valid:
                # 强制读取已校验的 pytorch_model.bin，不让同目录 safetensors 自动抢先加载。
                model, loading = T5EncoderModel.from_pretrained(
                    str(checkpoint), local_files_only=True, use_safetensors=False,
                    torch_dtype=torch.float32, output_loading_info=True,
                )
                if (loading.get("missing_keys") or loading.get("mismatched_keys")
                        or loading.get("error_msgs")):
                    raise ValueError(f"编码器权重未完整加载，禁止随机补齐：{loading}")
                unexpected = loading.get("unexpected_keys", [])
                if any(not key.startswith(("decoder.", "lm_head.")) for key in unexpected):
                    raise ValueError(f"发现未预期的 checkpoint 权重：{unexpected}")
                if model.config.d_model != DIMENSIONS[args.kind]:
                    raise ValueError("checkpoint d_model 与批准的特征宽度不一致")
                model.requires_grad_(False)
                model.eval().to(device=device, dtype=torch.float32)
                if model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
                    raise ValueError("模型未完全冻结为 eval/FP32")
                # 加载后再次验 SHA，避免启动校验与实际读取之间被更换文件。
                if checkpoint_fingerprint(checkpoint) != fingerprint:
                    raise ValueError("checkpoint 在加载期间发生变化")
                manifest["model_loading_info"] = loading
                longest = order[0]
                first = extract_one(model, torch, device, tokenized[longest])
                if longest in manifest["records"]:
                    old = np.load(output / metadata[longest]["file"], allow_pickle=False)
                    if not np.array_equal(first, old):
                        raise ValueError("续跑时最长样本复算未逐值重现旧缓存，禁止混合来源")
                else:
                    commit_array(output, manifest, first, metadata[longest])
                manifest["longest_preflight"] = {
                    "key": longest, "shape": list(first.shape), "passed": True,
                    "completed_at": utc_now(),
                }
                save_manifest(output, manifest)
                print(json.dumps({"event": "longest_preflight_passed", "kind": args.kind,
                                  "shape": list(first.shape)}, ensure_ascii=False), flush=True)
                for key in order:
                    if key in manifest["records"]:
                        continue
                    array = extract_one(model, torch, device, tokenized[key])
                    commit_array(output, manifest, array, metadata[key])
                    count = len(manifest["records"])
                    if count % 50 == 0 or count == len(items):
                        print(json.dumps({"kind": args.kind, "completed": count,
                                          "total": len(items)}), flush=True)
            if file_sha(args.items) != source_sha or file_sha(Path(__file__)) != code_sha:
                raise ValueError("提取期间输入文件或提取代码发生变化")
            manifest["status"] = "complete"
            manifest["completed_at"] = utc_now()
            save_manifest(output, manifest)
            print(json.dumps({"status": "complete", "kind": args.kind,
                              "records": len(manifest["records"]),
                              "manifest": str(output / "manifest.json")}), flush=True)
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            save_manifest(output, manifest)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(DIMENSIONS))
    parser.add_argument("--items", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda", help="默认 cuda；不可用则停止，不自动回退")
    parser.add_argument("--fingerprint-checkpoint", type=Path,
                        help="只读计算必要文件指纹，不加载模型、不提取特征")
    args = parser.parse_args()
    if args.fingerprint_checkpoint is not None:
        fingerprints = checkpoint_fingerprint(args.fingerprint_checkpoint.resolve())
        print(json.dumps({"expected_checkpoint_files": {k: v["sha256"] for k, v in fingerprints.items()},
                          "files": fingerprints}, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if any(getattr(args, name) is None for name in ("kind", "items", "checkpoint", "output")):
        parser.error("提取需要同时指定 --kind --items --checkpoint --output")
    run(args)


if __name__ == "__main__":
    main()
