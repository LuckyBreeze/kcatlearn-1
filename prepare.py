"""把固定 CataPro 行/划分转换成 token 查表输入，不重划分、不删样本。"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = {
    'inputs/feature_store.json': 'd489a8702bd58da9af8e117da88f9ab2e18a1806af218290774eb9ad43860d8d',
    'inputs/rowspec.json': 'e92f55cd7881a2a6615ebabf189bd421ced3e448a357881024501813e97667bc',
    'inputs/rows.csv': 'a0f15c13c715006513040387a5de600fdffc42481af5830277e683d5a6dc1710',
    'inputs/feature1959.npy': '18ca9ea49e78b338e6d64040874c2048b4fde07457a595fd77114fd91baaba86',
    'assignments.csv': '721cc57c15ea1e87ea289118868f9d5b9007a9aaa213e45a4d76844e13d08138',
}
COUNTS = {'train': 21585, 'valid': 3037, 'test': 3036}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def key(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def protein_input(raw):
    if not isinstance(raw, str) or not re.fullmatch('[A-Z]+', raw):
        raise ValueError('Invalid original protein sequence')
    # 用户确认的作者规则：先首尾裁剪，再替换四种非常规氨基酸。
    cropped = raw[:500] + raw[-500:] if len(raw) > 1000 else raw
    return re.sub('[UZOB]', 'X', cropped)

def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def prepare(base, output, checkpoints):
    if output.exists():
        raise RuntimeError('Prepared directory already exists; refusing overwrite')
    for rel, expected in EXPECTED.items():
        if sha(base / rel) != expected:
            raise RuntimeError('Frozen input changed: ' + rel)
    meta = json.loads((base / 'inputs/feature_store.json').read_text())
    if meta['feature_blocks'] != [dict(name='ProtT5', width=1024), dict(name='MolT5', width=768), dict(name='MACCS', width=167)]:
        raise RuntimeError('Unexpected original feature order')
    frame = pd.read_csv(base / 'inputs/rows.csv', dtype={'author_row_id': str}, keep_default_na=False, float_precision='round_trip')
    features = np.load(base / 'inputs/feature1959.npy', allow_pickle=False, mmap_mode='r')
    if len(frame) != 27658 or frame.author_row_id.duplicated().any() or features.shape != (27658, 1959) or features.dtype != np.float32:
        raise RuntimeError('Wrong population/feature shape/dtype')
    raw = frame.kcat_s_minus1.to_numpy(dtype=np.float64)
    labels = frame.log10_kcat.to_numpy(dtype=np.float64)
    if not np.isfinite(features).all() or not np.isfinite(labels).all() or not (raw > 0).all() or not np.allclose(labels, np.log10(raw), atol=1e-12, rtol=0):
        raise RuntimeError('Nonfinite feature or invalid target')
    indices = {sid: i for i, sid in enumerate(frame.author_row_id)}
    with (base / 'assignments.csv').open() as f:
        membership = {r['balance_row_uid'].removeprefix('catapro:'): r['new_split'] for r in csv.DictReader(f)}
    spec = json.loads((base / 'inputs/rowspec.json').read_text())
    if spec['meta']['scenario'] != 'cluster40' or set(spec['eval']) != {'test'}:
        raise RuntimeError('Wrong scenario')
    rows, maccs, proteins, smiles, seen, p_splits = [], [], {}, {}, set(), {}
    for split, pairs in [('train', spec['train']), ('valid', spec['valid']), ('test', spec['eval']['test'])]:
        if len(pairs) != COUNTS[split]:
            raise RuntimeError('Wrong split count')
        for source, sid in pairs:
            sid = str(sid)
            if source != 'catapro' or sid in seen or membership[sid] != split:
                raise RuntimeError('Wrong/repeated split membership')
            seen.add(sid)
            i = indices[sid]
            row = frame.iloc[i]
            p, s = protein_input(row.sequence), row.smiles
            if not isinstance(s, str) or not s:
                raise RuntimeError('Empty SMILES')
            pk, sk = key(p), key(s)
            rec = proteins.setdefault(pk, dict(key=pk, input=p, original_lengths=[]))
            if rec['input'] != p:
                raise RuntimeError('SHA collision')
            rec['original_lengths'].append(len(row.sequence))
            smiles.setdefault(sk, dict(key=sk, input=s))
            p_splits.setdefault(pk, set()).add(split)
            rows.append(dict(row_uid=sid, split=split, protein_key=pk, smiles_key=sk, label=float(labels[i])))
            # 只复用原167位MACCS，不把池化向量冒充token。
            maccs.append(np.array(features[i, 1792:1959], copy=True))
    if seen != set(indices) or len(proteins) != 13192 or len(smiles) != 3302 or any(len(v) != 1 for v in p_splits.values()):
        raise RuntimeError('Population/unique count/processed-sequence split mismatch')
    for rec in proteins.values():
        rec['original_lengths'] = sorted(set(rec['original_lengths']))
    output.mkdir(parents=True)
    dump(output / 'rows.json', rows)
    np.save(output / 'maccs.npy', np.stack(maccs).astype(np.float32), allow_pickle=False)
    for kind, table in [('protein', proteins), ('molt5', smiles)]:
        dump(output / (kind + '_items.json'), dict(kind=kind, items=list(table.values()), expected_checkpoint_files=checkpoints[kind]['files']))
    files = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    dump(output / 'manifest.json', dict(schema='catapro_token_trial_inputs_v1', status='complete', seed=668, scenario='cluster40', counts=COUNTS, unique_proteins=len(proteins), unique_smiles=len(smiles), source_base=str(base), source_sha256=EXPECTED, files=files, preprocessing='ProtT5 >1000:first500+last500; UZOB->X; SMILES unchanged; no token truncation', checkpoint_identity=checkpoints))
    print(json.dumps(dict(status='complete', counts=COUNTS, proteins=len(proteins), smiles=len(smiles))), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.output, json.loads(args.checkpoints.read_text()))
