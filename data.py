"""严格读取已审计token缓存，按batch动态补零；mask只由真实长度生成。"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from prepare import sha, key, EXPECTED, COUNTS

_VERIFIED = {}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_cache(root, items_path, kind):
    meta = read_json(root / 'manifest.json')
    items_document = read_json(items_path)
    items = {r['key']: r for r in items_document['items']}
    if len(items) != len(items_document['items']) or any(key(r['input']) != k for k, r in items.items()):
        raise RuntimeError('Invalid input key')
    if meta.get('schema_version') != 1 or meta.get('status') != 'complete' or meta.get('kind') != kind:
        raise RuntimeError('Token cache is incomplete/wrong kind')
    provenance = meta['provenance']
    if provenance['source_items_sha256'] != sha(items_path) or provenance['extract_code_sha256'] != sha(Path(__file__).with_name('extract_tokens.py')):
        raise RuntimeError('Token cache input/extractor mismatch')
    if {name: entry['sha256'] for name, entry in provenance['checkpoint_files'].items()} != items_document['expected_checkpoint_files']:
        raise RuntimeError('Token checkpoint identity mismatch')
    records = meta['records']
    if set(records) != set(items):
        raise RuntimeError('Token cache coverage is not exact')
    width = 1024 if kind == 'protein' else 768
    for k, rec in records.items():
        if rec['file'] != k + '.npy' or rec['key'] != k or rec['dtype'] != 'float32':
            raise RuntimeError('Invalid cache record')
        p = root / rec['file']
        if sha(p) != rec['sha256']:
            raise RuntimeError('Cache file SHA mismatch: ' + k)
        array = np.load(p, allow_pickle=False, mmap_mode='r')
        length = rec['content_token_length']
        if type(length) is not int or length < 1 or rec['input_length'] != len(items[k]['input']):
            raise RuntimeError('Invalid input/token length')
        if kind == 'protein' and length != len(items[k]['input']):
            raise RuntimeError('Protein length lost before interaction')
        if tuple(rec['shape']) != (length, width) or array.shape != (length, width) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise RuntimeError('Cache shape/dtype/finiteness mismatch')
    return records

def verified_inputs(prepared, protein_cache, molt5_cache):
    signature = tuple((str(p.resolve()), sha(p / 'manifest.json')) for p in (prepared, protein_cache, molt5_cache))
    if signature in _VERIFIED:
        return _VERIFIED[signature]
    meta = read_json(prepared / 'manifest.json')
    if meta.get('schema') != 'catapro_token_trial_inputs_v1' or meta.get('status') != 'complete' or meta['source_sha256'] != EXPECTED or meta['counts'] != COUNTS or meta['scenario'] != 'cluster40' or meta['seed'] != 668:
        raise RuntimeError('Wrong frozen experiment inputs')
    for name in ['rows.json', 'maccs.npy', 'protein_items.json', 'molt5_items.json']:
        if sha(prepared / name) != meta['files'][name]:
            raise RuntimeError('Prepared file changed: ' + name)
    rows = read_json(prepared / 'rows.json')
    if len(rows) != sum(COUNTS.values()) or len({r['row_uid'] for r in rows}) != len(rows):
        raise RuntimeError('Wrong/repeated input population')
    for split, count in COUNTS.items():
        if sum(r['split'] == split for r in rows) != count:
            raise RuntimeError('Wrong split population')
    maccs = np.load(prepared / 'maccs.npy', allow_pickle=False)
    if maccs.dtype != np.float32 or maccs.shape != (len(rows), 167) or not np.isfinite(maccs).all():
        raise RuntimeError('Invalid original MACCS')
    protein = verify_cache(protein_cache, prepared / 'protein_items.json', 'protein')
    molt5 = verify_cache(molt5_cache, prepared / 'molt5_items.json', 'molt5')
    for r in rows:
        if r['protein_key'] not in protein or r['smiles_key'] not in molt5 or not np.isfinite(r['label']):
            raise RuntimeError('Missing token or invalid label')
    result = rows, maccs, protein, molt5
    _VERIFIED[signature] = result
    return result

class TokenDataset(Dataset):
    def __init__(self, prepared, protein_cache, molt5_cache, split):
        if split not in COUNTS:
            raise ValueError('Unknown split')
        prepared, protein_cache, molt5_cache = map(Path, (prepared, protein_cache, molt5_cache))
        rows, maccs, protein, molt5 = verified_inputs(prepared, protein_cache, molt5_cache)
        ix = [i for i, r in enumerate(rows) if r['split'] == split]
        self.rows = [rows[i] for i in ix]
        self.maccs = np.array(maccs[ix], copy=True)
        self.protein_cache, self.molt5_cache = protein_cache, molt5_cache
        self.protein_lengths = [protein[r['protein_key']]['content_token_length'] for r in self.rows]
        self.molt5_lengths = [molt5[r['smiles_key']]['content_token_length'] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        p = np.load(self.protein_cache / (row['protein_key'] + '.npy'), allow_pickle=False)
        m = np.load(self.molt5_cache / (row['smiles_key'] + '.npy'), allow_pickle=False)
        if p.shape != (self.protein_lengths[index], 1024) or m.shape != (self.molt5_lengths[index], 768) or p.dtype != np.float32 or m.dtype != np.float32:
            raise RuntimeError('Cache modified after audit')
        return dict(protein_tokens=torch.from_numpy(p), molt5_tokens=torch.from_numpy(m), maccs=torch.from_numpy(self.maccs[index].copy()), label=torch.tensor(row['label'], dtype=torch.float32), row_uid=row['row_uid'])

def collate_tokens(rows):
    """每条序列独立保留长度；无效位置零填充，不能从特征值推断mask。"""
    if not rows:
        raise ValueError('Empty batch')
    batch, L, M = len(rows), max(r['protein_tokens'].shape[0] for r in rows), max(r['molt5_tokens'].shape[0] for r in rows)
    p, m = torch.zeros(batch, L, 1024), torch.zeros(batch, M, 768)
    pm, mm = torch.zeros(batch, L, dtype=torch.bool), torch.zeros(batch, M, dtype=torch.bool)
    for i, row in enumerate(rows):
        l, n = row['protein_tokens'].shape[0], row['molt5_tokens'].shape[0]
        p[i, :l], m[i, :n] = row['protein_tokens'], row['molt5_tokens']
        pm[i, :l], mm[i, :n] = True, True
    return dict(protein_tokens=p, protein_mask=pm, molt5_tokens=m, molt5_mask=mm, maccs=torch.stack([r['maccs'] for r in rows]), label=torch.stack([r['label'] for r in rows]), row_uid=[r['row_uid'] for r in rows])
