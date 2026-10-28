"""数据接口的合成核验：不使用任何正式train/valid/test样本。"""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from data import collate_tokens, verify_cache
from prepare import key, sha, protein_input, EXPECTED

class DataChecks(unittest.TestCase):
    def test_frozen_sha_syntax(self):
        import re
        for digest in EXPECTED.values():
            self.assertIsNotNone(re.fullmatch('[0-9a-f]{64}', digest))

    def test_author_crop_and_replacement(self):
        raw = 'U' * 500 + 'A' * 101 + 'ZOB' * 166 + 'ZZ'
        self.assertEqual(len(raw), 1101)
        self.assertEqual(protein_input(raw), 'X' * 1000)
        self.assertEqual(protein_input('ACUZOB'), 'ACXXXX')

    def test_length_masks_and_alignment(self):
        rows = []
        for i, (l, m) in enumerate([(2, 5), (4, 1)]):
            rows.append(dict(row_uid=str(i), protein_tokens=torch.full((l, 1024), float(i+1)), molt5_tokens=torch.full((m, 768), float(i+3)), maccs=torch.full((167,), float(i)), label=torch.tensor(float(i+7))))
        batch = collate_tokens(rows)
        self.assertEqual(batch['protein_tokens'].shape, (2, 4, 1024))
        self.assertEqual(batch['molt5_tokens'].shape, (2, 5, 768))
        self.assertEqual(batch['protein_mask'].sum(1).tolist(), [2, 4])
        self.assertEqual(batch['molt5_mask'].sum(1).tolist(), [5, 1])
        self.assertEqual(batch['row_uid'], ['0', '1'])
        self.assertEqual(batch['label'].tolist(), [7., 8.])
        self.assertEqual(batch['protein_tokens'][0, 2:].count_nonzero(), 0)
        self.assertTrue(torch.equal(batch['maccs'][1], rows[1]['maccs']))

    def test_cache_provenance_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            k = key('AC')
            item_path = root / 'items.json'
            item_path.write_text(json.dumps(dict(items=[dict(key=k, input='AC')], expected_checkpoint_files={'weight':'abc'})))
            path = root / (k+'.npy')
            np.save(path, np.ones((2, 1024), np.float32))
            manifest = dict(schema_version=1, kind='protein', status='complete', provenance=dict(source_items_sha256=sha(item_path), extract_code_sha256=sha(Path(__file__).with_name('extract_tokens.py')), checkpoint_files={'weight': {'sha256':'abc','bytes':10}}), records={k:dict(key=k, file=k+'.npy', dtype='float32', shape=[2,1024], input_length=2, content_token_length=2, sha256=sha(path))})
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertIn(k, verify_cache(root,item_path,'protein'))
            np.save(path, np.zeros((2, 1024),np.float32))
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                verify_cache(root,item_path,'protein')

if __name__ == '__main__':
    unittest.main(verbosity=2)
