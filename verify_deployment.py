"""每个sbatch阶段先验证提交时冻结的代码和配置。"""
import json
from pathlib import Path
from prepare import sha

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'deployment_manifest.json').read_text())
for rel, expected in manifest.items():
    if sha(root / rel) != expected:
        raise RuntimeError('Deployment changed: ' + rel)
print('deployment_sha256_passed', len(manifest), flush=True)
