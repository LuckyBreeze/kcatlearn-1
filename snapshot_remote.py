"""人工只读快照：区分特征提取、预检、正式训练与最终test。"""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
root = Path(__file__).resolve().parent
result = {'utc': datetime.now(timezone.utc).isoformat()}
receipt = root/'submission.json'
if receipt.exists():
    result['submission'] = json.loads(receipt.read_text())
    jobs = list(result['submission']['jobs'].values())
    for command, name in [(['squeue','-j',','.join(jobs),'-o','%.18i %.20j %.10T %.10M %.30R'],'squeue'), (['sacct','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','--parsable2'],'sacct')]:
        r = subprocess.run(command,capture_output=True,text=True)
        result[name] = r.stdout
        if r.returncode:
            result[name+'_error'] = r.stderr
for kind in ['protein','molt5']:
    p = root/'cache'/kind/'manifest.json'
    if p.exists():
        m = json.loads(p.read_text())
        result[kind] = dict(status=m['status'], completed=len(m['records']), total=m['provenance']['items_count'], longest_preflight=m.get('longest_preflight'), error=m.get('error'))
for rel in ['prepared/manifest.json','run_seed668/preflight/preflight.json','run_seed668/status.json','run_seed668/test/status.json']:
    p = root/rel
    if p.exists():
        m = json.loads(p.read_text())
        if rel.startswith('prepared'):
            m = {k:m[k] for k in ['status','counts','unique_proteins','unique_smiles']}
        result[rel] = m
result['logs'] = {}
for p in sorted((root/'logs').glob('*')):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-8192))
        result['logs'][p.name] = f.read().decode('utf-8',errors='replace')
print(json.dumps(result,ensure_ascii=False))
