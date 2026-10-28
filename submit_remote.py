"""只提交一次批准的任务链；每收到JobID立即落盘，断线后读取凭据。"""
import argparse
import fcntl
import json
import subprocess
import uuid
from pathlib import Path
from prepare import dump, sha

ROOT = Path(__file__).resolve().parent

def main(stage):
    with (ROOT / '.submit.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = ROOT / 'submission.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else dict(scenario='cluster40', seed=668, model='protein_Q_molt5_MACCS_KV', jobs={})
        def submit(name, dependency=None):
            if name in receipt['jobs']:
                return receipt['jobs'][name]
            if receipt.get('pending_submission'):
                raise RuntimeError('Previous submission outcome unknown; reconcile squeue/sacct before any retry: ' + str(receipt['pending_submission']))
            # 先写意图：即使Slurm接受后连接中断，也不会因缺JobID而自动重复提交。
            tag = 'model2_668_' + name + '_' + uuid.uuid4().hex
            receipt['pending_submission'] = dict(stage=name, comment=tag, dependency=dependency)
            dump(receipt_path, receipt)
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes']
            command.append('--comment=' + tag)
            if dependency:
                command.append('--dependency=afterok:' + ':'.join(dependency))
            command.append(name + '.sbatch')
            result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            job = result.stdout.strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Ambiguous submission response: ' + result.stdout)
            receipt['jobs'][name] = job
            receipt['pending_submission'] = None
            dump(receipt_path, receipt)
            print(name, job, flush=True)
            return job
        if stage == 'prepare-protein':
            p = submit('prepare')
            submit('protein', [p])
        else:
            if not {'prepare', 'protein'}.issubset(receipt['jobs']):
                raise RuntimeError('Prepare/protein receipt missing')
            # 权重传输成功不等于正确：目标端全部批准文件SHA通过后才排MolT5。
            checkpoints = json.loads((ROOT / 'checkpoints.json').read_text())
            target = Path(checkpoints['molt5']['path'])
            for name, expected in checkpoints['molt5']['files'].items():
                if sha(target / name) != expected:
                    raise RuntimeError('Transferred MolT5 file mismatch: ' + name)
            dump(ROOT / 'molt5_transfer_verified.json', dict(status='passed', files=checkpoints['molt5']['files']))
            m = submit('molt5', [receipt['jobs']['prepare']])
            submit('train', [receipt['jobs']['protein'], m])
        print(json.dumps(receipt), flush=True)
        subprocess.run(['squeue', '-j', ','.join(receipt['jobs'].values()), '-o', '%.18i %.20j %.10T %.10M %.30R'], check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare-protein', 'remaining'])
    main(parser.parse_args().stage)
