"""Slurm分配GPU内的短诊断，只写诊断文件，不提特征或训练。"""
import json
import os
from pathlib import Path
import socket
import subprocess
import traceback
import sys
import torch

assert os.environ.get('SLURM_JOB_ID'), 'GPU probe requires sbatch'
result=dict(node=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],torch=str(torch.__version__),cuda_runtime=torch.version.cuda,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'))
result.update(python=sys.executable,torch_file=torch.__file__,environment={k:os.environ.get(k) for k in ['SLURM_STEP_GPUS','CONDA_PREFIX','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','PYTORCH_NVML_BASED_CUDA_CHECK']})
r=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,uuid','--format=csv'],capture_output=True,text=True)
result['nvidia_smi']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
try:
    torch.cuda.init()
    result.update(available=torch.cuda.is_available(),count=torch.cuda.device_count(),name=torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32=False
    a=torch.ones(32,32,device='cuda',dtype=torch.float32)
    assert bool((a@a==32).all())
    torch.cuda.synchronize()
    result['status']='passed'
except Exception:
    result['status']='failed'
    result['error']=traceback.format_exc()
path=Path('gpu_diagnostics')
path.mkdir(exist_ok=True)
(path/(result['job']+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
if result['status']!='passed':raise SystemExit(1)
