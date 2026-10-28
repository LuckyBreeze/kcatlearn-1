#!/bin/bash
#SBATCH -J m2_cuda_probe
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:05:00
#SBATCH -o logs/cuda_probe.%j.out
#SBATCH -e logs/cuda_probe.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7
python -u cuda_probe.py
