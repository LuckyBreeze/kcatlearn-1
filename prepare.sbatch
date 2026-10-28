#!/bin/bash
#SBATCH -J m2_prepare668
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/prepare.%j.out
#SBATCH -e logs/prepare.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python verify_deployment.py
python prepare.py --base /share/home/u17015/data/GraphKcat/catapro_cluster40_six_seed668_20261025 --output prepared --checkpoints checkpoints.json
