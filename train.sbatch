#!/bin/bash
#SBATCH -J m2_train668
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/train.%j.out
#SBATCH -e logs/train.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 PYTHONHASHSEED=668
python verify_deployment.py
ARGS=(--prepared prepared --protein-cache cache/protein --molt5-cache cache/molt5 --output run_seed668 --device cuda:0)
python -u train.py "${ARGS[@]}" --mode preflight
python -u train.py "${ARGS[@]}" --mode train
python -u train.py "${ARGS[@]}" --mode test
