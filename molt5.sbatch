#!/bin/bash
#SBATCH -J m2_molt5_668
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH -o logs/molt5.%j.out
#SBATCH -e logs/molt5.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=668
python verify_deployment.py
python -u extract_tokens.py --kind molt5 --items prepared/molt5_items.json --checkpoint weights/molt5-base-smiles2caption --output cache/molt5
