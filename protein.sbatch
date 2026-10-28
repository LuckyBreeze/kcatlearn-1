#!/bin/bash
#SBATCH -J m2_prott5_668
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 7
#SBATCH --ntasks-per-node=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=2-00:00:00
#SBATCH -o logs/protein.%j.out
#SBATCH -e logs/protein.%j.err
set -euo pipefail
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate /share/home/u17015/.conda/envs/graphkcat_env
cd /share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
export OMP_NUM_THREADS=7 OPENBLAS_NUM_THREADS=7 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=668
python verify_deployment.py
python -u extract_tokens.py --kind protein --items prepared/protein_items.json --checkpoint /share/home/u17015/data/prot5/prot_t5_xl_uniref50 --output cache/protein
