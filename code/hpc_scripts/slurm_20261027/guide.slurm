#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-guide
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/guide-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/guide-%j.err
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1 PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
cd $RD || exit 1
echo "GUIDE start=$(date -Is) host=$(hostname)"
$RD/env/bin/python -u gen_guiding.py
echo "GUIDE rc=$? end=$(date -Is)"
