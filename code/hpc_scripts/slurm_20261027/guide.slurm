#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-guide
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/guide-%j.out
#SBATCH --error=/path/to/project/logs/guide-%j.err
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1 PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
cd $RD || exit 1
echo "GUIDE start=$(date -Is) host=$(hostname)"
$RD/env/bin/python -u gen_guiding.py
echo "GUIDE rc=$? end=$(date -Is)"
