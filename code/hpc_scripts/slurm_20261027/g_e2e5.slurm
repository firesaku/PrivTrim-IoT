#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2e5
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/e2e5-%j.out
#SBATCH --error=/path/to/project/logs/e2e5-%j.err
#
# ç¬¬å››è±¡é™ GPU æ‰“åˆ†ï¼šconflict æ¡ˆä¾‹ x 5 è‡‚ï¼ˆfull/delete/generalize_type/alias/alias_typedï¼‰
# 240 ä¾‹ x 5 = 1200 promptï¼ŒçŸ­ä¸Šä¸‹æ–‡
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
PY=$RD/env/bin/python
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
cd $RD/code || exit 1
echo "HOST=$(hostname) START=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u e2e5.py \
  --variants $RD/data/v4_variants.jsonl \
  --out $RD/out/e2e5_report.txt \
  --json $RD/out/e2e5_cells.json \
  --percase $RD/out/e2e5_percase.jsonl \
  --lm "$QWEN" --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --max-new 64 --batch 8 --dtype fp16
RC=$?
echo E2E5_RC=$RC
cat $RD/out/e2e5_report.txt 2>/dev/null | tail -40
exit $RC
