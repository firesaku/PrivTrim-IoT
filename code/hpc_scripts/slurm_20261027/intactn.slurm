#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-intactN
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/path/to/project/repos/text_sanitization
#SBATCH --output=/path/to/project/logs/intactN-%A_%a.out
#SBATCH --error=/path/to/project/logs/intactN-%A_%a.err
#
# INTACT 样本跑（100 例分 2 片）。实测 3 篇 35 秒 ≈ 12 秒/篇，50 篇约 10 分钟。
# 分 2 片是为了塞进碎片化空洞（此前 16 片 x 22 分钟验证过这个策略）。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export HF_HOME=$RD/hf_cache
export HF_HUB_CACHE=$RD/hf_cache
export TRANSFORMERS_CACHE=$RD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=$RD/env-ext/bin/python
S=$SLURM_ARRAY_TASK_ID

echo "=== INTACTN shard=$S host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

IN=$RD/data/intact_sample_s${S}.json
OUT=$RD/data/intact_sample_s${S}_out.json

T0=$(date +%s)
timeout -k 60 1800 $PY -u $RD/run_intact.py \
  --cases "$IN" \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --out "$OUT"
RC=$?
echo "=== INTACTN shard=$S rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
ls -l "$OUT" 2>/dev/null
exit 0
