#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pubcomp
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --array=0-4%5
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pubcomp-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pubcomp-%A_%a.err
#
# 在公开基准（LongBench 300 例）上跑 5 个外部方法的压缩。
#
# 为什么必须重压：现有 out/ext_full/<method>/ 里的产物是针对 longctx_max 的，
# key 是 longctx-*；公开基准的 key 是 lb-*，必须重新生成。
# 产物写入同一目录（key 不冲突），因此 build_ext_arms_pub.py 能直接收。
#
# 规模: 每个 task 一个方法、300 例。按实测速率最慢的 cpc 约 2.3s/例 -> 12min；
#       其余 0.4-1.0s/例 -> 2-5min。--time=25min 留足余量。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=$RD/env-ext/bin/python
B1=$RD/research/token_compression_batch1_v1
METHODS=(quito_x provence cpc llmlingua2 mask_llmlingua2)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
S=$SLURM_ARRAY_TASK_ID

cd $RD/code/longctx || exit 1
echo "=== PUBCOMP method=$M host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

T0=$(date +%s)
timeout -k 30 1400 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/public_lb.jsonl \
  --outdir $RD/out/ext_full \
  --methods $M
RC=$?
echo "=== PUBCOMP method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== count now: $(ls $RD/out/ext_full/$M 2>/dev/null | wc -l)"
exit 0
