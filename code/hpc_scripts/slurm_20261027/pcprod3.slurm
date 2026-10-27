#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-pcprod3
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-95%96
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcprod3-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcprod3-%A_%a.err
#
# 第三波：补齐前两波超时后剩下的案例。
# 前两波在 --time=1:30:00 到期时完成 193/300。driver 有 exists 判断，
# 已完成的会瞬间跳过，因此 96 个分片只处理缺失的那 107 例。
# 时限放宽到 3 小时（prod 分区上限 3 天，不受限）。
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/pc_stubs:$RD/repos/PerceptionCompressor:$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
S=$SLURM_ARRAY_TASK_ID
N=96

cd $RD || exit 1
echo "PCPROD3 shard=$S/$N host=$(hostname) start=$(date -Is) 已有=$(ls out/ext_full/perception/*.json 2>/dev/null | wc -l)"

T0=$(date +%s)
$RD/env-ext/bin/python -u run_pc.py \
  --cases $RD/data/pc_qa.jsonl \
  --rate 2.0 --shard $S --nshards $N \
  --outdir $RD/out/ext_full/perception
RC=$?
echo "PCPROD3_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - T0 ))s 全量=$(ls $RD/out/ext_full/perception/*.json 2>/dev/null | wc -l)"
