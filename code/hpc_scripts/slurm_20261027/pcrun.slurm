#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pcrun
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16
#SBATCH --depend=afterok:1506330
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcrun-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcrun-%A_%a.err
#
# 全量运行 Perception Compressor（NAACL 2025 Findings）于 LongBench 300 例。
#
# 依赖冒烟作业成功，避免系统性错误浪费 16 个 GPU 名额。
# 每个案例单独落盘，重跑会自动跳过已有结果（driver 里有 exists 判断），
# 因此分片失败可单独重投而不丢进度。
#
# 规模: 300 / 16 片 ≈ 19 例/片。单例耗时未知（CPU 微型案例 26s），
#       长上下文 GPU 上预计 1-5 min/例 -> 19-95 min/片，--time=2h 留足。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/pc_stubs:$RD/repos/PerceptionCompressor:$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
S=$SLURM_ARRAY_TASK_ID
N=16

cd $RD || exit 1
echo "PCRUN shard=$S/$N start=$(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

T0=$(date +%s)
$RD/env-ext/bin/python -u run_pc.py \
  --cases $RD/data/pc_qa.jsonl \
  --rate 2.0 --shard $S --nshards $N \
  --outdir $RD/out/ext_full/perception
RC=$?
echo "PCRUN_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - T0 ))s"
echo "  已完成文件数 = $(ls $RD/out/ext_full/perception/*.json 2>/dev/null | wc -l)"
