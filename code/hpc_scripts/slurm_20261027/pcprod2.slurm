#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-pcprod2
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=32-95%64
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcprod2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcprod2-%A_%a.err
#
# 在 CPU 分区（prod）上运行 Perception Compressor 全量 300 例。
#
# 为什么改投 CPU：gpu 分区 0 空闲、排队预估 8+ 小时；prod 有 127 个空闲节点、
# 19776 个空闲 CPU，且单节点 1.4 TB 内存，装 13.5 GB 的 Llama-2-7b 毫无压力。
# 实测 CPU 单例 89s（48 线程），本作业 24 线程 + 32 路并行，
# 每片约 9-10 例，预计 30-60 分钟完成。
#
# driver 会自动检测无 GPU 并退回 CPU/float32，无需改代码。
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
N=32

cd $RD || exit 1
echo "PCPROD2 shard=$S/$N host=$(hostname) cpus=$SLURM_CPUS_PER_TASK start=$(date -Is)"
free -g | head -2

T0=$(date +%s)
$RD/env-ext/bin/python -u run_pc.py \
  --cases $RD/data/pc_qa.jsonl \
  --rate 2.0 --shard $S --nshards $N \
  --outdir $RD/out/ext_full/perception
RC=$?
echo "PCPROD_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - T0 ))s"
echo "  全量已完成 = $(ls $RD/out/ext_full/perception/*.json 2>/dev/null | wc -l) / 300"
