#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-evsync
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:12:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/evsync-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/evsync-%A_%a.err
#
# ② INTACT + 问题同步的评测。
#
# 与 eval_intact.slurm 的唯一区别：--cases 换成 data/intact_sync_cases.jsonl
# （question 里的原始标识符已替换成 INTACT 实际用的替代值）。
# 上下文仍由 --ext-arms 提供（即 INTACT 的匿名化文本）。**不重跑 INTACT**。
#
# 目的：把"我们自己得到问题同步、INTACT 没有"这个不公平消掉。
#   若 F1 从 0 恢复 -> 差距来自压缩能力，我们赢得干净
#   若仍为 0    -> 说明 INTACT 的替代值不可对齐，同步机制是 PrivTrim 的实质贡献
#
# 注：--arms full 那一臂在同步 question + 原始上下文下无意义（问题问 CODE_5，
# 上下文里只有 ACCT-xxx），分析时忽略即可。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=2

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EVSYNC shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY $RD/build_ext_arms.py $RD/data/ext_arms_sync_s${S}.jsonl intact 2>&1 | tail -3

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/intact_sync_cases.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/evsync_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/evsync_s${S}.jsonl \
  --out $RD/out/evsync_s${S}_report.txt \
  --json $RD/out/evsync_s${S}_cells.json \
  --percase $RD/out/evsync_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_sync_s${S}.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EVSYNC_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
