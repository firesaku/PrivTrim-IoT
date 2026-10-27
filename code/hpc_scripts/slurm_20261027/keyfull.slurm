#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-keyfull
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/keyfull-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/keyfull-%A_%a.err
#
# 全量 21 臂在启用 PT_ALIAS_KEY 下的重跑 —— 与 max32 完全同构的替代数据集。
#
# 为什么要整份重跑：第 9 节的重跑只覆盖了 5 个臂（full/delete/alias/
# alias_budget/alias_budget_link @0.2,0.5）。剩下的 16 个臂
# （budget_cluster / budget_cluster_link / budget_random / delete_budget /
# 以及 0.25 / 0.75 预算点）仍是**带秩泄漏**的旧数字。为了让主表、消融表、
# 预算曲线三者出自**同一个设置**，用 21 臂整份重跑一次最干净。
#
# 与 max32 的唯一差别就是 PT_ALIAS_KEY 这一行。
# 规模: 2728 / 32 片 = 85 例/片 x 21 = 1785 次生成 ≈ 13min -> --time=25min
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=32

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) KEYFULL shard=$S/$N PT_ALIAS_KEY=$PT_ALIAS_KEY start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int($N)]
with io.open('$RD/data/keyfull_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/keyfull_s${S}.jsonl \
  --out $RD/out/keyfull_s${S}_report.txt \
  --json $RD/out/keyfull_s${S}_cells.json \
  --percase $RD/out/keyfull_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms all --budgets 0.25,0.5,0.75 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "KEYFULL_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
