#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-keyset
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-15%16
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/keyset-%A_%a.out
#SBATCH --error=/path/to/project/logs/keyset-%A_%a.err
#
# B: 全量重跑，启用 PT_ALIAS_KEY（消除秩泄漏后的正式数字）。
#
# 与主实验的唯一区别就是这一行环境变量。臂的精简依据：
#   - 外部基线不受影响（它们不用我们的 Aliaser），所以不必重跑
#   - 需要重跑的是**我们自己的臂**：
#       full / delete / alias（无预算，消融用）
#       alias_budget / alias_budget_link @ 0.20, 0.50（主表 + link 消融）
#   => 3 + 2x2 = 7 prompts/例
#
# 规模: 2728 例 / 16 片 = 171 例/片 x 7 = 1197 次生成 ≈ 8.5min -> --time=20min
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=16

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) KEYSET shard=$S/$N PT_ALIAS_KEY=$PT_ALIAS_KEY start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int($N)]
with io.open('$RD/data/keyset_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/keyset_s${S}.jsonl \
  --out $RD/out/keyset_s${S}_report.txt \
  --json $RD/out/keyset_s${S}_cells.json \
  --percase $RD/out/keyset_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,delete,alias,alias_budget,alias_budget_link --budgets 0.2,0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "KEYSET_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
