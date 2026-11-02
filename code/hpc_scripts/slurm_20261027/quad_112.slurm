#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-q112
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/q112-%A_%a.out
#SBATCH --error=/path/to/project/logs/q112-%A_%a.err
#
# 112 例基准（data/longctx_cases.jsonl）：为论文主表 tab:main 提供可替换数字。
# 注意：这套案例与 longctx_max.jsonl 共用 key 但内容不同（87/112 金标不同），
#       不能与 2,728 那套互相比较。
#
# 臂（8 个，全部 --no-oracle）：
#   full, delete, alias, budget_cluster, budget_cluster_link,
#   alias_budget, alias_budget_link, quad4
#
# 规模：112/2 = 56 例/片 × 8 臂 = 448 次生成
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
N=2

cd $RD/code/longctx || exit 1
echo "Q112 shard=$S/$N host=$(hostname) start=$(date -Is)"

$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(x) for x in io.open(RD + '/data/longctx_cases.jsonl', encoding='utf-8') if x.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int('$N')]
with io.open(RD + '/data/q112_s$S.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/q112_s${S}.jsonl \
  --out $RD/out/q112_s${S}_report.txt \
  --json $RD/out/q112_s${S}_cells.json \
  --percase $RD/out/q112_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,delete,alias,budget_cluster,budget_cluster_link,alias_budget,alias_budget_link,quad4 \
  --budgets 0.5 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "Q112_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
