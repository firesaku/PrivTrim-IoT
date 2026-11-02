#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-quadfull
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --array=0-7%8
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/quadfull-%A_%a.out
#SBATCH --error=/path/to/project/logs/quadfull-%A_%a.err
#
# 全量 2,728 例：四象限(quad4) + 无金标消融，8 分片。
# 目的：消除上一轮 300 例子集「92% 来自 pec_home」的代表性偏差，
#       让数字可与主表 / 外部基线直接并列。
#
# 臂（7 个，全部 --no-oracle）：
#   full, budget_cluster, budget_cluster_link, alias_budget,
#   alias_budget_link, quad4, quad4_del
#
# 规模：2728/8 = 341 例/片 × 7 臂 = 2387 次生成/片
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
N=8

cd $RD/code/longctx || exit 1
echo "QUADFULL shard=$S/$N host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY - <<PY
import json, io
RD = '$RD'
allc = []
for line in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8'):
    if line.strip():
        allc.append(json.loads(line))
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int('$N')]
with io.open(RD + '/data/quadfull_s$S.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
import collections
src = collections.Counter(c.get('source') for c in part)
print('  shard=$S total=%d slice=%d sources=%s'
      % (len(allc), len(part), dict(src)))
PY

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/quadfull_s${S}.jsonl \
  --out $RD/out/quadfull_s${S}_report.txt \
  --json $RD/out/quadfull_s${S}_cells.json \
  --percase $RD/out/quadfull_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,budget_cluster,budget_cluster_link,alias_budget,alias_budget_link,quad4,quad4_del \
  --budgets 0.5 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "QUADFULL_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
