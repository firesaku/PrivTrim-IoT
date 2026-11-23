#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-budgetctrl
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetctrl-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetctrl-%A_%a.err
#
# Re-run the two redundancy control arms at the low budgets WITHOUT the answer oracle.
#
# Why this exists. The archived `xbud` family (job 1498380, run 24-25 Oct) used the older
# e2e_longctx_r5full.py, which ranks the answer-bearing unit LAST for dropping. That is an oracle:
# at r = 0.15 those arms retain the answer in 100% of cases, where an honest run retains it in about
# 5%. Verified locally on 300 cases by rebuilding with and without --no-oracle: the archived numbers
# match the oracle build exactly, and the r = 0.50 family (quadfull) does not.
#
# Mixing the two in one figure is what produced the apparent cross-over, so the low-budget control
# points are regenerated here under the same protocol as every other arm the paper reports:
# --no-oracle. Everything else about the configuration is unchanged, including the deterministic
# key-sorted sharding, so the arms still pair per case against budgetdark and quadfull.
#
# DARK and its delete variant are NOT re-run: they keep the whole block, so the answer is in the
# prompt whatever the drop order says, and their low-budget points from job 1561910 were already
# produced with --no-oracle.
#
# `full` is not re-run either: it is length-preserving, so the oracle cannot change it.
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
export PYTHONIOENCODING=utf-8
PY=$RD/env/bin/python

S=${SLURM_ARRAY_TASK_ID:-0}
OUT=$RD/out/budgetctrl_s${S}
CASES=$RD/data/budgetctrl_s${S}.jsonl
if [ -f "$OUT/report.txt" ]; then echo "[budgetctrl $S] report exists, nothing to do"; exit 0; fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

# deterministic shard, identical stride to budgetdark / quadfull / xbud
$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int(8)]
with io.open(RD + '/data/budgetctrl_s' + '$S' + '.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

cd $RD/code/longctx || exit 1
echo "budgetctrl shard=$S host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=$(date +%s)

$PY -u e2e_quad.py \
  --cases   "$CASES" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    budget_cluster,budget_cluster_link \
  --budgets 0.05,0.1,0.15 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "budgetctrl_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
