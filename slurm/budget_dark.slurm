#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-budgetdark
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-7%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetdark-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetdark-%A_%a.err
#
# The missing half of the low-budget sweep.
#
# xbud already covers the full 2,728 cases at r = 0.05, 0.10, 0.15, 0.90 for every routing arm
# EXCEPT quad4 -- that family was produced before the routing arm existed. This job fills exactly
# that hole, and nothing else: quad4, quad4_del and the full control at the three low ratios.
#
# The control matters. `full` is a length-preserving arm, so it must return the same F1 at every
# ratio; if it does not, the batch is wrong and the low-budget numbers cannot be used.
#
# Sharding is the same deterministic key-sorted split as every other run, so arms pair per case
# against xbud and quadfull.
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

NSHARD=8
S=${SLURM_ARRAY_TASK_ID:-0}
CASES=$RD/data/longctx_max.jsonl
SLICE=$RD/data/budgetdark_s${S}.jsonl
OUT=$RD/out/budgetdark_s${S}

if [ ! -f "$CASES" ]; then
    echo "FATAL: missing $CASES (gunzip -k data/longctx_max.jsonl.gz first)"
    exit 2
fi
if [ -f "$OUT/report.txt" ]; then
    echo "[budgetdark $S] report exists, nothing to do"
    exit 0
fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

# deterministic shard: key-sorted stride, identical to quad_full / xmodel / lexcomp
$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int(8)]
with io.open(RD + '/data/budgetdark_s' + '$S' + '.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

cd $RD/code/longctx || exit 1
echo "budgetdark shard=$S/8 host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model dir: $MODEL_DIR"

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases   "$SLICE" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    quad4,quad4_del,full \
  --budgets 0.05,0.1,0.15 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "budgetdark_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
