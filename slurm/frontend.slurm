#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-frontend
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/frontend-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/frontend-%A_%a.err
#
# Experiment 1: DARK as a front end, with a published compressor applied on top.
#
# The question: if you alias the identifier anyway, can the strongest compressor match DARK?
# Four steps per shard, because the compressor must run on an ALIASED passage:
#
#   1. rebuild DARK's own passage and aliased question with the real tokenizer (no weights loaded)
#   2. run the published compressor on that passage at r = 0.5
#   3. convert to the arm JSONL the harness consumes
#   4. evaluate on the same sharded cases
#
# Leakage is 0 by construction here -- the passage never contained the values -- so what the composed
# arms measure is the F1 that survives a second compression pass. Any leakage reported by this job
# would mean step 1 broke, and build_frontend_cases.py refuses to write a file in that case.
#
# The published code runs through $RD/code/batch1_ratio, a copy whose only difference from the
# original is the budget expression.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true

RD=/dpc/kuex0095/100069961_ku/privtrim
B1=$RD/code/batch1_ratio
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYE=$RD/env-ext/bin/python
PY=$RD/env/bin/python

S=${SLURM_ARRAY_TASK_ID:-0}
OUT=$RD/out/frontend_s${S}
CASES=$RD/data/frontend_s${S}.jsonl
if [ -f "$OUT/report.txt" ]; then echo "[frontend $S] report exists, nothing to do"; exit 0; fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

# deterministic shard, identical stride to every other run
$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int(8)]
with io.open(RD + '/data/frontend_src_s' + '$S' + '.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

cd $RD/code/longctx || exit 1
echo "frontend shard=$S host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
START=$(date +%s)

# ---- 1. rebuild DARK's passage + aliased question (tokenizer only) ----
$PY -u build_frontend_cases.py \
    --cases $RD/data/frontend_src_s${S}.jsonl \
    --lm    "$MODEL_DIR" \
    --out   "$CASES" \
    --arm   quad4 --budget 0.5 2>&1 | tail -8
if [ ! -s "$CASES" ]; then echo "FATAL: frontend cases not built"; exit 3; fi

# ---- 2. published compressors on the aliased passage ----
PT_EXT_RATIO=0.5 $PYE -u ext_compress.py \
    --batch1 "$B1" \
    --cases  "$CASES" \
    --outdir "$OUT/ext" \
    --methods quito_x,cpc 2>&1 | tail -10

# ---- 3. arms JSONL ----
$PY -u ext_to_arms.py --extdir "$OUT/ext" --out "$OUT/arms.jsonl" 2>&1 | tail -5

# ---- 4. evaluate ----
$PY -u e2e_quad.py \
  --cases   "$CASES" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    full \
  --budgets 0.5 \
  --ext-arms "$OUT/arms.jsonl" --ext-ratio 0.5 \
  --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "frontend_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
