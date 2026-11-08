#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-budgetext
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%2
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetext-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/budgetext-%A_%a.err
#
# Published compressors at the two low budgets where the internal arms cross over.
#
# Why these two ratios: the internal sweep put the crossing at r < 0.10, with DARK 0.3297 against a
# leak-free pruner at 0.2377 and a leaking anchor-kept arm at 0.4730 (21% leakage). The question this
# job answers is whether the strongest PUBLISHED compressor also crosses, and at what leakage. Two
# points bracket it; r = 0.50 is already archived from the original run and needs no re-run.
#
# Array layout: T = ratio_index * 8 + shard. Two ratios x eight deterministic shards, so every arm
# pairs per case against budgetdark and xbud.
#
# The compressor runs through a COPY of the published code whose only difference is the budget
# expression ($RD/code/batch1_ratio/compress.py, one line, reads PT_EXT_RATIO with default 0.5). The
# smoke job confirmed that 0.5 reproduces n//2 exactly and that 0.1 yields 10%.
#
# Compression uses env-ext; evaluation uses env. Both run here because both are GPU work.
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

RATIOS=(0.05 0.15)
T=${SLURM_ARRAY_TASK_ID:-0}
RI=$((T / 8))
S=$((T % 8))
RATIO=${RATIOS[$RI]}
TAG=$(echo $RATIO | tr '.' 'p')

CASES=$RD/data/budgetext_s${S}.jsonl
OUT=$RD/out/budgetext_${TAG}_s${S}
if [ -f "$OUT/report.txt" ]; then echo "[budgetext $TAG s$S] report exists, nothing to do"; exit 0; fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

# deterministic shard, identical stride to budgetdark / xbud / quadfull
$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int(8)]
with io.open(RD + '/data/budgetext_s' + '$S' + '.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

echo "budgetext r=$RATIO shard=$S host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd $RD/code/longctx || exit 1
START=$(date +%s)

# ---- 1. compress at the target ratio with the published methods ----
PT_EXT_RATIO=$RATIO $PYE -u ext_compress.py \
    --batch1 "$B1" \
    --cases  "$CASES" \
    --outdir "$OUT/ext" \
    --methods quito_x,cpc 2>&1 | tail -12

# ---- 2. convert to the arm JSONL the harness consumes ----
$PY -u ext_to_arms.py --extdir "$OUT/ext" --out "$OUT/arms.jsonl" 2>&1 | tail -6

# ---- 3. evaluate those arms on the same shard ----
$PY -u e2e_quad.py \
  --cases   "$CASES" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    full \
  --budgets "$RATIO" \
  --ext-arms "$OUT/arms.jsonl" --ext-ratio "$RATIO" \
  --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "budgetext_${TAG}_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
