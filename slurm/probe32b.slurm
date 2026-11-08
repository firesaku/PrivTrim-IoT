#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-probe32b
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/probe32b-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/probe32b-%j.err
#
# 32B reader probe -- the go/no-go for experiment 5.
#
# The 300-case sample is the SAME construction mprobe.slurm uses for granite-8B, phi-4 and
# Qwen2.5-14B: keys sorted, multi_entity[:200] + needle[:100]. That is the whole point -- the 14B
# probe fails on multi-entity (94% UNKNOWN), so the 32B number only means something if it is
# computed on exactly the same cases.
#
# Why a pilot and not the full 2,728: if a 32B reader also reads "[ACCT-A]" as a redaction marker,
# every arm collapses and a full run would spend 12 GPU-hours to produce a column of UNKNOWNs. The
# probe answers that in one job.
#
# Both case kinds are reported separately on purpose. A 32B that holds up on needle and collapses
# on multi-entity is a sharper result than an aggregate, not a failed experiment: it locates the
# cost on the alias resolution rather than on compression.
#
# batch=1 and maxlen 24576: 32B fp16 is ~64 GB of weights on a 141 GB H200, and the longest
# excerpts are 18k tokens, so the KV cache needs the headroom that batch 1 leaves.
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

CASES=$RD/data/mprobe_cases.jsonl
OUT=$RD/out/mprobe_qwen32b
if [ -f "$OUT/report.txt" ]; then echo "[probe32b] report exists, nothing to do"; exit 0; fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-32B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then
    echo "FATAL: no Qwen2.5-32B snapshot under $RD/hf_cache (download job not finished?)"
    exit 2
fi

# identical sample construction to mprobe.slurm
$PY - <<PY
import json, io
RD = '$RD'
cases = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
cases.sort(key=lambda c: c['key'])
multi = [c for c in cases if c['case_kind'] == 'multi_entity']
needle = [c for c in cases if c['case_kind'] == 'needle']
sample = multi[:200] + needle[:100]
with io.open(RD + '/data/mprobe_cases.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in sample:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('probe sample: %d (%d multi / %d needle)' % (len(sample), len(multi[:200]), len(needle[:100])))
PY

cd $RD/code/longctx || exit 1
echo "probe32b host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model dir: $MODEL_DIR"
du -sh "$MODEL_DIR" 2>&1 | head -1

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases   "$CASES" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    full,quad4 \
  --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 1 --max-new 64 --seed 20261015
RC=$?
echo "probe32b_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
