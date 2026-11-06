#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-mprobe
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --array=0-3%2
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/mprobe-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/mprobe-%A_%a.err
#
# Reader-model probe: which candidate behaves like the readers DARK already works on?
#
# The rule needs the reader to accept an aliased handle as a normal identifier. A reader that
# reads "[ACCT-A]" as a redaction marker answers UNKNOWN and loses the task, which is what
# Qwen2.5-14B does on 94% of multi-entity cases. Rather than spend a full run finding that out
# per model, score 300 cases (200 multi-entity, 100 needle) with the two arms and look at the
# UNKNOWN rate before committing.
#
# One model per array task. Deterministic sample, same cases for every model.
#
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

MODELS=(
  "models--ibm-granite--granite-3.1-8b-instruct:granite8b"
  "models--microsoft--phi-4:phi4"
  "models--meta-llama--Llama-2-7b-hf:llama2_7b"
  "models--Qwen--Qwen2.5-14B-Instruct:qwen14b"
)
ENTRY=${MODELS[$SLURM_ARRAY_TASK_ID]}
REPO_DIR=${ENTRY%%:*}
TAG=${ENTRY##*:}

cd $RD/code/longctx || exit 1
echo "probe tag=$TAG host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

MODEL_DIR=$(ls -d $RD/hf_cache/$REPO_DIR/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no snapshot for $REPO_DIR"; exit 2; fi

# fixed 300-case probe sample: 200 multi-entity + 100 needle, sorted by key for determinism
$PY - <<PY
import json, io
RD = '$RD'
NL = chr(10)
cases = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
cases.sort(key=lambda c: c['key'])
multi = [c for c in cases if c['case_kind'] == 'multi_entity']
needle = [c for c in cases if c['case_kind'] == 'needle']
sample = multi[:200] + needle[:100]
with io.open(RD + '/data/mprobe_cases.jsonl', 'w', encoding='utf-8', newline=NL) as f:
    for c in sample:
        f.write(json.dumps(c, ensure_ascii=False) + NL)
print('probe sample: %d (%d multi / %d needle)' % (len(sample), len(multi[:200]), len(needle[:100])))
PY

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/mprobe_cases.jsonl \
  --out $RD/out/mprobe_${TAG}_report.txt \
  --json $RD/out/mprobe_${TAG}_cells.json \
  --percase $RD/out/mprobe_${TAG}_percase.jsonl \
  --lm "$MODEL_DIR" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,quad4 \
  --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "mprobe_${TAG}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC