#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-reid
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-7%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/reid-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/reid-%A_%a.err
#
# Re-identification attack: can a model recover a protected value from each arm's passage?
#
# The attacker is the same 7B model the paper uses as reader, held fixed across arms, and it sees
# the passage only -- no question -- which is exactly the threat model the paper assumes: the
# provider receives and may retain the prompt.
#
# Arms are built locally by reid_attack.py from the case file, so this job needs no precomputed
# compressor output. That is deliberate: the published-compressor paths are currently unavailable
# (batch1 is missing), and this experiment must not depend on them.
#
# Sharding is the same key-sorted stride as every other run, so arms pair per case.
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
OUT=$RD/out/reid_s${S}

if [ ! -f "$CASES" ]; then echo "FATAL: missing $CASES"; exit 2; fi
if [ -f "$OUT/report.txt" ]; then echo "[reid $S] report exists, nothing to do"; exit 0; fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

$PY - <<PY
import json, io
RD = '$RD'
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int(8)]
with io.open(RD + '/data/reid_s' + '$S' + '.jsonl', 'w', encoding='utf-8', newline=chr(10)) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + chr(10))
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

cd $RD/code/longctx || exit 1
echo "reid shard=$S/8 host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

START=$(date +%s)
$PY -u reid_attack.py \
  --cases   $RD/data/reid_s${S}.jsonl \
  --out     "$OUT/report.txt" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --arms    full,quad4,budget_cluster,alias_budget,alias_budget_link,quad4_del \
  --budgets 0.5 --alias-question \
  --dtype fp16 --batch 8 --max-new 96 --maxlen 24576
RC=$?
echo "reid_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
