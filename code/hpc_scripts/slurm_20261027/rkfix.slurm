#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-rkfix
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/rkfix-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/rkfix-%A_%a.err
#
# Attack A3: does removing the rank leak cost anything?
#
# A2 finding: without PT_ALIAS_KEY the alias letters follow ascending
# (type,value) order, so an attacker who knows the value set inverts 100% of
# aliases.  With a key it drops to 19.1% (random for k=5 is 20%).
#
# The main experiment ran WITHOUT the key.  This job re-runs the same arms on
# the same 300 cases WITH PT_ALIAS_KEY, to show the fix is free (F1 must not
# drop).  Consistency is preserved by design: context and question share one
# Aliaser, so the mapping stays self-consistent.
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
S=$SLURM_ARRAY_TASK_ID
N=2

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) RKFIX shard=$S/$N PT_ALIAS_KEY=$PT_ALIAS_KEY start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()][:300]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/rkfix_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/rkfix_s${S}.jsonl \
  --out $RD/out/rkfix_s${S}_report.txt \
  --json $RD/out/rkfix_s${S}_cells.json \
  --percase $RD/out/rkfix_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget_link --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "RKFIX_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
