#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extnew
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extnew-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extnew-%A_%a.err
#
# EVALUATE TWO ARMS THAT ARE ALREADY COMPRESSED BUT NEVER SCORED.
#
# out/ext_full/longllmlingua      -> 2728/2728 compressed, all status=ok, keep 0.500
# out/ext_full/selective_context  -> 2728/2728 compressed, all status=ok, keep 0.477
#
# Both are real citable methods and both are long-context SPECIFIC, so they are
# the right comparison for this benchmark - and unlike DAC they have no attention
# blow-up (compression already succeeded on 100% of cases including the 1029 that
# make DAC OOM). Only the scoring step is missing.
#
# build_ext_arms.py takes an OPTIONAL 2nd arg limiting which arms go in the
# manifest, so we do NOT recompute the 6 arms already scored at 2728.
#
# Cost: 2 arms x 2728 cases x ~0.43 s = ~39 min single-GPU; over 32 shards ~75 s.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=32
ARMS='longllmlingua,selective_context'
cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EXTFULL shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $RD/build_ext_arms.py $RD/data/ext_arms_new.jsonl "$ARMS" 2>&1 | tail -4
# cases that have BOTH new arms compressed but are not yet scored for them
$PY -c "
import json,io,glob,os
need_arms={'longllmlingua','selective_context'}
have={}
for p in glob.glob('$RD/out/extev*_s*_percase.jsonl'):
    for x in io.open(p,encoding='utf-8'):
        if not x.strip(): continue
        r=json.loads(x)
        if r.get('arm') in need_arms: have.setdefault(r['arm'],set()).add(r['case_key'])
done={}
for m in need_arms:
    d=set(f[:-5] for f in os.listdir('$RD/out/ext_full/%s'%m) if f.endswith('.json'))
    done[m]=d
allc=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
keys=[c['key'] for c in allc if all(c['key'] in done[m] for m in need_arms)]
missing=[k for k in keys if any(k not in have.get(m,set()) for m in need_arms)]
part=missing[int('$S')::int('$N')]
cs={c['key']:c for c in allc}
with io.open('$RD/data/extnew_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for k in part: f.write(json.dumps(cs[k],ensure_ascii=False)+'\n')
print('arms_ready=%d missing=%d shard=${S} cases=%d'%(len(keys),len(missing),len(part)))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/extnew_s${S}.jsonl \
  --out $RD/out/extnew_s${S}_report.txt \
  --json $RD/out/extnew_s${S}_cells.json \
  --percase $RD/out/extnew_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_new.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EXTNEW_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0