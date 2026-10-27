#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extnew2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extnew2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extnew2-%A_%a.err
#
# EVALUATE longllmlingua + selective_context (both already compressed, 2728 each).
#
# BUG FIXED FROM extnew.slurm: every shard used to call build_ext_arms.py, and all
# 32 shards wrote the SAME manifest file concurrently. A shard could therefore read
# a half-written line, which is exactly what killed the first attempt:
#     json.decoder.JSONDecodeError: Invalid control character at char 31453
# The manifest is now built ONCE by a separate prep job (pt-extnewprep) and every
# shard only READS it. Shards also fail loudly instead of exit 0 on error.
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
MAN=$RD/data/ext_arms_new.jsonl
cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EXTFULL2 shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
# manifest must already exist and be readable - do NOT rebuild it here
[ -s "$MAN" ] || { echo "FATAL: manifest missing: $MAN"; exit 4; }
$PY -c "import json,io,sys; n=0
for l in io.open('$MAN',encoding='utf-8'):
    if l.strip(): json.loads(l); n+=1
print('manifest ok lines=%d'%n)" || { echo 'FATAL: manifest unreadable'; exit 4; }
$PY -c "
import json,io,glob,os
need={'longllmlingua','selective_context'}
have={}
for p in glob.glob('$RD/out/extev*_s*_percase.jsonl')+glob.glob('$RD/out/extnew2_s*_percase.jsonl'):
    for x in io.open(p,encoding='utf-8'):
        if not x.strip(): continue
        r=json.loads(x)
        if r.get('arm') in need: have.setdefault(r['arm'],set()).add(r['case_key'])
done={m:set(f[:-5] for f in os.listdir('$RD/out/ext_full/%s'%m) if f.endswith('.json')) for m in need}
allc=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
keys=[c['key'] for c in allc if all(c['key'] in done[m] for m in need)]
missing=[k for k in keys if any(k not in have.get(m,set()) for m in need)]
part=missing[int('$S')::int('$N')]
cs={c['key']:c for c in allc}
with io.open('$RD/data/extnew2_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for k in part: f.write(json.dumps(cs[k],ensure_ascii=False)+'\n')
print('arms_ready=%d missing=%d shard=${S} cases=%d'%(len(keys),len(missing),len(part)))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/extnew2_s${S}.jsonl \
  --out $RD/out/extnew2_s${S}_report.txt \
  --json $RD/out/extnew2_s${S}_cells.json \
  --percase $RD/out/extnew2_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms "$MAN" \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EXTNEW2_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
# propagate failure so a broken shard is visible instead of silently 'COMPLETED'
exit $RC