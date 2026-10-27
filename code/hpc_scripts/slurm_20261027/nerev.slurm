#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-nerev
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/nerev-%A_%a.out
#SBATCH --error=/path/to/project/logs/nerev-%A_%a.err
#
# Score the ner_redact arm (off-the-shelf spaCy NER + email/phone regex).
# One arm x 2728 cases x ~0.43 s = ~20 min single GPU; over 32 shards ~40 s.
# Reads a pre-built manifest; never rebuilds it. Propagates failures.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=${SLURM_ARRAY_TASK_COUNT:-32}  # derive from array size so a small array still covers everything
N=${N%\*}
echo "array_size=$N task=$SLURM_ARRAY_TASK_ID"
MAN=$RD/data/ext_arms_ner.jsonl
cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) NEREV shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
[ -s "$MAN" ] || { echo "FATAL: manifest missing"; exit 4; }
$PY -c "import json,io;n=0
for l in io.open('$MAN',encoding='utf-8'):
    if l.strip(): json.loads(l); n+=1
print('manifest ok lines=%d'%n)" || { echo 'FATAL: manifest unreadable'; exit 4; }
$PY -c "
import json,io,glob,os
have=set()
for p in glob.glob('$RD/out/extev*_s*_percase.jsonl')+glob.glob('$RD/out/nerev*_percase.jsonl'):
    for x in io.open(p,encoding='utf-8'):
        if not x.strip(): continue
        r=json.loads(x)
        if r.get('arm')=='ext_ner_redact': have.add(r['case_key'])
done=set(f[:-5] for f in os.listdir('$RD/out/ext_full/ner_redact') if f.endswith('.json'))
allc=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
need=[c['key'] for c in allc if c['key'] in done and c['key'] not in have]
part=need[int('$S')::int('$N')]
cs={c['key']:c for c in allc}
with io.open('$RD/data/nerev_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for k in part: f.write(json.dumps(cs[k],ensure_ascii=False)+'\n')
print('compressed=%d scored=%d need=%d shard=${S} cases=%d'%(len(done),len(have),len(need),len(part)))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/nerev_s${S}.jsonl \
  --out $RD/out/nerev_s${S}_report.txt \
  --json $RD/out/nerev_${SLURM_JOB_ID}_s${S}_cells.json \
  --percase $RD/out/nerev_${SLURM_JOB_ID}_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms "$MAN" \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "NEREV_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC