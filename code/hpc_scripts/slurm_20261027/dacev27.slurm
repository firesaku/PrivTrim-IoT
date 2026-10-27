#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-dacev27
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacev27-%A_%a.out
#SBATCH --error=/path/to/project/logs/dacev27-%A_%a.err
#
# DAC EVALUATION, RESIZED FOR FULL COVERAGE.
#
# The original daceval.slurm was sized for the OLD 179-case gap: 4 shards x 15 min.
# Now the gap is ~2511 cases (2728 minus the 217 already scored), so 4 shards would
# mean 628 cases/shard x 13 arms x 0.43 s = 58 min against a 15-min limit - i.e.
# every shard would be killed before finishing.
#
# 32 shards -> ~79 cases/shard x 13 arms x 0.43 s = 442 s = 7.4 min, inside 20 min.
# Idempotent: build_ext_arms.py rebuilds the manifest from whatever DAC outputs exist,
# and the case list is (compressed minus already-scored), so re-runs only fill gaps.
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
N=32
cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) DACEV27 shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY $RD/build_ext_arms.py $RD/data/ext_arms_dacev.jsonl 2>&1 | tail -2
$PY -c "
import json,io,glob,os
have_eval=set()
for p in glob.glob('$RD/out/extev*_s*_percase.jsonl'):
    for x in io.open(p,encoding='utf-8'):
        if x.strip():
            r=json.loads(x)
            if r['arm']=='ext_dac': have_eval.add(r['case_key'])
dac=set(f[:-5] for f in os.listdir('$RD/out/ext_full/dac') if f.endswith('.json'))
need=sorted(dac-have_eval)
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
sel=[c for c in cs if c['key'] in set(need)]
part=sel[int('$S')::int('$N')]
with io.open('$RD/data/dacev27_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('dac_compressed=%d have_eval=%d need=%d shard=${S} cases=%d'%(len(dac),len(have_eval),len(need),len(part)))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/dacev27_s${S}.jsonl \
  --out $RD/out/dacev27_s${S}_report.txt \
  --json $RD/out/dacev27_s${S}_cells.json \
  --percase $RD/out/dacev27_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_dacev.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "DACEV27_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0