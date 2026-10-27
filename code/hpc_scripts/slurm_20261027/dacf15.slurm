#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-dacf15
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-169%170
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/dacf15-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/dacf15-%A_%a.err
#
# DAC on the feasible subset, 15-MINUTE shards for maximum backfill.
#
# Round 16-17: the 25-min variant sat at (Priority) with ZERO shards running for
# 18+ min, because backfill can only place a job inside a free window at least as
# long as its TimeLimit. Our priority (3376) is below other users (4058).
#
# 170 shards x 10 cases: 10 x 60 s = 10 min + model load, inside 15 min.
# More, shorter shards = many more windows we can slot into.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
PY=$RD/env-ext/bin/python
PYB=$RD/env/bin/python
B1=$RD/research/token_compression_batch1_v1
SHARD=$SLURM_ARRAY_TASK_ID
NSHARD=170
echo "HOST=$(hostname) SHARD=$SHARD/$NSHARD DATE=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
if ! grep -q "device_map='cuda'" $B1/compress.py; then echo 'FATAL: not GPU version'; exit 3; fi
if grep -q 'not torch.cuda.is_available' $B1/compress.py; then echo 'FATAL: cpu guard present'; exit 3; fi
$PYB -c "
import json,io
L=[x for x in io.open('$RD/data/longctx_dac_feasible.jsonl',encoding='utf-8') if x.strip()]
n=$NSHARD; s=int('$SHARD'); part=L[s::n]
with io.open('$RD/data/dacf15_shard%d.jsonl'%s,'w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('cases',len(part))"
cd $RD/code/longctx
START=$(date +%s)
$PY -u ext_compress.py --batch1 $B1 \
  --cases $RD/data/dacf15_shard${SHARD}.jsonl \
  --outdir $RD/out/ext_full --methods 'dac' 2>&1 | tail -3
RC=$?
echo "DACF15_S${SHARD}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC