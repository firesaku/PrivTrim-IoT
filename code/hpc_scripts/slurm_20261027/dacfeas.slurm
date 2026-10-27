#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-dacfeas
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --array=0-119%120
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacfeas-%A_%a.out
#SBATCH --error=/path/to/project/logs/dacfeas-%A_%a.err
#
# DAC compression over the FEASIBLE subset only.
#
# 1029 of the 2728 cases (37.7%) have >9000 tokens. DAC builds a full attention
# matrix, which grows with the SQUARE of the sequence, and with ~8.7 GB of a
# shared H200 it raises CUDA OOM on every one of them - verified twice (round 12
# found 116 OOMs with a clean 8790 / 9629-token split; round 13's retry produced
# 11 more OOMs on the same cases). Retrying them only burns GPU time.
#
# So this array runs on data/longctx_dac_feasible.jsonl (1699 cases <= 9000 tok).
# 120 shards -> 15 cases each -> 15 x 60 s = 15 min inside a 25-min limit.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
PY=$RD/env-ext/bin/python
PYB=$RD/env/bin/python
B1=$RD/research/token_compression_batch1_v1
SHARD=$SLURM_ARRAY_TASK_ID
NSHARD=120
echo "HOST=$(hostname) SHARD=$SHARD/$NSHARD DATE=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
if ! grep -q "device_map='cuda'" $B1/compress.py; then echo 'FATAL: not GPU version'; exit 3; fi
if grep -q 'not torch.cuda.is_available' $B1/compress.py; then echo 'FATAL: cpu guard present'; exit 3; fi
$PYB -c "
import json,io
L=[x for x in io.open('$RD/data/longctx_dac_feasible.jsonl',encoding='utf-8') if x.strip()]
n=$NSHARD; s=int('$SHARD'); part=L[s::n]
with io.open('$RD/data/dacfeas_shard%d.jsonl'%s,'w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('cases',len(part))"
cd $RD/code/longctx
START=$(date +%s)
$PY -u ext_compress.py --batch1 $B1 \
  --cases $RD/data/dacfeas_shard${SHARD}.jsonl \
  --outdir $RD/out/ext_full --methods 'dac' 2>&1 | tail -3
RC=$?
echo "DACFEAS_S${SHARD}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC