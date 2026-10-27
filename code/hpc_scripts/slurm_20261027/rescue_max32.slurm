#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-rescue
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/rescue-%A_%a.out
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1 PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
NSHARD=32
cd $RD/code/longctx || exit 1
echo "RESCUE shard=$S host=$(hostname) $(date -Is)"
$PY -c "
import io
L=[x for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=L[int('$S')::32]
with io.open('$RD/data/max32_shard%d.jsonl'%int('$S'),'w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('shard','$S','cases',len(part))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/max32_shard${S}.jsonl \
  --out $RD/out/max32_s${S}_report.txt \
  --json $RD/out/max32_s${S}_cells.json \
  --percase $RD/out/max32_s${S}_percase.jsonl \
  --lm "$QWEN" --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms all --budgets 0.25,0.5,0.75 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
echo "RESCUE_S${S}_RC=$?"
