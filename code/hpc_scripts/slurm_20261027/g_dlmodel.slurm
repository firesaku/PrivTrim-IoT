#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dlm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dlm-%j.out
#SBATCH --error=/path/to/project/logs/dlm-%j.err
#
# ç¬¬ä¸‰ä¸ªä¸‹æ¸¸æ¨¡åž‹ï¼šQwen2.5-14B-Instructï¼ˆä¸åŒè§„æ¨¡ï¼‰+ gemma-2-9b-itï¼ˆä¸åŒå®¶æ—ï¼Œéžgatedï¼‰
set -uo pipefail
module load intel_h200_gpu miniconda/3
RD=/path/to/project
export PYTHONNOUSERSITE=1
export HF_HOME=$RD/hf_cache
PY=$RD/env/bin/python
echo "HOST=$(hostname) DATE=$(date -Is)"
for M in \
  Qwen/Qwen2.5-14B-Instruct \
  google/gemma-2-9b-it \
  ; do
  echo "=== $M ==="
  $PY -c "
import sys
from huggingface_hub import snapshot_download
m=sys.argv[1]
try:
    p=snapshot_download(m,cache_dir='$RD/hf_cache',max_workers=8)
    print('OK',m,p)
except Exception as e:
    print('FAIL',m,type(e).__name__,str(e)[:300])
" "$M" 2>&1 | tail -4
done
du -sh $RD/hf_cache
echo DLM_DONE
