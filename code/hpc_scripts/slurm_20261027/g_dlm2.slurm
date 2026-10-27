#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dlm2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dlm2-%j.out
#SBATCH --error=/path/to/project/logs/dlm2-%j.err
set -uo pipefail
module load intel_h200_gpu miniconda/3
RD=/path/to/project
export PYTHONNOUSERSITE=1
export HF_HOME=$RD/hf_cache
PY=$RD/env/bin/python
for M in ibm-granite/granite-3.1-8b-instruct microsoft/phi-4 ; do
  echo "=== $M ==="
  $PY -c "
import sys
from huggingface_hub import snapshot_download
m=sys.argv[1]
try:
    p=snapshot_download(m,cache_dir='$RD/hf_cache',max_workers=8)
    print('OK',m,p)
except Exception as e:
    print('FAIL',m,type(e).__name__,str(e)[:200])
" "$M" 2>&1 | tail -4
done
echo DLM2_DONE
