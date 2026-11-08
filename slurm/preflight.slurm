#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-preflight
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/preflight-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/preflight-%j.err
#
# Preflight for the 32B download, run as a job rather than on the login node.
#
# The login node rules forbid df/du/python there, so the facts needed to decide whether a 65 GB
# download is safe -- free space, whether compute nodes can reach the Hub, which downloader exists --
# are collected here on a CPU node and written to one small file that can then be read.
set -uo pipefail

RD=/dpc/kuex0095/100069961_ku/privtrim
echo "preflight host=$(hostname) start=$(date -Is)"

echo "--- quota / free space on the project filesystem ---"
df -h /dpc/kuex0095 2>&1 | head -4
echo
echo "--- project usage (top level, one level only) ---"
du -sh $RD/hf_cache $RD/out $RD/data 2>&1 | head -5
echo
echo "--- internet from a compute node ---"
curl -sS -o /dev/null -w 'huggingface.co -> %{http_code} in %{time_total}s\n' \
    --max-time 20 https://huggingface.co 2>&1 | head -2
curl -sS -o /dev/null -w 'pypi.org       -> %{http_code} in %{time_total}s\n' \
    --max-time 20 https://pypi.org 2>&1 | head -2
echo
echo "--- downloaders available ---"
for PY in $RD/env/bin/python $RD/env-ext/bin/python; do
    if [ -x "$PY" ]; then
        echo "[$PY]"
        "$PY" -c "import sys;print(' python',sys.version.split()[0])" 2>&1 | head -1
        "$PY" -c "import huggingface_hub as h;print(' hf_hub',h.__version__)" 2>&1 | head -1
        "$PY" -c "import torch;print(' torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -1
        ls "$RD/env/bin/huggingface-cli" "$RD/env/bin/hf" 2>/dev/null | head -2
    fi
done
echo
echo "--- existing Qwen snapshots ---"
ls -d $RD/hf_cache/models--Qwen--*/ 2>&1 | head -6
echo "preflight done"
