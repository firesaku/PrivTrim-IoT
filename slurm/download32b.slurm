#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-dl32b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/dl32b-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/dl32b-%j.err
#
# Download Qwen2.5-32B-Instruct into the project cache.
#
# Run on a CPU node, not the login node: the preflight confirmed compute nodes reach the Hub
# (huggingface.co -> 200) and that /dpc has 4.9 PB free, while the project cache currently holds
# 150 GB. CPU partition is the right place for a network+IO job that needs no GPU.
#
# hf download resumes: re-submitting after a timeout continues rather than restarting, which is why
# the time limit can be generous without risking the whole download.
#
# HF_HOME is set to the project cache so the snapshot lands where every job already looks
# ($RD/hf_cache/models--Qwen--Qwen2.5-32B-Instruct/snapshots/<sha>), and NOT in $HOME.
set -uo pipefail

RD=/dpc/kuex0095/100069961_ku/privtrim
export HF_HOME=$RD/hf_cache
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
PY=$RD/env/bin/python
HF=$RD/env/bin/hf

REPO=Qwen/Qwen2.5-32B-Instruct

echo "dl32b host=$(hostname) start=$(date -Is)"
echo "--- before ---"
du -sh $RD/hf_cache 2>&1 | head -1
df -h /dpc 2>&1 | tail -1

if [ ! -x "$HF" ]; then echo "FATAL: no hf CLI at $HF"; exit 2; fi

START=$(date +%s)
"$HF" download "$REPO" \
    --exclude '*.pth' '*.gguf' '*.h5' \
    --max-workers 8 2>&1 | tail -30
RC=${PIPESTATUS[0]}

echo "--- after (rc=$RC, elapsed=$(( $(date +%s) - START ))s) ---"
du -sh $RD/hf_cache 2>&1 | head -1
ls -d $RD/hf_cache/models--Qwen--Qwen2.5-32B-Instruct/snapshots/* 2>&1 | head -2
du -sh $RD/hf_cache/models--Qwen--Qwen2.5-32B-Instruct 2>&1 | head -1
echo "dl32b rc=$RC done"
exit $RC
