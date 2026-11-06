#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/probe-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/probe-%j.err
#
# Inventory the model cache from a compute node (the login node is not for this).
#
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
OUT=$RD/hf_cache_inventory.txt
{
  echo "host=$(hostname)  date=$(date -Is)"
  echo
  echo "=== top-level entries in HF_HOME ==="
  ls -1 "$RD/hf_cache" 2>/dev/null
  echo
  echo "=== models--* with size ==="
  du -sh "$RD/hf_cache"/models--* 2>/dev/null | sort -k1 -h
  echo
  echo "=== snapshot dirs (loadable paths) ==="
  ls -d "$RD/hf_cache"/models--*/snapshots/* 2>/dev/null
} > "$OUT" 2>&1
echo "wrote $OUT"
tail -40 "$OUT"