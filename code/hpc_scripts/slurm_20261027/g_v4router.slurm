#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-v4r
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/v4r-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/v4r-%j.err
#
# ç¬¬å››è±¡é™ï¼šv4_router è·¯ç”± + éšç§æŒ‡æ ‡ï¼ˆCPUï¼‰ï¼Œäº§å‡º v4_variants.jsonl ä¾› e2e5
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code
PY=$RD/env/bin/python
cd $RD/code || exit 1
echo "HOST=$(hostname) START=$(date -Is)"
$PY -u v4_router.py \
  --cases $RD/data/conflict_cases.jsonl \
  --out $RD/out/v4_report.txt \
  --json $RD/out/v4_cells.json \
  --variants $RD/data/v4_variants.jsonl
RC=$?
echo V4R_RC=$RC
tail -30 $RD/out/v4_report.txt 2>/dev/null
exit $RC
