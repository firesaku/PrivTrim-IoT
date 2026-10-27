#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dacwd3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=18:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacwd3-%j.out
#
# DAC COMPLETION WATCHDOG v3 - target corrected to the ACHIEVABLE ceiling.
#
# v1/v2 used TARGET=2700, which is unreachable: DAC needs an attention matrix
# that grows with the SQUARE of the context, and with ~8.7 GB of a shared H200
# it OOMs on anything above ~9000 tokens. The benchmark has 1029 such cases
# (37.7%), so the true ceiling is 1699.
#
# v3 therefore waits for the feasible set (1699), not the whole benchmark.
# It also stops waiting once compression stops making progress, so a permanently
# stalled queue still hands off to evaluation rather than hanging for 18 h.
set -uo pipefail
RD=/path/to/project
TARGET=1699
COMPRESS_JOB=1489637
echo "wd3 start $(date -Is) target=$TARGET (DAC-attention feasibility ceiling)"
PREV=0; STALL=0
for i in $(seq 1 1080); do
  N=$($RD/env/bin/python -c "
import json,io,glob
c=0
for f in glob.glob('$RD/out/ext_full/dac/*.json'):
    try: r=json.load(io.open(f,encoding='utf-8'))
    except Exception: continue
    if r.get('status')=='ok': c+=1
print(c)" 2>/dev/null)
  N=${N:-0}
  LEFT=$(squeue -j $COMPRESS_JOB -h 2>/dev/null | wc -l)
  if [ "$N" -gt "$PREV" ]; then STALL=0; else STALL=$((STALL+1)); fi
  PREV=$N
  if [ $((i % 15)) -eq 0 ]; then echo "  [$(date +%H:%M:%S)] ok=$N/$TARGET tasks=$LEFT stall=$STALL"; fi
  # fire when the feasible ceiling is reached, or when the queue is empty,
  # or when compression has made no progress for 40 minutes (2.5 min x 16)
  if [ "$N" -ge "$TARGET" ] || { [ "$LEFT" -eq 0 ] && [ "$N" -gt 800 ]; } || [ "$STALL" -ge 16 ]; then
    echo "HANDOFF ok=$N tasks=$LEFT stall=$STALL at $(date -Is)"
    sleep 20
    cd $RD
    JID=$(sbatch --parsable dacev27.slurm 2>&1)
    echo "submitted dacev27: $JID"
    echo WD3_DONE
    exit 0
  fi
  sleep 150
done
echo "WD3 TIMEOUT ok=$N"
exit 1