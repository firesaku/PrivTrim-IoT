#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dacwd2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=16:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacwd2-%j.out
#
# DAC COMPLETION WATCHDOG v2.
#
# v1 was cancelled because it pointed at daceval.slurm, whose 4-shard / 15-min
# sizing was built for the old 179-case gap and would time out on the current
# ~2500-case gap. v2 submits dacev27.slurm (32 shards x 20 min) instead.
#
# Submits in STAGES so the largest shards get the GPU time as they become ready:
# whenever the number of un-evaluated compressed cases crosses a threshold, fire
# one more evaluation array. Idempotent - each array recomputes its own gap.
set -uo pipefail
RD=/path/to/project
TARGET=2700
COMPRESS_JOB=1489637
LAST_N=0
FIRED=0
echo "wd2 start $(date -Is) target=$TARGET"
for i in $(seq 1 960); do
  N=$(ls $RD/out/ext_full/dac/*.json 2>/dev/null | wc -l)
  LEFT=$(squeue -j $COMPRESS_JOB -h 2>/dev/null | wc -l)
  if [ $((i % 15)) -eq 0 ]; then echo "  [$(date +%H:%M:%S)] compressed=$N/$TARGET tasks=$LEFT"; fi
  # Fire an evaluation pass when compression has advanced a lot or finished.
  if [ "$FIRED" -eq 0 ] && { [ "$N" -ge "$TARGET" ] || { [ "$LEFT" -eq 0 ] && [ "$N" -gt 500 ]; }; }; then
    echo "COMPRESSION DONE compressed=$N at $(date -Is)"
    sleep 20
    cd $RD
    JID=$(sbatch --parsable dacev27.slurm 2>&1)
    echo "submitted dacev27: $JID"
    FIRED=1
    echo WD2_DONE
    exit 0
  fi
  sleep 60
done
echo "WD2 TIMEOUT at $(date -Is) compressed=$N"
exit 1