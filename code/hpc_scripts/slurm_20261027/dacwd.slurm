#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dacwd
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=14:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacwd-%j.out
#
# DAC COMPLETION WATCHDOG.
#
# Compression (1489637) runs on whatever GPUs we can get and will take hours.
# The evaluation step (daceval.slurm) must run AFTER it, and starting it by hand
# costs a whole round of latency. So this cheap CPU-only job polls the compression
# output and submits the evaluation the moment compression is done.
#
# Condition: dac compressed files >= 2700 (allow a small shortfall) OR
#            the compression array no longer has RUNNING/PENDING tasks.
set -uo pipefail
RD=/path/to/project
TARGET=2700
COMPRESS_JOB=1489637
echo "watchdog start $(date -Is) target=$TARGET"
for i in $(seq 1 800); do
  N=$(ls $RD/out/ext_full/dac/*.json 2>/dev/null | wc -l)
  LEFT=$(squeue -j $COMPRESS_JOB -h 2>/dev/null | wc -l)
  if [ $((i % 20)) -eq 0 ]; then echo "  [$(date +%H:%M:%S)] compressed=$N/$TARGET  compress_tasks=$LEFT"; fi
  if [ "$N" -ge "$TARGET" ] || { [ "$LEFT" -eq 0 ] && [ "$N" -gt 500 ]; }; then
    echo "COMPRESSION DONE compressed=$N at $(date -Is)"
    sleep 30
    cd $RD
    JID=$(sbatch --parsable daceval.slurm 2>&1)
    echo "submitted daceval: $JID"
    echo WD_DONE
    exit 0
  fi
  sleep 60
done
echo "WATCHDOG TIMEOUT at $(date -Is) compressed=$N"
exit 1