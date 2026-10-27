#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-dacwd4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=18:00:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/dacwd4-%j.out
#
# DAC COMPLETION WATCHDOG v4.
#  - watches compression job 1489705 (feasible subset, 1699 cases)
#  - target 1699, the DAC-attention feasibility ceiling (see dacfeas.slurm)
#  - hands off to dacev27.slurm when the ceiling is hit, the queue drains,
#    or compression stalls for 40 min
set -uo pipefail
RD=/path/to/project
TARGET=1699
COMPRESS_JOB=1489705
echo "wd4 start $(date -Is) target=$TARGET watching=$COMPRESS_JOB"
PREV=0; STALL=0
for i in $(seq 1 1080); do
  N=$($RD/env/bin/python -c "
import json,io,glob
feas=set()
for x in io.open('$RD/data/longctx_dac_feasible.jsonl',encoding='utf-8'):
    if x.strip(): feas.add(json.loads(x)['key'])
c=0
for f in glob.glob('$RD/out/ext_full/dac/*.json'):
    k=f.split('/')[-1][:-5]
    if k not in feas: continue
    try: r=json.load(io.open(f,encoding='utf-8'))
    except Exception: continue
    if r.get('status')=='ok': c+=1
print(c)" 2>/dev/null)
  N=${N:-0}
  LEFT=$(squeue -j $COMPRESS_JOB -h 2>/dev/null | wc -l)
  if [ "$N" -gt "$PREV" ]; then STALL=0; else STALL=$((STALL+1)); fi
  PREV=$N
  if [ $((i % 12)) -eq 0 ]; then echo "  [$(date +%H:%M:%S)] ok=$N/$TARGET tasks=$LEFT stall=$STALL"; fi
  if [ "$N" -ge "$TARGET" ] || { [ "$LEFT" -eq 0 ] && [ "$N" -gt 1000 ]; } || [ "$STALL" -ge 16 ]; then
    echo "HANDOFF ok=$N tasks=$LEFT stall=$STALL at $(date -Is)"
    sleep 20
    cd $RD
    JID=$(sbatch --parsable dacev27.slurm 2>&1)
    echo "submitted dacev27: $JID"
    echo WD4_DONE
    exit 0
  fi
  sleep 150
done
echo "WD4 TIMEOUT ok=$N"
exit 1