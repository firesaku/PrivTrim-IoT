#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-dacwd5
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=18:00:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/dacwd5-%j.out
#
# WATCHDOG v5 - watches the 15-minute-shard compression job 1489726.
#
# Fires evaluation when: ceiling reached (1699), or the queue drains with a
# usable amount done, or compression stalls for 90 min (36 x 2.5 min).
# The stall window is deliberately long: our GPU access is intermittent
# (0-3 shards, priority 3376 vs others at 4058), so a short window would fire
# evaluation on a tiny partial set and then compete with compression for GPUs.
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
TARGET=1699
COMPRESS_JOB=1489726
echo "wd5 start $(date -Is) target=$TARGET watching=$COMPRESS_JOB"
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
  if [ "$N" -ge "$TARGET" ] || { [ "$LEFT" -eq 0 ] && [ "$N" -gt 1200 ]; } || [ "$STALL" -ge 36 ]; then
    echo "HANDOFF ok=$N tasks=$LEFT stall=$STALL at $(date -Is)"
    sleep 20
    cd $RD
    JID=$(sbatch --parsable dacev27.slurm 2>&1)
    echo "submitted dacev27: $JID"
    echo WD5_DONE
    exit 0
  fi
  sleep 150
done
echo "WD5 TIMEOUT ok=$N"
exit 1