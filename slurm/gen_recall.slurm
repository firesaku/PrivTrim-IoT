#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-genrecall
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/genrecall-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/genrecall-%j.err
#
# Generate the degraded-detector case files ON THE CLUSTER from data that is already there.
#
# Why not upload them: they are 94 MB derived from longctx_max.jsonl, which is already on the node.
# Regenerating is deterministic (seed 20261108, per-case seed = key|rho), takes about two minutes on
# one core, and keeps the login node out of it -- this is a CPU job on prod, not an ssh-time script.
#
# The recall array depends on this job finishing: sbatch --dependency=afterok:<this job> slurm/recall.slurm
set -uo pipefail

RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH=$RD/code:$RD/code/longctx
PY=$RD/env/bin/python

CASES=$RD/data/longctx_max.jsonl
if [ ! -f "$CASES" ]; then echo "FATAL: missing $CASES"; exit 2; fi

cd $RD/code/longctx || exit 1
echo "genrecall host=$(hostname) start=$(date -Is)"
$PY -u imperfect_detector.py \
  --cases  "$CASES" \
  --outdir $RD/data/recall \
  --ratios 1.0,0.995,0.99,0.98,0.95,0.90
RC=$?
echo "genrecall RC=$RC"
ls -l $RD/data/recall 2>&1 | head -10
exit $RC
