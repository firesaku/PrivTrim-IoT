#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-nerred
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/nerred-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/nerred-%j.err
#
# Off-the-shelf PII redaction baseline over all 2728 cases.
# Pure CPU: spaCy en_core_web_sm + email/phone regex. No GPU, no compression.
# Produces out/ext_full/ner_redact/<key>.json in the arm format that
# build_ext_arms.py reads (status=ok, matched.text).
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
echo "HOST=$(hostname) DATE=$(date -Is)"
START=$(date +%s)
$RD/env-ext/bin/python -u $RD/ner_redact.py 2>&1 | tail -6
RC=$?
echo "NERRED_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
echo -n 'files: '; ls $RD/out/ext_full/ner_redact/*.json 2>/dev/null | wc -l