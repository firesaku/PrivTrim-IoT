#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-nerred
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/nerred-%j.out
#SBATCH --error=/path/to/project/logs/nerred-%j.err
#
# Off-the-shelf PII redaction baseline over all 2728 cases.
# Pure CPU: spaCy en_core_web_sm + email/phone regex. No GPU, no compression.
# Produces out/ext_full/ner_redact/<key>.json in the arm format that
# build_ext_arms.py reads (status=ok, matched.text).
set -uo pipefail
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
echo "HOST=$(hostname) DATE=$(date -Is)"
START=$(date +%s)
$RD/env-ext/bin/python -u $RD/ner_redact.py 2>&1 | tail -6
RC=$?
echo "NERRED_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
echo -n 'files: '; ls $RD/out/ext_full/ner_redact/*.json 2>/dev/null | wc -l