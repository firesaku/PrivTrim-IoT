#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-nerprep
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/nerprep-%j.out
#
# Build the manifest for the ner_redact arm ONCE, serially, so the evaluation
# shards never race on it (that race is what killed the first extnew attempt).
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
echo "HOST=$(hostname) DATE=$(date -Is)"
$RD/env/bin/python $RD/build_ext_arms.py $RD/data/ext_arms_ner.jsonl 'ner_redact' 2>&1 | tail -6
echo -n 'lines: '; wc -l < $RD/data/ext_arms_ner.jsonl
$RD/env/bin/python -c "import json,io; n=0
for l in io.open('$RD/data/ext_arms_ner.jsonl',encoding='utf-8'):
    if l.strip(): json.loads(l); n+=1
print('valid lines=%d'%n)"
echo NERPREP_DONE