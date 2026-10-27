#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=prod
#SBATCH --job-name=pt-extprep
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extprep-%j.out
#
# Build the ext-arms manifest ONCE (serial) so the 32 evaluation shards never
# race on the same file. Only the two arms that still need scoring.
set -uo pipefail
RD=/dpc/kuex0095/100069961_ku/privtrim
echo "HOST=$(hostname) DATE=$(date -Is)"
$RD/env/bin/python $RD/build_ext_arms.py $RD/data/ext_arms_new.jsonl 'longllmlingua,selective_context' 2>&1 | tail -8
RC=$?
echo -n 'manifest lines: '; wc -l < $RD/data/ext_arms_new.jsonl
$RD/env/bin/python -c "import json,io; n=0
for l in io.open('$RD/data/ext_arms_new.jsonl',encoding='utf-8'):
    if l.strip(): json.loads(l); n+=1
print('manifest valid lines=%d'%n)"
echo PREP_DONE
exit $RC