#!/bin/bash
# Almesbar: v3 MERGE-MATCHED comparison -- rand4_emb stops at exactly as many
# accepted unions as bound_emb achieved on the same record, so the only remaining
# difference is the selection criterion (cluster size already matched).
set -uo pipefail
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
mkdir -p "$RD/logs" "$RD/v3out3"

echo "===== verify patched v3 + interface ====="
"$PY" -c "import ast;src=open('$RD/v3_batch.py',encoding='utf-8').read();ast.parse(src);print('AST_OK rand_stop',('--rand-stop' in src),'matched',('matched' in src))" 2>&1
"$PY" "$RD/v3_batch.py" --help 2>&1 | grep -E 'rand-stop|rand-min-gap' | head -4

cat > "$RD/v3match.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v3match
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v3match-%A_%a.out
#SBATCH --error=$RD/logs/v3match-%A_%a.err
set -uo pipefail
TAGS=(matched_gap2 matched_gap0 matched_nomaxsize)
EX=("--rand-stop matched --rand-min-gap 2" "--rand-stop matched --rand-min-gap 0" "--rand-stop matched")
i=\$SLURM_ARRAY_TASK_ID
TG=\${TAGS[\$i]}; E=\${EX[\$i]}
echo "HOST=\$(hostname) TASK=\$i TAG=\$TG EXTRA=[\$E] DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RD/v3out3/\$TG"
PYTHONUNBUFFERED=1 "$PY" -u v3_batch.py \\
  --config bound --datadir "$RD" --outdir "$RD/v3out3/\$TG" --model "$EMB" \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015 \$E
rc=\$?
echo "V3MATCH_RC=\$rc TAG=\$TG"
echo "===== REPORT (\$TG) ====="
cat "$RD/v3out3/\$TG/v3_bound_report.txt" 2>&1
exit \$rc
EOF
chmod +x "$RD/v3match.slurm"

cd "$RD"
J=$(sbatch --parsable --array=0-2 "$RD/v3match.slurm" 2>&1); echo "ALM_V3MATCH_JOB=$J"
sleep 5
squeue -u "$(id -un)" -o '%.14i %.8P %.10j %.8T %.10M' 2>&1 | head -8
echo "V3MATCH_SUBMIT_DONE"
