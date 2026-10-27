#!/bin/bash
# Almesbar: e2e5 -- router arms on conflict cases (the fourth-quadrant test).
set -uo pipefail
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
mkdir -p "$RD/logs" "$RD/e2eout"

echo "===== input check ====="
for f in "$RD/e2e5.py" "$RD/v4_variants.jsonl" "$RD/conflict_cases.jsonl" "$SC" "$LM/config.json"; do
  [ -r "$f" ] && echo "OK   $f" || echo "MISS $f"
done
wc -l "$RD/v4_variants.jsonl"

cat > "$RD/e2e5.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2e5
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2e5-%j.out
#SBATCH --error=$RD/logs/e2e5-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e5.py \\
  --variants "$RD/v4_variants.jsonl" \\
  --out "$RD/e2eout/e2e5_report.txt" \\
  --json "$RD/e2eout/e2e5_cells.json" \\
  --percase "$RD/e2eout/e2e5_percase.jsonl" \\
  --lm "$LM" --scorer "$SC" \\
  --max-new 64 --batch 8 --dtype fp16
rc=\$?
echo "E2E5_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/e2eout/e2e5_report.txt" 2>&1
echo "E2E5_DONE"
exit \$rc
EOF
chmod +x "$RD/e2e5.slurm"

cd "$RD"
J=$(sbatch --parsable "$RD/e2e5.slurm" 2>&1); echo "ALM_E2E5_JOB=$J"
sleep 4
squeue -u "$(id -un)" -o '%.12i %.9P %.12j %.9T %.11M %.8b %R' 2>&1
echo "E2E5_SUBMIT_DONE"
