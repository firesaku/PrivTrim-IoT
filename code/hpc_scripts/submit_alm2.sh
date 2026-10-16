#!/bin/bash
# Almesbar (V100) submissions:
#   1) v2d pair-budget fix, both modes  (fixed: v2d_gpu.py has no --dtype)
#   2) e2e4 same-budget selector comparison
set -uo pipefail
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
CASES="$B/run/qwen25_7b/cases.jsonl.gz"
DB="$B/run/qwen25_7b/results.sqlite"
mkdir -p "$RD/logs" "$RD/e2eout"

echo "===== input check ====="
for f in "$RD/v2d_gpu.py" "$RD/e2e4.py" "$CASES" "$SC" "$DB" "$LM/config.json" "$EMB/config.json"; do
  [ -r "$f" ] && echo "OK   $f" || echo "MISS $f"
done

echo "===== v2d argparse surface (no --dtype expected) ====="
"$PY" "$RD/v2d_gpu.py" --help 2>&1 | tr ',' '\n' | grep -E '^\s*--' | head -20

echo "===== write v2d slurm (1xV100) ====="
cat > "$RD/v2d_fix_alm.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v2dalm
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2dalm-%j.out
#SBATCH --error=$RD/logs/v2dalm-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
for M in legacy disjoint; do
  echo "########## MODE=\$M ##########"
  PYTHONUNBUFFERED=1 "$PY" -u v2d_gpu.py \\
    --samples "$RD/frozen_samples.jsonl" \\
    --out "$RD/v2d_alm_report_\$M.txt" \\
    --json "$RD/v2d_alm_cells_\$M.json" \\
    --model "$EMB" \\
    --ratios 0.125,0.25,0.5 \\
    --cap 200 --batch 64 --maxlen 256 --seed 20261015 \\
    --pair-mode \$M
  echo "V2DALM_\${M}_RC=\$?"
done
echo "V2DALM_DONE"
EOF
chmod +x "$RD/v2d_fix_alm.slurm"

echo "===== write e2e4 slurm (1xV100) ====="
cat > "$RD/e2e4.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2e4
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2e4-%j.out
#SBATCH --error=$RD/logs/e2e4-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e4.py \\
  --cases "$CASES" --db "$DB" \\
  --out "$RD/e2eout/e2e4_report.txt" \\
  --json "$RD/e2eout/e2e4_cells.json" \\
  --percase "$RD/e2eout/e2e4_percase.jsonl" \\
  --lm "$LM" --emb "$EMB" --scorer "$SC" \\
  --n-per-dataset 500 --reduce-frac 0.25 --max-new 64 --batch 8 \\
  --dtype fp16 --seed 20261015
rc=\$?
echo "E2E4_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/e2eout/e2e4_report.txt" 2>&1
echo "E2E4_DONE"
exit \$rc
EOF
chmod +x "$RD/e2e4.slurm"

echo "===== submit ====="
cd "$RD"
J1=$(sbatch --parsable "$RD/v2d_fix_alm.slurm" 2>&1); echo "ALM_V2D_JOB=$J1"
J2=$(sbatch --parsable "$RD/e2e4.slurm" 2>&1); echo "ALM_E2E4_JOB=$J2"
sleep 5
squeue -u "$(id -un)" -o '%.12i %.10P %.12j %.9T %.11M %.8b %R' 2>&1
echo "ALM_SUBMIT2_DONE"
