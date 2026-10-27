#!/bin/bash
# 在 <CLUSTER> 上建目录、解析 python、写 slurm 并提交（CPU/prod，不占 GPU）
set -uo pipefail

RD=/path/to/100065732/privtrim_v2c_20261015
mkdir -p "$RD/logs"

echo "===== 1. resolve python with numpy ====="
PY=""
for cand in \
  /path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python \
  /path/to/100065732/privtrim_iot_hpc_20261005/.venv-main/bin/python \
  "$(command -v python3 2>/dev/null)" ; do
  [ -n "$cand" ] || continue
  [ -x "$cand" ] || continue
  if "$cand" -c "import numpy,sys; print('NUMPY_OK', numpy.__version__)" 2>/dev/null; then
    PY="$cand"; echo "PICKED $PY"; break
  fi
done
if [ -z "$PY" ]; then echo "NO_PYTHON_WITH_NUMPY"; exit 3; fi

echo "===== 2. files present? ====="
ls -l "$RD" 2>&1

echo "===== 3. write slurm ====="
cat > "$RD/v2c.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-v2c
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2c-%j.out
#SBATCH --error=$RD/logs/v2c-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
echo "PY=$PY"
cd "$RD"
PYTHONUNBUFFERED=1 "$PY" -u v2c_feasibility.py \\
  --samples "$RD/frozen_samples.jsonl" \\
  --out "$RD/v2c_report.txt" \\
  --json "$RD/v2c_cells.json" \\
  --ratios 0.125,0.25,0.5 \\
  --cap 200 \\
  --seed 20261015
rc=\$?
echo "V2C_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/v2c_report.txt" 2>&1
echo "V2C_DONE"
exit \$rc
EOF
chmod +x "$RD/v2c.slurm"
echo "--- slurm written ---"
head -12 "$RD/v2c.slurm"

echo "===== 4. submit ====="
cd "$RD"
sbatch "$RD/v2c.slurm" 2>&1
sleep 3
echo "===== 5. queue ====="
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %R' 2>&1
echo "SETUP_DONE"
