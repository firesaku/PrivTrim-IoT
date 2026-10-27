#!/bin/bash
# 重跑修正后的 V1（限制 digitchunk 只对 phone），并查 V3 数组
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python

cat > "$RD/v1.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-v1fix
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v1fix-%j.out
#SBATCH --error=$RD/logs/v1fix-%j.err
set -uo pipefail
cd "$RD"
PYTHONUNBUFFERED=1 "$PY" -u v1_checker.py --out "$RD/v1fix_report.txt" --json "$RD/v1fix_cells.json"
echo "V1FIX_RC=\$?"
EOF
chmod +x "$RD/v1.slurm"
cd "$RD"
JID=$(sbatch --parsable "$RD/v1.slurm" 2>&1 | tr -d ' \r')
echo "V1FIX_JOB=$JID"
for i in $(seq 1 40); do
  st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null | tr -d ' \r')
  [ -z "$st" ] && break
  sleep 5
done
echo "===== V1 修正后结果 ====="
cat "$RD/v1fix_report.txt" 2>&1
echo
echo "===== V3 数组状态 ====="
squeue -u "$(id -un)" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
ls -l "$RD/v3out" 2>&1
echo "V1FIX_DONE"
