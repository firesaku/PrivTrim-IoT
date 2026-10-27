#!/bin/bash
# 在 Almesbar 上提交 E2E（V100 单卡，FP16），带输入预检
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"

echo "===== 1. 输入预检 ====="
for f in "$RD/e2e.py" "$B/run/qwen25_7b/cases.jsonl.gz" "$B/project/external/squad_v11/evaluate-v2.0.py" ; do
  if [ -r "$f" ]; then echo "OK   $f"; else echo "MISS $f"; fi
done
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
for f in "$LM/config.json" "$EMB/config.json" ; do
  if [ -r "$f" ]; then echo "OK   $f"; else echo "MISS $f"; fi
done
echo
echo "===== 2. 列出 cases 与 scorer 目录 ====="
ls -l "$B/run/qwen25_7b/" 2>&1 | head -8
ls -l "$B/project/external/squad_v11/" 2>&1 | head -8

echo
echo "===== 3. 写 slurm ====="
mkdir -p "$RD/logs" "$RD/e2eout"
cat > "$RD/e2e_alm.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2ea
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2ea-%j.out
#SBATCH --error=$RD/logs/e2ea-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e.py \\
  --cases "$B/run/qwen25_7b/cases.jsonl.gz" \\
  --out "$RD/e2eout/e2e_alm_report.txt" \\
  --json "$RD/e2eout/e2e_alm_cells.json" \\
  --lm "$LM" --emb "$EMB" --scorer "$B/project/external/squad_v11/evaluate-v2.0.py" \\
  --n-per-dataset 500 --budget-frac 0.5 --max-new 64 --batch 8 \\
  --dtype fp16 --seed 20261015
rc=\$?
echo "E2EA_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/e2eout/e2e_alm_report.txt" 2>&1
echo "E2EA_DONE"
exit \$rc
EOF
chmod +x "$RD/e2e_alm.slurm"

echo "===== 4. 提交 ====="
cd "$RD"
JID=$(sbatch --parsable "$RD/e2e_alm.slurm" 2>&1 | tr -d ' \r')
echo "E2EA_JOB=$JID"
sleep 6
squeue -j "$JID" -o '%.12i %.10P %.10j %.9T %.11M %.8b %.16R' 2>&1
echo "ALM_SUBMIT_DONE"
