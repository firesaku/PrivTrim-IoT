#!/bin/bash
# 重跑 E2E（修正金标准提取），仍在 Almesbar V100 上
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
CASES="$B/run/qwen25_7b/cases.jsonl.gz"

echo "--- 确认修正后的 e2e.py 已就位 ---"
grep -n 'def gold_texts' "$RD/e2e.py" | head -2

mkdir -p "$RD/logs" "$RD/e2eout"
cat > "$RD/e2e_alm2.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2eb
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2eb-%j.out
#SBATCH --error=$RD/logs/e2eb-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e.py \\
  --cases "$CASES" \\
  --out "$RD/e2eout/e2e_b_report.txt" \\
  --json "$RD/e2eout/e2e_b_cells.json" \\
  --lm "$LM" --emb "$EMB" --scorer "$SC" \\
  --n-per-dataset 500 --budget-frac 0.5 --max-new 64 --batch 8 \\
  --dtype fp16 --seed 20261015
rc=\$?
echo "E2EB_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/e2eout/e2e_b_report.txt" 2>&1
echo "E2EB_DONE"
exit \$rc
EOF
chmod +x "$RD/e2e_alm2.slurm"
cd "$RD"
JID=$(sbatch --parsable "$RD/e2e_alm2.slurm" 2>&1 | tr -d ' \r')
echo "E2EB_JOB=$JID"
echo "$JID" > "$RD/.e2eb_jid"
sleep 5
squeue -j "$JID" -o '%.12i %.10P %.10j %.9T %.11M %.8b %.16R' 2>&1
echo "E2EB_SUBMIT_DONE"
