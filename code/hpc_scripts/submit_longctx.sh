#!/bin/bash
# Almesbar: long-context E2E on the 112 deterministic longctx cases (842-4880 words).
set -uo pipefail
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
mkdir -p "$RD/logs" "$RD/longctx_out"

echo "===== inputs ====="
for f in "$RD/e2e_longctx.py" "$RD/longctx_cases.jsonl" "$SC" "$LM/config.json" "$EMB/config.json"; do
  [ -r "$f" ] && echo "OK   $f" || echo "MISS $f"
done
wc -l "$RD/longctx_cases.jsonl" 2>&1
"$PY" -c "import ast;ast.parse(open('$RD/e2e_longctx.py',encoding='utf-8').read());print('AST_OK e2e_longctx')" 2>&1

cat > "$RD/longctx.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-longctx
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/longctx-%j.out
#SBATCH --error=$RD/logs/longctx-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e_longctx.py \\
  --cases "$RD/longctx_cases.jsonl" \\
  --out "$RD/longctx_out/report.txt" \\
  --json "$RD/longctx_out/cells.json" \\
  --percase "$RD/longctx_out/percase.jsonl" \\
  --lm "$LM" --scorer "$SC" --emb "$EMB" \\
  --arms all --budgets 0.25,0.5,0.75 \\
  --dtype fp16 --batch 4 --maxlen 8192 --max-new 64 --seed 20261015
rc=\$?
echo "LONGCTX_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/longctx_out/report.txt" 2>&1
echo "LONGCTX_DONE"
exit \$rc
EOF
chmod +x "$RD/longctx.slurm"

cd "$RD"
J=$(sbatch --parsable "$RD/longctx.slurm" 2>&1); echo "ALM_LONGCTX_JOB=$J"
sleep 5
squeue -u "$(id -un)" -o '%.12i %.9P %.10j %.8T %.10M' 2>&1 | head -8
echo "LONGCTX_SUBMIT_DONE"
