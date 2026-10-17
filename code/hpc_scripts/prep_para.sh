RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
for V in A B; do
  mkdir -p "$RD/para${V}_out"
  cat > "$RD/para${V}.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-para${V}
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/para${V}-%j.out
#SBATCH --error=$RD/logs/para${V}-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$B/.venv-main/bin/python" -u e2e_longctx_r5.py \\
--cases "$RD/longctx_para${V}.jsonl" \\
--out "$RD/para${V}_out/report.txt" \\
--percase "$RD/para${V}_out/percase.jsonl" \\
--json "$RD/para${V}_out/cells.json" \\
--lm "$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" \\
--scorer "$B/project/external/squad_v11/evaluate-v2.0.py" \\
--arms full,budget_cluster,budget_cluster_link,alias_budget,alias_budget_link \\
--budgets 0.5 \\
--maxlen 24576 --maxlen-policy raise --sel lex \\
--dtype fp16 --batch 2 --max-new 64 --seed 20261015 \\
--alias-question
rc=\$?
echo "PARA${V}_RC=\$rc"
echo "PARA${V}_DONE"
exit \$rc
EOF
  head -n 1 "$RD/para${V}.slurm"
  echo "para${V} bytes=$(wc -c < "$RD/para${V}.slurm")"
done
echo "=== inputs ==="
wc -l "$RD/longctx_paraA.jsonl" "$RD/longctx_paraB.jsonl" 2>&1
