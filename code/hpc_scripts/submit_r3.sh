#!/bin/bash
# Almesbar: longctx RERUN with the truncation defects fixed.
#   maxlen 24576 (was 8192)  -> zero truncation on all 112 cases (verified with real tokenizer)
#   --maxlen-policy raise     -> refuse to silently cut instead of truncating
#   budget capped by maxlen   -> no arm can exceed what the model sees
#   prompt_tokens per row     -> any future truncation is visible in the artefacts
set -uo pipefail
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
mkdir -p "$RD/logs" "$RD/longctx_r3_out"

echo "===== preflight ====="
"$PY" -c "import ast;src=open('$RD/e2e_longctx_r3.py',encoding='utf-8').read();ast.parse(src);print('AST_OK maxlen24576',('default=24576' in src),'policy',('--maxlen-policy' in src),'prompt_tokens',('prompt_tokens=prompt_tokens' in src or 'prompt_tokens=(prompt_tokens' in src))" 2>&1
"$PY" "$RD/e2e_longctx_r3.py" --help 2>&1 | grep -E 'maxlen|policy' | head -4

cat > "$RD/longctx_r3.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-lctxr3
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/lctxr3-%j.out
#SBATCH --error=$RD/logs/lctxr3-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u e2e_longctx_r3.py \\
  --cases "$RD/longctx_cases.jsonl" \\
  --out "$RD/longctx_r3_out/report.txt" \\
  --json "$RD/longctx_r3_out/cells.json" \\
  --percase "$RD/longctx_r3_out/percase.jsonl" \\
  --lm "$LM" --scorer "$SC" \\
  --arms all --budgets 0.25,0.5,0.75 \\
  --maxlen 24576 --maxlen-policy raise --sel lex \\
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
rc=\$?
echo "LCTXR3_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/longctx_r3_out/report.txt" 2>&1
echo "LCTXR3_DONE"
exit \$rc
EOF
chmod +x "$RD/longctx_r3.slurm"

cd "$RD"
J=$(sbatch --parsable "$RD/longctx_r3.slurm" 2>&1); echo "ALM_LCTXR3_JOB=$J"
sleep 6
squeue -u "$(id -un)" -h -o '%i %j %T %M' 2>&1 | head -4
echo "LCTXR3_SUBMIT_DONE"
