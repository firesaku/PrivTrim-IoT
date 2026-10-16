#!/bin/bash
# 多种子方差实验（seed 20261016 / 20261017；r3 主结果用的是 20261015）
# + 安装巡检脚本
set -uo pipefail
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM="$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
mkdir -p "$RD/logs"

echo "===== 1. 集群空闲情况 ====="
sinfo -p gpu -o '%20P %6D %6t %G' 2>&1 | head -6
echo "--- 我的队列 ---"
squeue -u "$(id -un)" -h -o '%i %j %T %M' 2>&1 | head -5

echo "===== 2. 写多种子 array slurm ====="
cat > "$RD/lctxseed.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-lctxseed
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/lctxseed-%A_%a.out
#SBATCH --error=$RD/logs/lctxseed-%A_%a.err
set -uo pipefail
SEEDS=(20261016 20261017)
S=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
echo "HOST=\$(hostname) TASK=\$SLURM_ARRAY_TASK_ID SEED=\$S DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RD/seedout/\$S"
PYTHONUNBUFFERED=1 "$PY" -u e2e_longctx_r3.py \\
  --cases "$RD/longctx_cases.jsonl" \\
  --out "$RD/seedout/\$S/report.txt" \\
  --json "$RD/seedout/\$S/cells.json" \\
  --percase "$RD/seedout/\$S/percase.jsonl" \\
  --lm "$LM" --scorer "$SC" \\
  --arms all --budgets 0.25,0.5,0.75 \\
  --maxlen 24576 --maxlen-policy raise --sel lex \\
  --dtype fp16 --batch 2 --max-new 64 --seed \$S
rc=\$?
echo "LCTXSEED_RC=\$rc SEED=\$S"
exit \$rc
EOF
chmod +x "$RD/lctxseed.slurm"

echo "===== 3. 安装巡检脚本 ====="
cat > "$RD/patrol.sh" <<'PEOF'
#!/bin/bash
# 巡检：一眼看清每个作业是"在跑 / 卡住 / 死了 / 完成"
# 用法：bash patrol.sh
R=/dpc/kuex0095/100065732/privtrim_v2c_20261015
NOW=$(date -u +%s)
echo "=== PATROL $(date -u '+%Y-%m-%d %H:%M:%S') UTC ==="
echo "--- 队列 ---"
squeue -u 100065732 -h -o '%i|%j|%T|%M|%R' 2>&1 | head -12
echo "--- 最近作业终态 ---"
sacct -u 100065732 --starttime today -o JobID,JobName%14,State,ExitCode,Elapsed 2>&1 | grep -vE '\.(ba|ex)' | tail -8
echo "--- 日志新鲜度（>900s 未更新且在跑 = 可疑） ---"
for f in $(ls -t "$R"/logs/*.out 2>/dev/null | head -6); do
  AGE=$(( NOW - $(stat -c %Y "$f") ))
  printf '%-46s age=%5ds  last=%s\n' "$(basename $f)" "$AGE" "$(tail -1 "$f" 2>/dev/null | cut -c1-90)"
done
echo "--- err 非空的作业 ---"
for f in $(ls -t "$R"/logs/*.err 2>/dev/null | head -10); do
  SZ=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$SZ" -gt 800 ]; then
    printf '%-46s %6sB  %s\n' "$(basename $f)" "$SZ" "$(tail -1 "$f" | cut -c1-90)"
  fi
done
echo "--- 产物 ---"
for d in longctx_r3_out seedout v3out3 v3out2 longctx_r2_out; do
  n=$(ls "$R/$d" 2>/dev/null | wc -l)
  printf '%-18s %s files\n' "$d" "$n"
done
echo "=== PATROL_DONE ==="
PEOF
chmod +x "$RD/patrol.sh"
echo "patrol.sh installed"

echo "===== 4. 提交多种子 array ====="
cd "$RD"
J=$(sbatch --parsable --array=0-1 "$RD/lctxseed.slurm" 2>&1); echo "ALM_SEED_JOB=$J"
sleep 6
squeue -u "$(id -un)" -h -o '%i %j %T %M' 2>&1 | head -6
echo "SEED_SUBMIT_DONE"
