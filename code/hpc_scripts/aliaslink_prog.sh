RD=/path/to/100065732/privtrim_v2c_20261015
date -Iseconds
echo "=== elapsed ==="
squeue -j 10602422 -o "%.10i %.9T %.10M" 2>&1 | tail -n 2
echo "=== log files ==="
ls -l --time-style=+%H:%M:%S "$RD"/logs/aliaslink-10602422.* 2>&1
echo "=== .out full ==="
cat "$RD"/logs/aliaslink-10602422.out 2>&1 | tail -n 30
echo "=== .err tail ==="
tail -n 6 "$RD"/logs/aliaslink-10602422.err 2>&1
echo "=== outdir ==="
ls -l --time-style=+%H:%M:%S "$RD"/aliaslink_out/ 2>&1
echo "=== gpu util ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>&1 | head -n 3
