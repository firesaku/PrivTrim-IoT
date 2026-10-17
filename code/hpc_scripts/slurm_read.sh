RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== aliaslink.slurm ==="
cat "$RD/aliaslink.slurm" 2>&1
echo "=== squeue --me ==="
squeue --me -o "%.10i %.14j %.9T %.8M %.10R" 2>&1 | head
echo "=== job state ==="
squeue -j 10602422 -o "%.10i %.9T %.10M" 2>&1 | tail -n 1
echo "=== out tail ==="
tail -n 6 "$RD/logs/aliaslink-10602422.out" 2>&1
echo "=== outdir ==="
ls -l --time-style=+%H:%M:%S "$RD/aliaslink_out/" 2>&1 | head
