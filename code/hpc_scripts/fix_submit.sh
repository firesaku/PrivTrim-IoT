RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== fix shebang ==="
head -n 1 "$RD/mistral.slurm"
sed -i '1i #!/bin/bash' "$RD/mistral.slurm"
echo "--- after ---"
head -n 3 "$RD/mistral.slurm"
echo "=== sbatch ==="
sbatch "$RD/mistral.slurm" 2>&1
echo "=== queue ==="
squeue --me -o "%.10i %.16j %.9T %.8M %.10R" 2>&1 | head
echo
echo "=== aliaslink progress ==="
tail -n 3 "$RD/logs/aliaslink-10602422.out" 2>&1
echo
echo "=== dry-run 0.10/0.15 table ==="
cat /tmp/dry_x.txt 2>&1 | head -n 60
