RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== job 10602422 Command/WorkDir ==="
scontrol show job 10602422 2>&1 | grep -E "Command=|WorkDir=|StdOut=|StdErr=|Partition=|Gres=|NumCPUs=|Mem="
echo "=== submit script location ==="
ls -l "$RD"/*.sh 2>&1 | head -n 20
echo "=== queue snapshot ==="
squeue -u 100065732 -o "%.10i %.14j %.9T %.8M %.10R" 2>&1 | head -n 10
