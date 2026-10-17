RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== squeue ==="
squeue -u 100065732 -o "%.10i %.14j %.9T %.10M %.12R" 2>&1 | head -n 20
echo "=== sacct (last 2 days) ==="
sacct -u 100065732 --starttime now-2days --format=JobID%12,JobName%16,State%12,Elapsed%10,ExitCode%8,NodeList%12 2>&1 | tail -n 15
echo "=== outdir ==="
ls -la "$RD/aliaslink_out" 2>&1 | head -n 20
echo "=== logs ==="
ls -lat "$RD/logs" 2>&1 | head -n 12
echo "=== log tail ==="
for f in $(ls -t "$RD"/logs/*aliaslink* 2>/dev/null | head -n 2); do echo "--- $f"; tail -n 15 "$f"; done
echo "=== gpu ==="
scontrol show job 10602422 2>&1 | head -n 5
