RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== job 10602443 status ==="
squeue -j 10602443 -o "%.10i %.9T %.8M" 2>&1 | tail -n 2
sacct -j 10602443 --format=JobID%12,JobName%12,State%12,Elapsed%9,ExitCode%8 2>&1 | tail -n 4
echo "=== .out tail ==="
tail -n 25 "$RD/logs/inscope-10602443.out" 2>&1
echo "=== .err (bytes / tail) ==="
wc -c "$RD/logs/inscope-10602443.err" 2>&1
tail -n 25 "$RD/logs/inscope-10602443.err" 2>&1
echo "=== outdir ==="
ls -l --time-style=+%H:%M "$RD/inscope_out/" 2>&1 | head
