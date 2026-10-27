RD=/path/to/100065732/privtrim_v2c_20261015
echo '=== queue ==='
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %R' 2>&1
echo '=== job 10601686 ==='
sacct -j 10601686 -o JobID,JobName%14,Partition,State,Elapsed,ExitCode 2>&1 | head -6
echo '=== v2dalm out tail ==='
for f in "$RD"/logs/v2dalm-*.out; do [ -f "$f" ] && echo "--- $f ---" && tail -25 "$f"; done
echo '=== v2dalm err tail ==='
for f in "$RD"/logs/v2dalm-*.err; do [ -f "$f" ] && echo "--- $f ---" && tail -15 "$f"; done
echo '=== result files ==='
ls -l "$RD"/v2d_alm_* 2>&1
echo '=== ALM_STATUS_DONE ==='
