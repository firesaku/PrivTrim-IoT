RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo '=== e2e4 logs ==='
ls -l "$RD"/logs/e2e4-* 2>&1
echo '=== e2e4 out tail ==='
tail -40 "$RD"/logs/e2e4-*.out 2>&1
echo '=== e2e4 err tail ==='
tail -25 "$RD"/logs/e2e4-*.err 2>&1
echo '=== all queue (any user, my account) ==='
squeue -A kuex0095 -o '%.12i %.10u %.9P %.12j %.9T %.11M %R' 2>&1 | head -15
echo '=== sacct last 10 jobs ==='
sacct -u "$(id -un)" --starttime today -o JobID,JobName%12,State,Elapsed,ExitCode 2>&1 | grep -vE '\.(ba|ex)' | tail -12
echo 'DIAG_DONE'
