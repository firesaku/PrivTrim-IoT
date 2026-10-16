RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo '=== cancel impossible interactive job ==='
scancel 1451131 2>&1 && echo 'cancelled 1451131' || echo 'scancel failed'
echo '=== my queue ==='
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %R' 2>&1
echo '=== recent jobs ==='
sacct -u "$(id -un)" --starttime today -o JobID,JobName%14,Partition,State,Elapsed,ExitCode 2>&1 | head -20
echo '=== cpu job 1454173 stdout tail ==='
tail -30 "$RD/logs/v2cfix-1454173.out" 2>&1
echo '=== cpu job report: legacy vs fixed ==='
for M in legacy disjoint; do
  echo "--- v2c_report_$M.txt (ALL rows) ---"
  grep -E 'ALL|VERDICT|^ratio=' "$RD/v2c_report_$M.txt" 2>&1 | head -10
done
echo '=== gpu job err tail (if started) ==='
ls -l "$RD/logs/" 2>&1 | tail -8
echo 'STATUS_DONE'
