RD=/path/to/100065732/privtrim_v2c_20261015
echo '=== final states ==='
sacct -j 10601687,10601688,10601689 -o JobID,JobName%12,State,Elapsed,ExitCode 2>&1 | grep -vE '\.(ba|ex)'
echo '=== v2d job out tail ==='
tail -25 "$RD/logs/v2dalm-10601687.out" 2>&1
echo '=== v2d files present ==='
ls -l "$RD"/v2d_alm_* 2>&1
echo '=== e2e5 report ==='
tail -22 "$RD/e2eout/e2e5_report.txt" 2>&1
echo '=== e2e4 report ==='
tail -20 "$RD/e2eout/e2e4_report.txt" 2>&1
echo '=== e2eout listing ==='
ls -l "$RD/e2eout/" 2>&1 | tail -12
echo '=== err files with content ==='
for f in "$RD"/logs/v2dalm-*.err "$RD"/logs/e2e4-*.err "$RD"/logs/e2e5-*.err; do
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$sz" -gt 1500 ] && echo "--- $f ($sz bytes) ---" && tail -6 "$f"
done
echo 'DONE'
