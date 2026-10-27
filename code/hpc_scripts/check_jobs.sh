RD=/path/to/100065732/privtrim_v2c_20261015
echo '=== queue ==='
squeue -u "$(id -un)" -o '%.12i %.9P %.12j %.9T %.11M %.8b %R' 2>&1
echo '=== sacct ==='
sacct -j 10601687,10601688,10601689 -o JobID,JobName%12,State,Elapsed,ExitCode 2>&1 | grep -vE '\.(ba|ex)' | head -8
echo '=== v2d alm legacy report (if any) ==='
tail -18 "$RD/v2d_alm_report_legacy.txt" 2>&1
echo '=== v2d alm fixed report (if any) ==='
tail -18 "$RD/v2d_alm_report_disjoint.txt" 2>&1
echo '=== e2e5 report (if any) ==='
tail -25 "$RD/e2eout/e2e5_report.txt" 2>&1
echo '=== e2e4 report (if any) ==='
tail -22 "$RD/e2eout/e2e4_report.txt" 2>&1
echo '=== err tails ==='
for f in "$RD"/logs/v2dalm-*.err "$RD"/logs/e2e5-*.err "$RD"/logs/e2e4-*.err; do
  [ -s "$f" ] && echo "--- $f ---" && tail -8 "$f"
done
echo 'ALMCHECK_DONE'
