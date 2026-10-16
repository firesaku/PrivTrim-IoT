#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "===== QUEUE ====="
squeue -u "$(id -un)" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo
echo "===== V3 记账（各任务）====="
sacct -j 1444036 -o JobID,JobName%10,State,ExitCode,Elapsed,AllocCPUS -X 2>&1
echo
echo "===== V3 报告 ====="
for f in "$RD"/v3out/v3_scatter_report.txt "$RD"/v3out/v3_guard_report.txt "$RD"/v3out/v3_gran_report.txt "$RD"/v3out/v3_bound_report.txt; do
  if [ -f "$f" ]; then echo "########## $(basename $f)"; cat "$f"; echo; fi
done
echo "===== V3 错误尾部（若有）====="
for e in "$RD"/logs/v3-1444036_*.err; do
  [ -s "$e" ] && { echo "--- $(basename $e)"; tail -6 "$e"; }
done
echo "COLLECT_DONE"
