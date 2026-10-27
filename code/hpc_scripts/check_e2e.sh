#!/bin/bash
RD=/path/to/100065732/privtrim_v2c_20261015
JID=${1:-1451077}
echo "===== 状态 ====="
squeue -j "$JID" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
sacct -j "$JID" -o JobID,JobName%10,State,ExitCode,Elapsed,AllocCPUS,MaxRSS -X 2>&1
echo "===== out 尾部 ====="
tail -25 "$RD/logs/e2e-$JID.out" 2>&1
echo "===== err 尾部 ====="
tail -15 "$RD/logs/e2e-$JID.err" 2>&1
echo "===== 报告 ====="
[ -f "$RD/e2eout/e2e_report.txt" ] && cat "$RD/e2eout/e2e_report.txt" || echo "(尚未生成)"
echo "CHECK_E2E_DONE"
