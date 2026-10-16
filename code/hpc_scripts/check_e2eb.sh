#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
JID=$(cat "$RD/.e2eb_jid" 2>/dev/null | tr -d ' \r')
[ -z "$JID" ] && JID=${1:-10599046}
echo "JID=$JID"
squeue -j "$JID" -o '%.12i %.10P %.10j %.9T %.11M %.8b %.16R' 2>&1
sacct -j "$JID" -o JobID,JobName%10,State,ExitCode,Elapsed -X 2>&1
echo "===== out 尾部 ====="
tail -18 "$RD/logs/e2eb-$JID.out" 2>&1
echo "===== 报告 ====="
[ -f "$RD/e2eout/e2e_b_report.txt" ] && cat "$RD/e2eout/e2e_b_report.txt" || echo "(尚未生成)"
echo "E2EB_CHECK_DONE"
