#!/bin/bash
RD=/path/to/100065732/privtrim_v2c_20261015
JID=$(cat "$RD/.e2e3_jid" 2>/dev/null | tr -d ' \r')
[ -z "$JID" ] && JID=${1:-10600642}
echo "JID=$JID"
squeue -j "$JID" -o '%.12i %.10P %.10j %.9T %.11M %.8b %.16R' 2>&1
sacct -j "$JID" -o JobID,JobName%10,State,ExitCode,Elapsed -X 2>&1
echo "===== out 尾部 ====="
tail -16 "$RD/logs/e2e3-$JID.out" 2>&1
echo "===== 报告 ====="
[ -f "$RD/e2eout/e2e3_report.txt" ] && cat "$RD/e2eout/e2e3_report.txt" || echo "(尚未生成)"
echo "E2E3_CHECK_DONE"
