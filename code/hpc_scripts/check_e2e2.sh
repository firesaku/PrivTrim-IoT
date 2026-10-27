#!/bin/bash
RD=/path/to/100065732/privtrim_v2c_20261015
JID=$(cat "$RD/.e2e2_jid" 2>/dev/null | tr -d ' \r')
[ -z "$JID" ] && JID=${1:-1451107}
echo "JID=$JID"
echo "===== 状态 ====="
squeue -j "$JID" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
sacct -j "$JID" -o JobID,JobName%10,State,ExitCode,Elapsed,AllocCPUS -X 2>&1
echo "===== GPU 空闲 ====="
sinfo -p gpu -o "%N %t %C" 2>&1
echo "===== out 尾部 ====="
tail -30 "$RD/logs/e2e2-$JID.out" 2>&1
echo "===== 报告 ====="
[ -f "$RD/e2eout/e2e_report.txt" ] && cat "$RD/e2eout/e2e_report.txt" || echo "(尚未生成)"
echo "CHECK2B_DONE"
