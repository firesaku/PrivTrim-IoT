#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
JID=${1:-10598862}
echo "===== 状态 ====="
squeue -j "$JID" -o '%.12i %.10P %.10j %.9T %.11M %.8b %.16R' 2>&1
sacct -j "$JID" -o JobID,JobName%10,Partition,State,ExitCode,Elapsed,AllocCPUS -X 2>&1
echo "===== out 尾部 ====="
tail -22 "$RD/logs/e2ea-$JID.out" 2>&1
echo "===== err 尾部 ====="
tail -12 "$RD/logs/e2ea-$JID.err" 2>&1
echo "===== 报告 ====="
[ -f "$RD/e2eout/e2e_alm_report.txt" ] && cat "$RD/e2eout/e2e_alm_report.txt" || echo "(尚未生成)"
echo "ALM_CHECK_DONE"
