#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "===== QUEUE ====="
squeue -j 1443690 -o '%.12i %.10P %.14j %.9T %.11M %.20R' 2>&1
echo "===== SACCT ====="
sacct -j 1443690 -o JobID,JobName%14,Partition,State,ExitCode,Elapsed,AllocCPUS,MaxRSS -X 2>&1
echo "===== FILES ====="
ls -l "$RD" 2>&1
ls -l "$RD/logs" 2>&1
echo "===== REPORT ====="
if [ -f "$RD/v2c_report.txt" ]; then cat "$RD/v2c_report.txt"; else echo "(no report yet)"; fi
echo "===== OUT TAIL ====="
tail -25 "$RD"/logs/v2c-1443690.out 2>&1
echo "===== ERR TAIL ====="
tail -25 "$RD"/logs/v2c-1443690.err 2>&1
echo "CHECK_DONE"
