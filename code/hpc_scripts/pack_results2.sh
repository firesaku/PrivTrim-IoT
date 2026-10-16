#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
cd "$RD" || exit 1
LIST=$(mktemp)
for f in e2e.py e2e2.py e2e3.py v1_checker.py v1b_diag.py v2c_feasibility.py v3_batch.py \
         v2d_gpu.py join_baseline.py peek_payload.py diag_e2e.py \
         v2c_report.txt v2c_cells.json v1_report.txt v1fix_report.txt v1b_report.txt \
         v2d_report.txt frozen_samples.jsonl frozen_samples_scattered.jsonl ; do
  [ -e "$f" ] && echo "$f" >> "$LIST"
done
for d in e2eout v3out logs ; do
  [ -d "$d" ] && echo "$d" >> "$LIST"
done
echo "--- 将打包 ---"
cat "$LIST"
tar czf /dpc/kuex0095/100065732/privtrim_results_20261016.tgz -T "$LIST" 2>&1 | head -5
rm -f "$LIST"
echo "--- 结果 ---"
ls -l /dpc/kuex0095/100065732/privtrim_results_20261016.tgz
echo "TAR2_DONE"
