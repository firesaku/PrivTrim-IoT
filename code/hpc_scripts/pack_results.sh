#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
cd "$RD" || exit 1
echo "===== 目录内容 ====="
ls -l
echo "===== e2eout ====="
ls -l e2eout
echo "===== v3out ====="
ls -l v3out 2>/dev/null
echo
echo "===== 打包 ====="
tar czf /dpc/kuex0095/100065732/privtrim_results_20261016.tgz \
  --exclude='*.sqlite' \
  e2e.py e2e2.py e2e3.py v1_checker.py v1b_diag.py v2c_feasibility.py v3_batch.py \
  v2d_gpu.py join_baseline.py peek_payload.py \
  v2c_report.txt v2c_cells.json v1_report.txt v1fix_report.txt v1b_report.txt v2d_report.txt \
  frozen_samples.jsonl frozen_samples_scattered.jsonl \
  e2eout v3out 2>&1
ls -l /dpc/kuex0095/100065732/privtrim_results_20261016.tgz
echo "TAR_DONE"
