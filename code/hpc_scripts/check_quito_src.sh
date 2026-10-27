#!/bin/bash
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "===== project/research 下的压缩源码 ====="
ls -1 "$B/project/research/" 2>&1 | head -30
echo
echo "===== token_compression_multillm_full_v2 目录 ====="
ls -l "$B/project/research/token_compression_multillm_full_v2/" 2>&1 | head -30
echo
echo "===== 找 QUITO 相关实现 ====="
grep -rl -i 'quito' "$B/project/research/token_compression_multillm_full_v2/" 2>/dev/null | head -10
echo
echo "===== lib.py 里的方法定义 ====="
for f in "$B/project/research/token_compression_multillm_full_v2/lib.py" "$B/project/research/token_compression_multillm_full_v2/compress_entry_recovery_v1.py"; do
  if [ -f "$f" ]; then echo "--- $f"; grep -n -i 'def \|quito\|flan\|reconstruct\|sigma' "$f" 2>&1 | head -30; fi
done
echo
echo "===== DB 里 mask_quito_x 的 payload 长什么样 ====="
PY="$B/.venv-main/bin/python"
"$PY" - <<'EOF' 2>&1 | head -25
import sqlite3
p='/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/run/qwen25_7b/results.sqlite'
con=sqlite3.connect('file:%s?mode=ro'%p, uri=True)
cur=con.cursor()
r=cur.execute("SELECT key,case_key,dataset,method,rate,length(payload),status FROM jobs WHERE method='mask_quito_x' AND rate=0.5 LIMIT 5").fetchall()
for x in r: print(x)
print('--- jobs 表列 ---')
print([d[1] for d in cur.execute('PRAGMA table_info(jobs)')])
print('--- 各 method 的 job 数 ---')
for x in cur.execute("SELECT method, rate, COUNT(*) FROM jobs GROUP BY method, rate ORDER BY method, rate"):
    print(x)
EOF
echo "SRC_CHECK_DONE"
