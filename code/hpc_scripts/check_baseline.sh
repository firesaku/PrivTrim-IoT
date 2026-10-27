#!/bin/bash
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "===== run/qwen25_7b 内容 ====="
ls -l "$B/run/qwen25_7b/" 2>&1 | head -15
echo
echo "===== per_case.csv 表头 ====="
for f in "$B/run/qwen25_7b/per_case.csv" "$B/run/qwen25_7b/tables.csv" ; do
  if [ -f "$f" ]; then
    echo "--- $f ($(stat -c%s "$f") bytes)"
    head -3 "$f"
  else
    echo "--- $f 不存在"
  fi
done
echo
echo "===== sqlite 是否有 metrics 表 ====="
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
DB="$B/run/qwen25_7b/results.sqlite"
ls -l "$DB" 2>&1
"$PY" - <<'EOF' 2>&1 | head -30
import sqlite3, os
p='/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/run/qwen25_7b/results.sqlite'
if not os.path.exists(p):
    print('DB missing'); raise SystemExit
con=sqlite3.connect('file:%s?mode=ro'%p, uri=True)
cur=con.cursor()
print('tables:', [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")])
for t in ('metrics','conditions','jobs','requests'):
    try:
        n=cur.execute('SELECT COUNT(*) FROM %s'%t).fetchone()[0]
        cols=[d[1] for d in cur.execute('PRAGMA table_info(%s)'%t)]
        print(t, 'rows=',n, 'cols=',cols[:12])
    except Exception as e:
        print(t,'ERR',e)
EOF
echo "BASE_CHECK_DONE"
