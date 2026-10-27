import sqlite3, zlib, json, sys

DB = '/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/run/qwen25_7b/results.sqlite'
con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
cur = con.cursor()

print("=== jobs 表列 ===")
print([d[1] for d in cur.execute('PRAGMA table_info(jobs)')])
print()
print("=== 各 method × rate 的 job 数 ===")
for x in cur.execute("SELECT method, rate, COUNT(*) FROM jobs GROUP BY method, rate ORDER BY method, rate"):
    print("   ", x)
print()
print("=== 取一条 mask_quito_x @0.5 看 payload 结构 ===")
row = cur.execute("SELECT key, case_key, dataset, payload FROM jobs WHERE method='mask_quito_x' AND rate=0.5 LIMIT 1").fetchone()
if row:
    key, ck, ds, payload = row
    print("key=", key, "case_key=", ck, "dataset=", ds, "payload_type=", type(payload), "len=", len(payload) if payload else 0)
    try:
        raw = payload if isinstance(payload, bytes) else bytes(payload)
        obj = json.loads(zlib.decompress(raw))
        print("解压后 keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))
        if isinstance(obj, dict):
            for k, v in obj.items():
                s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                print("   %-14s %s" % (k, s[:300]))
    except Exception as e:
        print("解压失败:", repr(e))
        print("raw head:", bytes(payload)[:200])
else:
    print("(没有 mask_quito_x @0.5 的行)")
