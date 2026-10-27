import io, json, os
p = r"/path/to/100065732/privtrim_v2c_20261015/e2eout/e2e_alm_cells.json"
d = json.loads(io.open(p, encoding='utf-8').read())
print("cases:", d.get('cases'), "arms:", d.get('arms'))
print()
print("=== agg ===")
for k, v in sorted(d.get('agg', {}).items()):
    print("  %-24s n=%d em=%.4f f1=%.4f" % (k, v[0], v[1]/max(v[0],1), v[2]/max(v[0],1)))
print()
print("=== answers_sample (前 20 条生成) ===")
for i, a in enumerate(d.get('answers_sample', [])[:20]):
    print("  [%02d] %r" % (i, (a or '')[:220]))
