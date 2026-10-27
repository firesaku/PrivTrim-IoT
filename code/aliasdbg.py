import json, io, sys, os
RD = '/path/to/project'
sys.path.insert(0, RD + '/code/longctx')
import v4_router as V

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
c = cases[0]
cant = c['canaries']
print('canary 原始顺序（前 8 个）:')
for x in cant[:8]:
    print('   %-8s %s' % (x['type'], x['value']))

for key in (None, 'demo-key-2026'):
    al = V.Aliaser('alias', order_key=key)
    al.preseed(cant)
    print('\n=== order_key=%r ===' % key)
    print('   order_key 实际值:', al.order_key)
    for x in cant[:8]:
        print('   %-34s -> %s' % (x['value'], al.token(x)))
    # 看该类型下的字母顺序 vs 值顺序
    by = {}
    for x in cant:
        by.setdefault(x['type'], []).append((x['value'], al.token(x)))
    t, pairs = list(by.items())[0]
    print('   类型 %s 的值升序: %s' % (t, [v for v, _ in sorted(pairs)][:6]))
    print('   类型 %s 的字母序: %s' % (t, [k for _, k in pairs][:6]))
