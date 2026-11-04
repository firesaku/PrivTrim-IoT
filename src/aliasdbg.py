"""Print the alias map under an unkeyed and a keyed assignment.

Shows, for the canaries of one case, how `v4_router.Aliaser` hands out alias
tokens: without `PT_ALIAS_KEY` the letters follow ascending value order (so the
rank of an alias leaks the rank of its value); with a key the order is a keyed
permutation.

Usage:
    gunzip -k data/longctx_max.jsonl.gz
    python src/aliasdbg.py
"""
import json, io, sys, os

RD = os.environ.get('PRIVTRIM_ROOT',
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RD, 'src', 'longctx'))
import v4_router as V

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
c = cases[0]
cant = c['canaries']
print('canaries of %s, in file order (first 8):' % c['key'])
for x in cant[:8]:
    print('   %-8s %s' % (x['type'], x['value']))

for key in (None, 'demo-key-2026'):
    al = V.Aliaser('alias', order_key=key)
    al.preseed(cant)
    print('\n=== order_key=%r ===' % key)
    print('   effective order_key:', al.order_key)
    for x in cant[:8]:
        print('   %-34s -> %s' % (x['value'], al.token(x)))
    # compare the letter order against ascending value order within one type
    by = {}
    for x in cant:
        by.setdefault(x['type'], []).append((x['value'], al.token(x)))
    t, pairs = list(by.items())[0]
    print('   type %s, values ascending: %s' % (t, [v for v, _ in sorted(pairs)][:6]))
    print('   type %s, alias letters   : %s' % (t, [k for _, k in pairs][:6]))
