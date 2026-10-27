"""为公开基准（public_lb.jsonl）构建 ext_arms 清单。

与 build_ext_arms.py 的差别：案例来源换成长度公开集，并支持只收指定方法。
输出格式一致：{"key","method","text"}，e2e 会把它变成 arm 名 ext_<method>。
"""
import json, io, os, sys

RD = '/dpc/kuex0095/100069961_ku/privtrim'
EXT = RD + '/out/ext_full'
CASES = ' '.join(sys.argv[1].split()) if len(sys.argv) > 1 else RD + '/data/public_lb.jsonl'
OUT = sys.argv[2] if len(sys.argv) > 2 else RD + '/data/ext_arms_pub.jsonl'
ONLY = [x.strip() for x in sys.argv[3].split(',')] if len(sys.argv) > 3 and sys.argv[3].strip() else None

METHODS = ['quito_x', 'provence', 'cpc', 'llmlingua2', 'mask_llmlingua2', 'dac',
           'longllmlingua', 'selective_context', 'intact', 'perception']
if ONLY:
    METHODS = [m for m in METHODS if m in ONLY]

cases = [json.loads(x) for x in io.open(CASES, encoding='utf-8') if x.strip()]
keys = set(c['key'] for c in cases)
print('cases=%d  methods=%s' % (len(cases), METHODS))

rows = []
cov = {}
for m in METHODS:
    d = os.path.join(EXT, m)
    if not os.path.isdir(d):
        print('  %-18s MISSING DIR' % m); continue
    n = 0
    for f in os.listdir(d):
        if not f.endswith('.json'):
            continue
        k = f[:-5]
        if k not in keys:
            continue
        try:
            o = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        except Exception:
            continue
        if o.get('status') != 'ok':
            continue
        mt = (o.get('matched') or {}).get('text')
        nt = (o.get('native') or {}).get('text')
        if mt:
            rows.append({'key': k, 'method': m, 'text': mt}); n += 1
        if nt:
            rows.append({'key': k, 'method': m + '_native', 'text': nt})
    cov[m] = n

# 原始上下文参照臂
for c in cases:
    rows.append({'key': c['key'], 'method': 'full', 'text': c['context']})

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('wrote %s rows=%d' % (OUT, len(rows)))
for m, n in sorted(cov.items()):
    print('  %-18s %d / %d' % (m, n, len(cases)))
