import json, io, glob, collections

RD = '/dpc/kuex0095/100069961_ku/privtrim'


def load(pat):
    out = []
    for p in sorted(glob.glob(RD + '/' + pat)):
        for x in io.open(p, encoding='utf-8'):
            if x.strip():
                out.append(json.loads(x))
    return out


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


def curve(rows, arm, label):
    d = collections.defaultdict(list)
    for r in rows:
        if r.get('arm') == arm:
            d[r.get('ratio')].append(num(r, 'f1'))
    print('\n%s' % label)
    out = []
    for ratio in sorted(d, key=lambda x: (x is None, x)):
        v = d[ratio]
        f1 = sum(v) / len(v)
        out.append((ratio, f1, len(v)))
        print('  budget=%-5s F1=%.4f  n=%d' % (ratio, f1, len(v)))
    if out:
        best = max(out, key=lambda x: x[1])
        print('  -> 峰值 budget=%s  F1=%.4f' % (best[0], best[1]))
    return {r: f for r, f, _ in out}


# 有 key 的预算点来自两个作业：keyfull(0.25/0.5/0.75) + keyset(0.2/0.5)
kf = load('out/keyfull_s*_percase.jsonl')
ks = load('out/keyset_s*_percase.jsonl')
c1 = curve(kf, 'alias_budget_link', '有 key（keyfull）: alias_budget_link 预算曲线')
c2 = {r: f for r, f in
      [(0.2, (lambda v: sum(v) / len(v))([num(r, 'f1') for r in ks
                                          if r.get('arm') == 'alias_budget_link' and r.get('ratio') == 0.2]))
       ] if True}
merged = dict(c1)
for r, f in c2.items():
    if r not in merged:
        merged[r] = f

print('\n=== 合并后的有 key 预算曲线 ===')
for r in sorted(merged):
    print('  %.2f  F1=%.4f' % (r, merged[r]))
if merged:
    b = max(merged, key=lambda x: merged[x])
    print('  -> 最优点 budget=%.2f  F1=%.4f' % (b, merged[b]))

# 无 key 的对照（max32 + sweep）
mx = load('out/max32_percase.jsonl')
sw = load('out/sweep_s*_percase.jsonl')
mk = {}
for src in (mx, sw):
    for r in src:
        if r.get('arm') == 'alias_budget_link' and r.get('ratio') is not None:
            mk.setdefault(r['ratio'], []).append(num(r, 'f1'))
print('\n=== 无 key（max32+sweep，含 0.05–0.20）===')
for r in sorted(mk):
    v = mk[r]
    print('  %.2f  F1=%.4f  n=%d' % (r, sum(v) / len(v), len(v)))
if mk:
    b = max(mk, key=lambda x: sum(mk[x]) / len(mk[x]))
    print('  -> 最优点 budget=%.2f  F1=%.4f' % (b, sum(mk[b]) / len(mk[b])))
