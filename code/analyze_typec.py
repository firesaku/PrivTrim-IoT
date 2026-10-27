"""Type C 分析：link 机制在公开上下文（锚点/答案分属不同单元）上是否生效。"""
import json, io, glob, collections
import numpy as np

RD = '/dpc/kuex0095/100069961_ku/privtrim'
rng = np.random.default_rng(20261028)
BOOT = 4000


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


rows = []
for p in sorted(glob.glob(RD + '/out/typec_s*_percase.jsonl')):
    for x in io.open(p, encoding='utf-8'):
        if x.strip():
            rows.append(json.loads(x))

print('Type C: %d 行 / %d 例' % (len(rows), len(set(r['case_key'] for r in rows))))

g = collections.defaultdict(list)
for r in rows:
    g[r['arm']].append(r)

print('\n%-22s %6s %8s %8s %9s %10s %9s' % ('arm', 'n', 'F1', 'EM', 'tgt_pres', 'ctx_tokens', 'leak'))
print('-' * 80)
for a in ['full', 'alias', 'alias_budget', 'alias_budget_link']:
    v = g.get(a) or []
    if not v:
        continue
    n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    tp = sum(num(r, 'tgt_entity_present') for r in v) / n
    ct = sum(num(r, 'ctx_tokens') for r in v if num(r, 'ctx_tokens') > 0) / max(1, sum(1 for r in v if num(r, 'ctx_tokens') > 0))
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    print('%-22s %6d %8.4f %8.4f %9.3f %10.0f %9.3f' % (a, n, f1, em, tp, ct, lk))

byk = collections.defaultdict(dict)
for r in rows:
    byk[r['case_key']][r['arm']] = r


def paired(x, y, label):
    d, same_tok = [], 0
    for m in byk.values():
        if x in m and y in m:
            d.append(num(m[x], 'f1') - num(m[y], 'f1'))
            if m[x].get('ctx_tokens') == m[y].get('ctx_tokens'):
                same_tok += 1
    if len(d) < 10:
        return
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  %-40s n=%-3d dF1=%+.4f CI[%+.4f,%+.4f] win=%d lose=%d  ctx_tokens相同=%d'
          % (label, len(d), arr.mean(), lo, hi,
             int((arr > 1e-12).sum()), int((arr < -1e-12).sum()), same_tok))


print('\n=== 关键：link 增益 ===')
paired('alias_budget_link', 'alias_budget', 'link - nolink (alias_budget_link - alias_budget)')
print('\n=== 其他对照 ===')
paired('alias_budget_link', 'full', 'abl - full')
paired('alias', 'full', 'alias - full')
paired('alias_budget', 'alias', 'ab - alias')
