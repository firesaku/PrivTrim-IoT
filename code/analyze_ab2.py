"""A（原问题） vs B（引用 PII 的合成问题）对照。

A: out/pubeval_s*_percase.jsonl
B: out/typeb_s*_percase.jsonl
两者上下文完全相同、已压缩产物相同，唯一差别是问题。
"""
import json, io, glob, collections
import numpy as np

RD = '/path/to/project'
rng = np.random.default_rng(20261027)
BOOT = 4000


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


A = load('out/pubeval_s*_percase.jsonl')
B = load('out/typeb_s*_percase.jsonl')


def agg(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r['arm']].append(r)
    return d


gA, gB = agg(A), agg(B)
arms = sorted(set(gA) & set(gB), key=lambda a: -sum(num(r, 'f1') for r in gB[a]) / len(gB[a]))

print('A 行数=%d  B 行数=%d  共同臂=%d' % (len(A), len(B), len(arms)))
print('\n%-28s %10s %10s %10s' % ('arm', 'F1(A 原问)', 'F1(B 引用PII)', 'Δ(B-A)'))
print('-' * 62)
for a in arms:
    fa = sum(num(r, 'f1') for r in gA[a]) / len(gA[a])
    fb = sum(num(r, 'f1') for r in gB[a]) / len(gB[a])
    print('%-28s %10.4f %10.4f %+10.4f' % (a, fa, fb, fb - fa))

# 关键：link 增益在 A 与 B 下分别是多少
def paired(rows, x, y):
    byk = collections.defaultdict(dict)
    for r in rows:
        byk[r['case_key']][r['arm']] = r
    d = [num(m[x], 'f1') - num(m[y], 'f1') for m in byk.values() if x in m and y in m]
    if len(d) < 10:
        return None
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return arr.mean(), lo, hi, len(d), int((arr > 1e-12).sum()), int((arr < -1e-12).sum())


print('\n=== link 增益（alias_budget_link − alias_budget）===')
for label, rows in (('A 原问题', A), ('B 引用 PII', B)):
    r = paired(rows, 'alias_budget_link', 'alias_budget')
    if r:
        print('  %-12s dF1=%+.4f CI[%+.4f,%+.4f] n=%d win=%d lose=%d'
              % (label, r[0], r[1], r[2], r[3], r[4], r[5]))

print('\n=== abl vs full ===')
for label, rows in (('A 原问题', A), ('B 引用 PII', B)):
    r = paired(rows, 'alias_budget_link', 'full')
    if r:
        print('  %-12s dF1=%+.4f CI[%+.4f,%+.4f] n=%d win=%d lose=%d'
              % (label, r[0], r[1], r[2], r[3], r[4], r[5]))

print('\n=== 泄漏（B 条件） ===')
for a in ['alias_budget_link', 'alias_budget', 'alias', 'full', 'ext_quito_x', 'ext_cpc']:
    if a in gB:
        v = gB[a]
        lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / len(v)
        print('  %-22s 泄漏率 %.3f  (n=%d)' % (a, lk, len(v)))
