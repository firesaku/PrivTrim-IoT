"""公开基准（LongBench 子集）结果分析：效用 + 泄漏，含配对检验。"""
import json, io, glob, collections
import numpy as np

RD = '/path/to/project'
rng = np.random.default_rng(20261026)
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


rows = load('out/pubeval_s*_percase.jsonl')
keys = set(r['case_key'] for r in rows)
print('公开基准: %d 行 / %d 例（LongBench 子集）' % (len(rows), len(keys)))

# 按 source 分层
src = collections.Counter()
for p in sorted(glob.glob(RD + '/data/pubeval_s*_jsonl')):
    pass
cases = {}
for p in sorted(glob.glob(RD + '/data/public_lb.jsonl')):
    for x in io.open(p, encoding='utf-8'):
        if x.strip():
            c = json.loads(x)
            cases[c['key']] = c
by_src = collections.Counter(cases[k]['source'] for k in keys if k in cases)
print('  来源分布:', dict(by_src))

agg = collections.defaultdict(list)
for r in rows:
    agg[r['arm']].append(r)

print('\n%-26s %6s %8s %8s %9s %9s' % ('arm', 'n', 'F1', 'EM', 'tgt_pres', 'leak'))
print('-' * 74)
order = sorted(agg, key=lambda a: -sum(num(r, 'f1') for r in agg[a]) / len(agg[a]))
for a in order:
    v = agg[a]
    n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    tp = sum(num(r, 'tgt_entity_present') for r in v) / n
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    print('%-26s %6d %8.4f %8.4f %9.3f %9.3f' % (a, n, f1, em, tp, lk))

print('\n=== 配对检验 vs alias_budget_link@0.5 ===')
byk = collections.defaultdict(dict)
for r in rows:
    byk[r['case_key']][r['arm']] = r
REF = 'alias_budget_link'
for a in order:
    if a in (REF, 'full'):
        continue
    d = [num(m[REF], 'f1') - num(m[a], 'f1') for m in byk.values() if a in m and REF in m]
    if len(d) < 10:
        continue
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  abl - %-24s n=%-3d dF1=%+.4f CI[%+.4f,%+.4f] win=%d lose=%d'
          % (a, len(d), arr.mean(), lo, hi,
             int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))

# 与 full 比
d = [num(m[REF], 'f1') - num(m['full'], 'f1') for m in byk.values() if REF in m and 'full' in m]
if len(d) > 10:
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('\n  abl - full                     n=%-3d dF1=%+.4f CI[%+.4f,%+.4f] win=%d lose=%d'
          % (len(d), arr.mean(), lo, hi, int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))
