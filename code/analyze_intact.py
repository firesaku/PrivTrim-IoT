"""INTACT 与我们的「等隐私」对比：两者都零泄漏，比的是同等隐私下留下多少可用信息。

数据:
  外部/匿名化: out/evint_s*_percase.jsonl  (ext_intact, ext_full, full)
  我们        : out/max32_percase.jsonl (alias_budget_link@0.50 / @0.20, full)
              + out/sweep_s*_percase.jsonl (alias_budget_link@0.20)
"""
import json, io, glob, collections
import numpy as np

RD = '/path/to/project'
rng = np.random.default_rng(20261022)
BOOT = 4000


def load(p):
    return [json.loads(x) for x in io.open(p, encoding='utf-8') if x.strip()]


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


ext = []
for p in sorted(glob.glob(RD + '/out/evint_s*_percase.jsonl')):
    ext += load(p)
keys = set(r['case_key'] for r in ext)
print('INTACT 评测案例: %d  (rows=%d)' % (len(keys), len(ext)))

by = collections.defaultdict(dict)
for r in ext:
    by[r['case_key']][r['arm']] = r

maxr = load(RD + '/out/max32_percase.jsonl')
sweep = []
for p in sorted(glob.glob(RD + '/out/sweep_s*_percase.jsonl')):
    sweep += load(p)

for r in maxr:
    if r['case_key'] in keys and r.get('arm') == 'alias_budget_link' and r.get('ratio') == 0.50:
        by[r['case_key']]['OURS@0.50(equal)'] = r
for r in sweep:
    if r['case_key'] in keys and r.get('arm') == 'alias_budget_link' and r.get('ratio') == 0.20:
        by[r['case_key']]['OURS@0.20(best)'] = r

arms = ['ext_intact', 'ext_full', 'full', 'OURS@0.50(equal)', 'OURS@0.20(best)']
print('\n%-20s %5s %8s %8s %9s %9s %12s' % ('arm', 'n', 'F1', 'EM', 'tgt_pres', 'leak', 'ctx_tokens'))
print('-' * 82)
for a in arms:
    v = [d[a] for d in by.values() if a in d]
    if not v:
        print('%-20s  (no data)' % a); continue
    n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    tp = sum(num(r, 'tgt_entity_present') for r in v) / n
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    ct = sum(num(r, 'ctx_tokens') for r in v) / n
    print('%-20s %5d %8.4f %8.4f %9.3f %9.3f %12.0f' % (a, n, f1, em, tp, lk, ct))

print('\n=== 配对检验（同一批案例）===')
refs = ['OURS@0.50(equal)', 'OURS@0.20(best)', 'ext_full']
for ref in refs:
    if ref not in arms:
        continue
    d = []
    for case, m in by.items():
        if 'ext_intact' in m and ref in m:
            d.append(num(m['ext_intact'], 'f1') - num(m[ref], 'f1'))
    if len(d) < 10:
        continue
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  ext_intact - %-18s n=%-4d dF1=%+.4f  95%%CI[%+.4f,%+.4f]  win=%d lose=%d'
          % (ref, len(d), arr.mean(), lo, hi, int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))

# 长度口径
print('\n=== 长度口径（同一批 100 例）===')
for a in arms:
    v = [num(d[a], 'ctx_tokens') for d in by.values() if a in d
         and num(d[a], 'ctx_tokens') > 0]
    if v:
        print('  %-20s 平均上下文 token = %.0f' % (a, sum(v) / len(v)))
