"""keyfull（启用 PT_ALIAS_KEY 的全 21 臂）vs max32（旧，无 key）逐臂比对。"""
import json, io, glob, collections
import numpy as np

RD = '/dpc/kuex0095/100069961_ku/privtrim'
rng = np.random.default_rng(20261025)
BOOT = 4000


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


def load(pat):
    out = []
    for p in sorted(glob.glob(RD + '/' + pat)):
        for x in io.open(p, encoding='utf-8'):
            if x.strip():
                out.append(json.loads(x))
    return out


kf = load('out/keyfull_s*_percase.jsonl')
mx_all = load('out/max32_percase.jsonl')
# 关键：max32 必须裁到 keyfull 实际覆盖的同一批案例上，否则 Δ 列是在比两个不同的集合
_kf_keys = set(r['case_key'] for r in kf)
mx = [r for r in mx_all if r['case_key'] in _kf_keys]
print('keyfull: %d 行 / %d 例' % (len(kf), len(_kf_keys)))
print('max32  : %d 行 / %d 例（已裁到同一批案例）' % (len(mx), len(set(r['case_key'] for r in mx))))


def agg(rows):
    a = collections.defaultdict(list)
    for r in rows:
        a[(r['arm'], r.get('ratio'))].append(r)
    return a


A = agg(kf)
B = agg(mx)
keys = sorted(set(A) | set(B), key=lambda k: (str(k[0]), -1 if k[1] is None else float(k[1])))

print('\n%-24s %6s %10s %10s %10s %8s' % ('arm', 'ratio', 'F1(无key)', 'F1(有key)', 'Δ', 'n'))
print('-' * 74)
for k in keys:
    va, vb = B.get(k, []), A.get(k, [])
    if not va and not vb:
        continue
    m1 = sum(num(r, 'f1') for r in va) / len(va) if va else float('nan')
    m2 = sum(num(r, 'f1') for r in vb) / len(vb) if vb else float('nan')
    d = m2 - m1 if (va and vb) else float('nan')
    print('%-24s %6s %10.4f %10.4f %+10.4f %8d'
          % (k[0], k[1], m1, m2, d, len(vb)))

# 关键对比的配对检验
print('\n=== 配对检验（有 key，同 2728 例）===')
byk = collections.defaultdict(dict)
for r in kf:
    byk[r['case_key']][(r['arm'], r.get('ratio'))] = r


def paired(a, b, label):
    d = []
    for k, m in byk.items():
        ka, kb = (a[0], a[1]), (b[0], b[1])
        if ka in m and kb in m:
            d.append(num(m[ka], 'f1') - num(m[kb], 'f1'))
    if len(d) < 10:
        return
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  %-42s n=%-4d dF1=%+.4f CI[%+.4f,%+.4f] win=%d lose=%d'
          % (label, len(d), arr.mean(), lo, hi,
             int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))


paired(('alias_budget_link', 0.5), ('full', None), 'abl@0.5 - full')
paired(('alias_budget_link', 0.5), ('budget_cluster_link', 0.5), 'abl@0.5 - bcl@0.5')
paired(('alias_budget_link', 0.5), ('alias_budget', 0.5), 'abl@0.5 - ab@0.5  (link 增益)')
paired(('alias', None), ('full', None), 'alias - full')
paired(('delete', None), ('full', None), 'delete - full')

# 预算曲线
print('\n=== 有 key 的预算曲线（alias_budget_link）===')
vals = []
for ratio in [0.25, 0.5, 0.75]:
    v = A.get(('alias_budget_link', ratio), [])
    if v:
        f1 = sum(num(r, 'f1') for r in v) / len(v)
        vals.append((ratio, f1))
        print('  %.2f  F1=%.4f  n=%d' % (ratio, f1, len(v)))
if vals:
    print('  峰值预算 = %.2f' % max(vals, key=lambda x: x[1])[0])
