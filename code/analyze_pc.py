"""Perception Compressor（NAACL 2025）与其余方法的公开基准对比。

数据来源两份，按 case_key 合并：
  out/pcev2_s*_percase.jsonl   -> full, perception, perception_native, ext_full
  out/pubeval_s*_percase.jsonl -> 其余全部臂（300 例）
"""
import json, io, glob, collections
import numpy as np

RD = '/dpc/kuex0095/100069961_ku/privtrim'
rng = np.random.default_rng(20261029)
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


pc = load('out/pcev2_s*_percase.jsonl') + load('out/pcfill_percase.jsonl')
pub = load('out/pubeval_s*_percase.jsonl')
print('perception 评测: %d 行 / %d 例' % (len(pc), len(set(r['case_key'] for r in pc))))
print('既有评测      : %d 行 / %d 例' % (len(pub), len(set(r['case_key'] for r in pub))))

# 合并：perception 的结果优先（同一臂重复时去重）
rows, seen = [], set()
for r in pc + pub:
    k = (r['case_key'], r['arm'], r.get('ratio'))
    if k in seen:
        continue
    seen.add(k)
    rows.append(r)
print('合并后: %d 行 / %d 例' % (len(rows), len(set(r['case_key'] for r in rows))))

pckeys = set(r['case_key'] for r in pc if r['arm'] == 'ext_perception')
print('perception 覆盖: %d 例' % len(pckeys))

# 只在 perception 覆盖到的案例上比较，保证同集合
sub = [r for r in rows if r['case_key'] in pckeys]
g = collections.defaultdict(list)
for r in sub:
    g[r['arm']].append(r)

print('\n%-28s %5s %8s %8s %9s %9s' % ('arm', 'n', 'F1', 'EM', 'ctx_tok', 'leak'))
print('-' * 72)
order = sorted(g, key=lambda a: -sum(num(r, 'f1') for r in g[a]) / len(g[a]))
for a in order:
    v = g[a]
    if len(v) < 50:
        continue
    n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    ct = [num(r, 'ctx_tokens') for r in v if num(r, 'ctx_tokens') > 0]
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    print('%-28s %5d %8.4f %8.4f %9.0f %9.3f'
          % (a, n, f1, em, (sum(ct) / len(ct) if ct else 0), lk))

# 配对检验
byk = collections.defaultdict(dict)
for r in sub:
    byk[r['case_key']][r['arm']] = r


def paired(x, y, label):
    d = [num(m[x], 'f1') - num(m[y], 'f1') for m in byk.values() if x in m and y in m]
    if len(d) < 20:
        return
    arr = np.array(d)
    idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
    bs = arr[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('  %-36s n=%-3d dF1=%+.4f CI[%+.4f,%+.4f] win=%d lose=%d'
          % (label, len(d), arr.mean(), lo, hi,
             int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))


print('\n=== 关键配对检验（同 %d 例）===' % len(pckeys))
paired('alias_budget_link', 'ext_perception', '我们 - Perception Compressor')
paired('alias_budget_link', 'full', '我们 - 全上下文')
paired('ext_perception', 'full', 'Perception - 全上下文')
for a in ['ext_quito_x', 'ext_cpc', 'ext_provence', 'ext_llmlingua2']:
    paired('ext_perception', a, 'Perception - %s' % a)
