"""Paired significance tests for the max32 full-dataset run.

不需要 GPU：只读 out/max32_percase.jsonl（2828 例 x 21 臂，逐例 F1/EM）。
仅依赖标准库 + numpy（HPC venv 自带）。

用法: cd /dpc/kuex0095/100069961_ku/privtrim && env/bin/python sigtest.py
"""
import json, io, math, collections, sys

RD = '/dpc/kuex0095/100069961_ku/privtrim'
SRC = RD + '/out/max32_percase.jsonl'
BOOT = 10000
random_seed = 20261020

try:
    import numpy as np
except ImportError:
    print('need numpy'); sys.exit(1)
rng = np.random.default_rng(random_seed)

rows = [json.loads(x) for x in io.open(SRC, encoding='utf-8') if x.strip()]
print('records: %d' % len(rows))

# case_key -> {(arm, ratio): row}
by = collections.defaultdict(dict)
kinds = {}
sources = {}
for r in rows:
    by[r['case_key']][(r['arm'], r['ratio'])] = r
    kinds[r['case_key']] = r.get('case_kind')
    sources[r['case_key']] = r.get('source')

print('cases  : %d' % len(by))


def f1(r):
    v = r.get('f1')
    return float(v) if isinstance(v, (int, float)) else 0.0


def emok(r):
    v = r.get('em')
    return 1 if isinstance(v, (int, float)) and v >= 0.5 else 0


def mcnemar_exact(b, c):
    """two-sided exact binomial p for McNemar, H0: p=0.5"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 2 * P(X <= k) for X~Bin(n, .5), 全整数运算避免 2.0**n 溢出
    num = sum(math.comb(n, i) for i in range(0, k + 1))
    den = 1 << n                      # exact 2**n
    return min(1.0, 2.0 * (num / den))


def paired(A, B, label, kind=None, src=None):
    keys = [k for k, v in by.items() if A in v and B in v
            and (kind is None or kinds[k] == kind)
            and (src is None or sources[k] == src)]
    if not keys:
        return
    n = len(keys)
    da = np.array([f1(by[k][A]) for k in keys])
    db = np.array([f1(by[k][B]) for k in keys])
    d = da - db
    mean = d.mean()
    # bootstrap 95% CI of the mean paired difference
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # sign counts
    win = int((d > 1e-12).sum()); loss = int((d < -1e-12).sum()); tie = n - win - loss
    # McNemar on exact-match
    b_ = sum(1 for k in keys if emok(by[k][A]) and not emok(by[k][B]))
    c_ = sum(1 for k in keys if (not emok(by[k][A])) and emok(by[k][B]))
    p = mcnemar_exact(b_, c_)
    tag = 'ALL' if kind is None and src is None else '%s/%s' % (kind or '*', src or '*')
    print('%-9s %-42s n=%-5d dF1=%+.4f [%+.4f,%+.4f] CI%s0  W/T/L=%d/%d/%d  McNemar b=%d c=%d p=%.3g'
          % (tag, label, n, mean, lo, hi, '+' if lo > 0 else ('-' if hi < 0 else '?'),
             win, tie, loss, b_, c_, p))


def show(A, B, name):
    print('\n--- %s ---' % name)
    for kind in [None, 'multi_entity', 'needle']:
        paired(A, B, name, kind=kind)


print('\n=== 1. 头条主张: alias_budget_link@0.25 vs full ===')
show(('alias_budget_link', 0.25), ('full', None), 'abl@.25 vs full')

print('\n=== 2. 链路 vs 非链路(同预算) ===')
show(('alias_budget_link', 0.25), ('alias_budget', 0.25), 'abl_link vs abl @.25')

print('\n=== 3. 对照: 纯别名 vs full ===')
show(('alias', None), ('full', None), 'alias vs full')

print('\n=== 4. 对照: 删除式 vs full ===')
show(('delete', None), ('full', None), 'delete vs full')

print('\n=== 5. 竞争: alias_budget_link@0.25 vs budget_cluster_link@0.25 ===')
show(('alias_budget_link', 0.25), ('budget_cluster_link', 0.25), 'abl vs bcl @.25')

print('\n=== 6. 预算单调性: F1 vs budget (alias_budget_link) ===')
for kind in [None, 'multi_entity', 'needle']:
    keys = [k for k, v in by.items() if kind is None or kinds[k] == kind]
    out = []
    for ratio in [None, 0.25, 0.5, 0.75]:
        A = ('alias_budget_link', ratio)
        v = [f1(by[k][A]) for k in keys if A in by[k]]
        out.append('%s:%.4f(n=%d)' % (ratio, sum(v) / len(v) if v else -1, len(v)))
    print('  %-13s %s' % (kind or 'ALL', '  '.join(out)))
