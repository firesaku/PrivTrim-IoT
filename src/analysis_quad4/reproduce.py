# -*- coding: utf-8 -*-
"""Recompute every reported number from the archived results and diff each
one against its expected value.

Usage (from anywhere in the repository):
    python src/analysis_quad4/reproduce.py

Prints PASS / FAIL per item.  All PASS means the data in this repository
agrees with every number stated in the documentation, i.e. the reported
results are reproducible from what is shipped.
No cluster, no model: it only reads the jsonl(.gz) files under
results/percase/.  Exit code is 1 if any item fails.
"""
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PC = REPO / 'results' / 'percase'
Q4 = PC / 'quad4'
TOL = 0.0015          # tolerance: fp16 non-determinism level

FAILS = []


def load(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def collect(patterns):
    """patterns: [(dir, [globs])] -> arm -> case_key -> row (ratio 0.5 or no budget)"""
    out = {}
    for base, pats in patterns:
        for pat in pats:
            for p in sorted(Path(base).glob(pat)):
                for r in load(p):
                    if r.get('ratio') not in (0.5, None):
                        continue
                    out.setdefault(r.get('arm', ''), {})[r['case_key']] = r
    return out


def f1(r):
    v = r.get('f1')
    return float(v) if isinstance(v, (int, float)) else 0.0


def leaked(r):
    return (r.get('ctx_leak_value') or 0) > 0 or (r.get('out_leak_value') or 0) > 0


def stats(arms, arm):
    d = arms.get(arm) or {}
    if not d:
        return None
    n = len(d)
    return dict(n=n,
                f1=sum(f1(r) for r in d.values()) / n,
                leak=sum(1 for r in d.values() if leaked(r)) / n)


def check(label, got, want, tol=TOL):
    if got is None:
        print('  FAIL  %-46s no data' % label)
        FAILS.append(label)
        return
    ok = abs(got - want) <= tol
    print('  %s  %-46s got %.4f  expected %.4f' % ('PASS' if ok else 'FAIL', label, got, want))
    if not ok:
        FAILS.append(label)


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------- 1) 2728
section('1) longctx_max extension, 2,728 cases (forward fill, no answer key)')
m = collect([(Q4, ['quad4_2728_fwd_*.jsonl.gz'])])
check('full',               (stats(m, 'full') or {}).get('f1'), 0.6081)
check('quad4',              (stats(m, 'quad4') or {}).get('f1'), 0.4937)
check('alias_budget_link',  (stats(m, 'alias_budget_link') or {}).get('f1'), 0.3482)
check('budget_cluster_link', (stats(m, 'budget_cluster_link') or {}).get('f1'), 0.3352)
check('quad4_del',          (stats(m, 'quad4_del') or {}).get('f1'), 0.3142)
check('alias_budget',       (stats(m, 'alias_budget') or {}).get('f1'), 0.1469)
check('budget_cluster',     (stats(m, 'budget_cluster') or {}).get('f1'), 0.0981)
q = stats(m, 'quad4') or {}
print('  ---- quad4 leakage must be 0; measured %.3f' % (q.get('leak', -1),))
if abs(q.get('leak', -1)) > 1e-9:
    FAILS.append('quad4 leak')

section('1b) external baselines (same extevfull batch)')
e = collect([(PC, ['extevfull_s*_percase.jsonl.gz',
               'extevfull_s*_percase.gz'])])
check('ext_full (identity baseline; should match our full)', (stats(e, 'ext_full') or {}).get('f1'), 0.6078)
check('quito_x',     (stats(e, 'ext_quito_x') or {}).get('f1'), 0.4831)
check('cpc',         (stats(e, 'ext_cpc') or {}).get('f1'), 0.4740)
check('provence',    (stats(e, 'ext_provence') or {}).get('f1'), 0.4080)
check('llmlingua2',  (stats(e, 'ext_llmlingua2') or {}).get('f1'), 0.2651)
check('mask_llmlingua2 (leak-free baseline)', (stats(e, 'ext_mask_llmlingua2') or {}).get('f1'), 0.1712)
ml = stats(e, 'ext_mask_llmlingua2') or {}
if abs(ml.get('leak', -1)) > 1e-9:
    FAILS.append('mask_llmlingua2 leak')

# ---------------------------------------------------------------- 2) 112
section('2) main set, 112 cases (paper main table, forward fill)')
m112 = collect([(Q4, ['quad4_112_fwd_*.jsonl.gz'])])
ref = collect([(PC, ['longctx_aliasq_percase.jsonl',
               'longctx_aliasq_percase.gz',
               'aliasq_qwen_percase.gz'])])
check('full',              (stats(m112, 'full') or {}).get('f1'), 0.7189)
check('alias',             (stats(m112, 'alias') or {}).get('f1'), 0.6475)
check('quad4',             (stats(m112, 'quad4') or {}).get('f1'), 0.6366)
check('delete',            (stats(m112, 'delete') or {}).get('f1'), 0.4916)
check('budget_cluster_link', (stats(m112, 'budget_cluster_link') or {}).get('f1'), 0.4308)
check('alias_budget_link', (stats(m112, 'alias_budget_link') or {}).get('f1'), 0.4219)
check('alias_budget',      (stats(m112, 'alias_budget') or {}).get('f1'), 0.2232)
check('budget_cluster',    (stats(m112, 'budget_cluster') or {}).get('f1'), 0.2143)

# identity baseline: our full arm must match the paper's reference batch case by case
a = (m112.get('full') or {})
b = (ref.get('full') or {})
if not b:
    print('  ---- SKIP: no reference batch in results/percase/ yet')
    FAILS.append('112 identity baseline (reference batch missing)')
else:
    same = sum(1 for k in b if k in a and f1(a[k]) == f1(b[k]))
    print('  ---- full arm identical case-by-case with the paper reference batch: %d / %d'
          % (same, len(b)))
    if same < len(b):
        FAILS.append('112 identity baseline')

# token efficiency of quad4
tk = [r.get('ctx_tokens') or 0 for r in (m112.get('quad4') or {}).values()]
tkf = [r.get('ctx_tokens') or 0 for r in a.values()]
if tk and tkf:
    ratio = (sum(tk) / len(tk)) / (sum(tkf) / len(tkf))
    print('  ---- quad4 mean tokens as a share of the full context: %.1f%% (expected ~48.6%%)'
      % (100 * ratio))
    if abs(ratio - 0.486) > 0.01:
        FAILS.append('112 token ratio')

# ---------------------------------------------------------------- 3) public_lb
section('3) public_lb, three variants')
for tag, pattern, expect in (
    ('Type A', 'quad4_publb_fwd_typea_*.jsonl.gz',
     {'full': 0.4181, 'alias': 0.4074, 'quad4': 0.3516, 'alias_budget_link': 0.3112}),
    ('Type B', 'quad4_publb_fwd_typeb_*.jsonl.gz',
     {'full': 0.9889, 'quad4': 0.9607, 'alias': 0.9197,
      'alias_budget_link': 0.3100, 'quad4_del': 0.1674}),
    ('Type C', 'quad4_publb_fwd_typec_*.jsonl.gz',
     {'alias': 0.7078, 'quad4': 0.6733, 'full': 0.0967,
      'alias_budget_link': 0.0167, 'quad4_del': 0.0000}),
):
    print('  --- %s' % tag)
    d = collect([(Q4, [pattern])])
    for arm, want in expect.items():
        check('    %s' % arm, (stats(d, arm) or {}).get('f1'), want)

# ---------------------------------------------------------------- 4) 方向对比
section('4) the two refill directions (reverse = reversed(drop_order))')
for tag, pat, arm, want in (
    ('2728 reverse quad4', 'quad4_2728_rev_*.jsonl.gz', 'quad4', 0.4832),
    ('112 reverse quad4', 'quad4_112_rev_*.jsonl.gz', 'quad4', 0.5844),
    ('TypeA reverse quad4', 'quad4_publb_rev_typea_*.jsonl.gz', 'quad4', 0.3172),
    ('TypeB reverse quad4', 'quad4_publb_rev_typeb_*.jsonl.gz', 'quad4', 0.9719),
    ('TypeC reverse quad4', 'quad4_publb_rev_typec_*.jsonl.gz', 'quad4', 0.0167),
):
    d = collect([(Q4, [pat])])
    check(tag, (stats(d, arm) or {}).get('f1'), want)


# ---------------------------------------------------------------- 5) 预算曲线
section('5) budget sweep (112 cases, 5 ratios, no answer key)')


def collect_ratio(pattern):
    """Aggregate by (arm, ratio) for the multi-ratio runs."""
    out = {}
    for q in sorted(Q4.glob(pattern)):
        for r in load(q):
            m = out.setdefault((r.get('arm', ''), r.get('ratio')), {})
            m[r['case_key']] = r
    return out


sw = collect_ratio('quad4_sweep112_*.jsonl.gz')
RATIOS = [0.10, 0.15, 0.25, 0.50, 0.75]
EXPECT_SW = {
    'quad4':               [0.4357, 0.4681, 0.5116, 0.6366, 0.6158],
    'quad4_del':           [0.3297, 0.3788, 0.3955, 0.5205, 0.5086],
    'alias_budget_link':   [0.2304, 0.2052, 0.2488, 0.4219, 0.5512],
    'budget_cluster_link': [0.2304, 0.2052, 0.2310, 0.4308, 0.5576],
}
for arm, wants in EXPECT_SW.items():
    for ratio, want in zip(RATIOS, wants):
        d = sw.get((arm, ratio)) or {}
        got = (sum(f1(r) for r in d.values()) / len(d)) if d else None
        check('%s @%.2f' % (arm, ratio), got, want)

# ---------------------------------------------------------------- 6) 跨模型
section('6) cross-model (Mistral-7B, 112 cases @0.50)')
mis = collect([(Q4, ['quad4_mistral112_*.jsonl.gz'])])
for arm, want in (('full', 0.6931), ('alias', 0.5931), ('delete', 0.4559),
                  ('quad4', 0.6946), ('quad4_del', 0.5520),
                  ('budget_cluster_link', 0.3837), ('alias_budget_link', 0.3748)):
    check('Mistral %s' % arm, (stats(mis, arm) or {}).get('f1'), want)

# ---------------------------------------------------------------- 7) 改述
section('7) paraphrase robustness (112 cases @0.50)')
for tag, pat, expect in (
    ('paraphrase A', 'quad4_para112_*_paraA_*.jsonl.gz',
     {'full': 0.6831, 'alias': 0.6297, 'delete': 0.4731, 'quad4': 0.6637,
      'budget_cluster_link': 0.3926, 'alias_budget_link': 0.3926}),
    ('paraphrase B', 'quad4_para112_*_paraB_*.jsonl.gz',
     {'full': 0.6575, 'alias': 0.6141, 'delete': 0.4705, 'quad4': 0.6023,
      'budget_cluster_link': 0.4182, 'alias_budget_link': 0.4029}),
):
    print('  --- %s' % tag)
    d = collect([(Q4, [pat])])
    for arm, want in expect.items():
        check('    %s' % arm, (stats(d, arm) or {}).get('f1'), want)

# ---------------------------------------------------------------- summary
print()
print('=' * 78)
if FAILS:
    print('RESULT: %d FAIL' % len(FAILS))
    for f in FAILS:
        print('  - %s' % f)
    sys.exit(1)
print('RESULT: all PASS - the archived results in this repository agree with '
      'every number stated in the docs')
