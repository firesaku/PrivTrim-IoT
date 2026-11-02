# -*- coding: utf-8 -*-
"""从仓库内的原始结果重算全部报告数字，并逐项比对预期值。

用法（在仓库任意位置）：
    python code/analysis_quad4/reproduce.py

输出每项 PASS / FAIL。全部 PASS 表示仓库内的数据与文档中的数字一致，可复现。
不依赖集群、不加载模型，只读 results/percase/ 下的 jsonl(.gz)。
"""
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PC = REPO / 'results' / 'percase'
Q4 = PC / 'quad4'
TOL = 0.0015          # 允许的差异（fp16 非确定性级别）

FAILS = []


def load(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def collect(patterns):
    """patterns: [(目录, glob 列表)] -> arm -> case_key -> row（只取 ratio 0.5 或无预算）"""
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
        print('  FAIL  %-46s 无数据' % label)
        FAILS.append(label)
        return
    ok = abs(got - want) <= tol
    print('  %s  %-46s 实测 %.4f  预期 %.4f' % ('PASS' if ok else 'FAIL', label, got, want))
    if not ok:
        FAILS.append(label)


def section(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ---------------------------------------------------------------- 1) 2728
section('1) longctx_max 2,728 例（正向填充，全部无金标）')
m = collect([(Q4, ['quad4_2728_fwd_*.jsonl.gz'])])
check('full',               (stats(m, 'full') or {}).get('f1'), 0.6081)
check('quad4',              (stats(m, 'quad4') or {}).get('f1'), 0.4937)
check('alias_budget_link',  (stats(m, 'alias_budget_link') or {}).get('f1'), 0.3482)
check('budget_cluster_link', (stats(m, 'budget_cluster_link') or {}).get('f1'), 0.3352)
check('quad4_del',          (stats(m, 'quad4_del') or {}).get('f1'), 0.3142)
check('alias_budget',       (stats(m, 'alias_budget') or {}).get('f1'), 0.1469)
check('budget_cluster',     (stats(m, 'budget_cluster') or {}).get('f1'), 0.0981)
q = stats(m, 'quad4') or {}
print('  ---- quad4 泄漏率应为 0，实测 %.3f' % (q.get('leak', -1),))
if abs(q.get('leak', -1)) > 1e-9:
    FAILS.append('quad4 leak')

section('1b) 外部基线（同一 extevfull 批次）')
e = collect([(PC, ['extevfull_s*_percase.jsonl.gz',
               'extevfull_s*_percase.gz'])])
check('ext_full（恒等基线，应约等于我们的 full）', (stats(e, 'ext_full') or {}).get('f1'), 0.6078)
check('quito_x',     (stats(e, 'ext_quito_x') or {}).get('f1'), 0.4831)
check('cpc',         (stats(e, 'ext_cpc') or {}).get('f1'), 0.4740)
check('provence',    (stats(e, 'ext_provence') or {}).get('f1'), 0.4080)
check('llmlingua2',  (stats(e, 'ext_llmlingua2') or {}).get('f1'), 0.2651)
check('mask_llmlingua2（零泄漏基线）', (stats(e, 'ext_mask_llmlingua2') or {}).get('f1'), 0.1712)
ml = stats(e, 'ext_mask_llmlingua2') or {}
if abs(ml.get('leak', -1)) > 1e-9:
    FAILS.append('mask_llmlingua2 leak')

# ---------------------------------------------------------------- 2) 112
section('2) longctx_cases 112 例（论文主表，正向填充）')
m112 = collect([(Q4, ['quad4_112_fwd_*.jsonl.gz'])])
ref = collect([(PC, ['longctx_aliasq_percase.jsonl',
               'longctx_aliasq_percase.gz'])])
check('full',              (stats(m112, 'full') or {}).get('f1'), 0.7189)
check('alias',             (stats(m112, 'alias') or {}).get('f1'), 0.6475)
check('quad4',             (stats(m112, 'quad4') or {}).get('f1'), 0.6366)
check('delete',            (stats(m112, 'delete') or {}).get('f1'), 0.4916)
check('budget_cluster_link', (stats(m112, 'budget_cluster_link') or {}).get('f1'), 0.4308)
check('alias_budget_link', (stats(m112, 'alias_budget_link') or {}).get('f1'), 0.4219)
check('alias_budget',      (stats(m112, 'alias_budget') or {}).get('f1'), 0.2232)
check('budget_cluster',    (stats(m112, 'budget_cluster') or {}).get('f1'), 0.2143)

# 恒等基线：本文 full 必须与论文参照批次逐例一致
a = (m112.get('full') or {})
b = (ref.get('full') or {})
same = sum(1 for k in b if k in a and f1(a[k]) == f1(b[k]))
print('  ---- 与论文参照批次的 full 逐例相同: %d / %d' % (same, len(b)))
if same < len(b):
    FAILS.append('112 identity baseline')

# quad4 的 token 效率
tk = [r.get('ctx_tokens') or 0 for r in (m112.get('quad4') or {}).values()]
tkf = [r.get('ctx_tokens') or 0 for r in a.values()]
if tk and tkf:
    ratio = (sum(tk) / len(tk)) / (sum(tkf) / len(tkf))
    print('  ---- quad4 平均 token 是全文的 %.1f%%（预期约 48.6%%）' % (100 * ratio))
    if abs(ratio - 0.486) > 0.01:
        FAILS.append('112 token ratio')

# ---------------------------------------------------------------- 3) public_lb
section('3) public_lb 三套')
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
section('4) 两种填充方向（反向 = reversed(drop_order)）')
for tag, pat, arm, want in (
    ('2728 反向 quad4', 'quad4_2728_rev_*.jsonl.gz', 'quad4', 0.4832),
    ('112 反向 quad4', 'quad4_112_rev_*.jsonl.gz', 'quad4', 0.5844),
    ('TypeA 反向 quad4', 'quad4_publb_rev_typea_*.jsonl.gz', 'quad4', 0.3172),
    ('TypeB 反向 quad4', 'quad4_publb_rev_typeb_*.jsonl.gz', 'quad4', 0.9719),
    ('TypeC 反向 quad4', 'quad4_publb_rev_typec_*.jsonl.gz', 'quad4', 0.0167),
):
    d = collect([(Q4, [pat])])
    check(tag, (stats(d, arm) or {}).get('f1'), want)


# ---------------------------------------------------------------- 5) 预算曲线
section('5) 预算曲线（112 例，5 个预算点，全部无金标）')


def collect_ratio(pattern):
    """按 (arm, ratio) 聚合，用于多预算点实验。"""
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
section('6) 跨模型（Mistral-7B，112 例 @0.50）')
mis = collect([(Q4, ['quad4_mistral112_*.jsonl.gz'])])
for arm, want in (('full', 0.6931), ('alias', 0.5931), ('delete', 0.4559),
                  ('quad4', 0.6946), ('quad4_del', 0.5520),
                  ('budget_cluster_link', 0.3837), ('alias_budget_link', 0.3748)):
    check('Mistral %s' % arm, (stats(mis, arm) or {}).get('f1'), want)

# ---------------------------------------------------------------- 7) 改述
section('7) 改述鲁棒性（112 例 @0.50）')
for tag, pat, expect in (
    ('改述A', 'quad4_para112_*_paraA_*.jsonl.gz',
     {'full': 0.6831, 'alias': 0.6297, 'delete': 0.4731, 'quad4': 0.6637,
      'budget_cluster_link': 0.3926, 'alias_budget_link': 0.3926}),
    ('改述B', 'quad4_para112_*_paraB_*.jsonl.gz',
     {'full': 0.6575, 'alias': 0.6141, 'delete': 0.4705, 'quad4': 0.6023,
      'budget_cluster_link': 0.4182, 'alias_budget_link': 0.4029}),
):
    print('  --- %s' % tag)
    d = collect([(Q4, [pat])])
    for arm, want in expect.items():
        check('    %s' % arm, (stats(d, arm) or {}).get('f1'), want)

# ---------------------------------------------------------------- 汇总
print()
print('=' * 78)
if FAILS:
    print('结果：%d 项 FAIL' % len(FAILS))
    for f in FAILS:
        print('  - %s' % f)
    sys.exit(1)
print('结果：全部 PASS —— 仓库内的原始结果与文档中的数字一致，可复现')
