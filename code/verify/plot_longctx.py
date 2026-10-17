"""长上下文主结果图（yf-skill 数据图规范：白底、低饱和、方向符号、单位标注）

面板：
  A  F1 vs token 预算 —— 三个选择器臂 + full 参考线，标出 50-75% 平坦区间
  B  目标实体存活率 vs 预算 —— (A) 的机制中间量
  C  多实体子集：同预算下 cluster vs cluster_link
  D  delete vs alias：效用持平但 collision 二元

数据源：results/cells/longctx_r3_cells.json（112 例，maxlen=24576，零截断）
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SRC = r'D:\CodexData\tmp\privtrim-review-20261015\fixres\r3_cells.json'
OUT = r'D:\CodexData\tmp\privtrim-review-20261015\remote_probe\results\figures'
os.makedirs(OUT, exist_ok=True)

d = json.load(open(SRC, encoding='utf-8'))
al = d['all']

BUD = [0.25, 0.50, 0.75]
ARMS = {
    'budget_random':       ('Random drop',      '#9aa5b1', 'o', '-'),
    'budget_cluster':      ('Redundancy drop',  '#6b8e9f', 's', '-'),
    'budget_cluster_link': ('+ entity anchor',  '#2f5d7c', 'D', '-'),
}


def get(arm, bud):
    k = '%s|%s' % (arm, bud)
    v = al.get(k)
    if not v:
        return None
    n = v['n']
    return dict(f1=v['f1'] / n, tok=v['tok'] / n, tent=v.get('tent', 0) / n,
                coll=v.get('coll', 0) / n, ans=v.get('ans', 0) / n, n=n)


full = get('full', 'None') or al.get('full|None')
full = al['full|None']
full_f1 = full['f1'] / full['n']
full_tok = full['tok'] / full['n']

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))

# ---------------- A: F1 vs budget, with the flat region shaded --------------
CI = {}
_cip = r'D:\CodexData\tmp\privtrim-review-20261015\fixres\ci_f1.json'
if os.path.exists(_cip):
    CI = json.load(open(_cip, encoding='utf-8'))

ax = axes[0][0]
for arm, (lab, c, mk, ls) in ARMS.items():
    ys = [get(arm, b)['f1'] for b in BUD]
    xs = [get(arm, b)['tok'] for b in BUD]
    err = None
    if CI:
        lo = [ys[i] - CI['%s|%s' % (arm, BUD[i])][1] for i in range(len(BUD))]
        hi = [CI['%s|%s' % (arm, BUD[i])][2] - ys[i] for i in range(len(BUD))]
        err = [lo, hi]
    ax.errorbar(xs, ys, yerr=err, marker=mk, color=c, ls=ls, lw=1.6, ms=5,
                capsize=3, elinewidth=1.0, label=lab, zorder=3)
ax.axhline(full_f1, color='#b06a5a', ls='--', lw=1.3, zorder=2)
if CI.get('full|None'):
    f_lo, f_hi = CI['full|None'][1], CI['full|None'][2]
    ax.axhspan(f_lo, f_hi, color='#b06a5a', alpha=0.10, zorder=1)
    ax.text(1620, f_lo - 0.004, 'uncompressed  %.4f   95%% CI [%.3f, %.3f]'
            % (full_f1, f_lo, f_hi), ha='left', va='top', color='#b06a5a', fontsize=7.5)
else:
    ax.text(1620, full_f1 - 0.030, 'uncompressed  %.4f' % full_f1,
            ha='left', va='top', color='#b06a5a', fontsize=8)
x50, x75 = get('budget_cluster_link', 0.50)['tok'], get('budget_cluster_link', 0.75)['tok']
ax.axvspan(x50, x75, color='#2f5d7c', alpha=0.07, zorder=1)
ax.annotate('flat region 50-75%', xy=((x50 + x75) / 2, 0.775), ha='center',
            fontsize=8, color='#2f5d7c')
ax.set_xlabel('context tokens  (↓)')
ax.set_ylabel('F1  (↑)')
ax.set_title('A  Utility vs budget', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.grid(alpha=0.18, lw=0.6)

# ---------------- B: mechanism — target entity survival --------------------
ax = axes[0][1]
for arm in ('budget_random', 'budget_cluster', 'budget_cluster_link'):
    lab, c, mk, ls = ARMS[arm]
    ax.plot([get(arm, b)['tok'] for b in BUD], [get(arm, b)['tent'] for b in BUD],
            marker=mk, color=c, ls=ls, lw=1.6, ms=5, label=lab)
ax.axhline(1.0, color='#b06a5a', ls='--', lw=1.3)
ax.text(full_tok * 0.985, 1.012, 'uncompressed = 1.000', ha='right', va='bottom',
        color='#b06a5a', fontsize=8)
ax.set_xlabel('context tokens  (↓)')
ax.set_ylabel('target entity present  (↑)')
ax.set_title('B  Why A happens: the entity survives or it does not',
             loc='left', fontweight='bold')
ax.set_ylim(0.55, 1.09)
ax.grid(alpha=0.18, lw=0.6)

# ---------------- C: the anchor rule, multi vs needle control --------------
ax = axes[1][0]
sub = d.get('by_case_kind', {})
KINDS = [('multi', 'multi-entity\n(disambiguation)'), ('needle', 'needle\n(single fact)')]
BUDC = '0.5'
agg = {}
for k, v in sub.items():
    parts = k.split('|')
    if len(parts) != 3 or parts[2] != BUDC:
        continue
    kind, arm = parts[0], parts[1]
    # 注意：by_case_kind 的键是 'multi_entity' / 'needle'，必须前缀匹配。
    # 用等值判断会静默漏掉 multi（而 needle 恰好匹配），得到一张"部分正确"的图。
    kk = ('multi' if kind.startswith('multi')
          else 'needle' if kind.startswith('needle') else None)
    if kk is None or arm not in ('budget_cluster', 'budget_cluster_link'):
        continue
    agg[(kk, arm)] = (v['f1'] / v['n'], v['n'])
xx = np.arange(len(KINDS))
w = 0.34
for off, arm in ((-w / 2, 'budget_cluster'), (w / 2, 'budget_cluster_link')):
    lab, c, mk, ls = ARMS[arm]
    ys = [agg.get((kd, arm), (0, 0))[0] for kd, _ in KINDS]
    ns = [agg.get((kd, arm), (0, 0))[1] for kd, _ in KINDS]
    ax.bar(xx + off, ys, w, color=c, label=lab, edgecolor='white', lw=0.6)
    for xi, y, nn in zip(xx + off, ys, ns):
        ax.text(xi, y + 0.016, '%.3f' % y, ha='center', fontsize=8)
        ax.text(xi, 0.014, 'n=%d' % nn, ha='center', fontsize=6.5, color='white')
for xi, (kd, _) in zip(xx, KINDS):
    a = agg.get((kd, 'budget_cluster_link'), (0, 0))[0]
    b = agg.get((kd, 'budget_cluster'), (0, 0))[0]
    ax.text(xi, max(a, b) + 0.075, 'gain %+.3f' % (a - b), ha='center', fontsize=8.5,
            fontweight='bold', color='#2f5d7c')
ax.set_xticks(xx)
ax.set_xticklabels([lab for _, lab in KINDS], fontsize=8)
ax.set_ylabel('F1  (↑)')
ax.set_ylim(0, 1.04)
ax.set_title('C  Gain by case type',
             loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.grid(alpha=0.18, lw=0.6, axis='y')
multi_n = agg.get(('multi', 'budget_cluster'), (0, 40))[1] or 40


# ---------------- D: delete vs alias ---------------------------------------
ax = axes[1][1]
names = ['delete|None', 'alias|None', 'delete_budget|0.5', 'alias_budget|0.5']
labels = ['delete\nno compression', 'alias\nno compression',
          'delete\n@50% budget', 'alias\n@50% budget']
# collision 的分母是**多实体案例数**，不是全部 112 例
multi_n = 0
for k, v in sub.items():
    if k.startswith('multi') and k.endswith('0.5') and 'budget_cluster|' in k:
        multi_n = max(multi_n, v['n'])
if not multi_n:
    multi_n = 40
f1s = [al[n]['f1'] / al[n]['n'] for n in names]
colls = [al[n].get('coll', 0) for n in names]
xx = np.arange(len(names))
ax.bar(xx, f1s, 0.55, color='#6b8e9f', edgecolor='white', lw=0.6)
for xi, y, c in zip(xx, f1s, colls):
    ax.text(xi, y + 0.012, '%.4f' % y, ha='center', fontsize=7.5)
    pct = 100.0 * c / multi_n
    ax.text(xi, 0.030, 'identity\nlost\n%.0f%%' % pct, ha='center', va='bottom', fontsize=6.2,
            color='white' if pct > 50 else '#1d3d52', fontweight='bold')
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('F1  (↑)')
ax.set_ylim(0, 0.66)
ax.set_title('D  Same utility, opposite identity preservation',
             loc='left', fontweight='bold')
ax.grid(alpha=0.18, lw=0.6, axis='y')
print('multi_n (collision denominator) =', multi_n)

fig.suptitle('Long-context compression: 112 deterministic IoT cases, Qwen2.5-7B, '
             'zero truncation (maxlen=24576)', fontsize=9.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, 'longctx_main.' + ext), dpi=200,
                facecolor='white', bbox_inches='tight')
print('WROTE', os.path.join(OUT, 'longctx_main.png'))

# 打印图上用到的数字，便于正文核对
print('\n--- figure source numbers ---')
for arm in ('budget_random', 'budget_cluster', 'budget_cluster_link'):
    for b in BUD:
        r = get(arm, b)
        print('%-22s %.2f  F1=%.4f tok=%.0f tent=%.3f' % (arm, b, r['f1'], r['tok'], r['tent']))
print('%-22s --    F1=%.4f tok=%.0f' % ('full', full_f1, full_tok))
for k in ('multi|budget_cluster|0.5', 'multi|budget_cluster_link|0.5'):
    v = sub.get(k)
    if v:
        print('%-34s n=%d F1=%.4f tent=%.3f ans=%.3f' % (k, v['n'], v['f1'] / v['n'],
                                                         v.get('tent', 0) / v['n'],
                                                         v.get('ans', 0) / v['n']))
