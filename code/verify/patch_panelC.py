"""把图的面板 C 从"三个预算 × 两臂"改成"两种案例类型 × 两臂"，
让 40 例多实体的效应和它 72 例 needle 的对照同图出现。

按行号范围替换（105..138，1-based），不依赖精确文本匹配。
"""
import io

P = r'D:\CodexData\tmp\privtrim-review-20261015\hpc\plot_longctx.py'
L = io.open(P, encoding='utf-8').read().split('\n')

# 定位：从 "# ---------------- C:" 到 "# ---------------- D:" 之前
i0 = next(i for i, l in enumerate(L) if l.startswith('# ---------------- C:'))
i1 = next(i for i, l in enumerate(L) if l.startswith('# ---------------- D:'))

NEW = '''# ---------------- C: the anchor rule, multi vs needle control --------------
ax = axes[1][0]
sub = d.get('by_case_kind', {})
KINDS = [('multi', 'multi-entity\\n(disambiguation)'), ('needle', 'needle\\n(single fact)')]
BUDC = '0.5'
agg = {}
for k, v in sub.items():
    parts = k.split('|')
    if len(parts) != 3 or parts[2] != BUDC:
        continue
    kind, arm = parts[0], parts[1]
    if kind not in ('multi', 'needle') or arm not in ('budget_cluster', 'budget_cluster_link'):
        continue
    agg[(kind, arm)] = (v['f1'] / v['n'], v['n'])
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
ax.set_title('C  Anchor rule: gain concentrated in multi-entity cases',
             loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.grid(alpha=0.18, lw=0.6, axis='y')
multi_n = agg.get(('multi', 'budget_cluster'), (0, 40))[1] or 40

'''

L[i0:i1] = NEW.split('\n')
io.open(P, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))

import py_compile
py_compile.compile(P, doraise=True)
print('OK patched panel C (lines %d..%d replaced)' % (i0 + 1, i1))
