"""比对 --alias-question 前后（作业 A vs r5），并利用内建不变量做校验。

内建不变量
----------
`question_for_arm` 对**非别名臂**逐字返回原问题，所以 r5 运行里
`full` / `budget_cluster_link` / `budget_cluster` / `delete*` 等臂的每一格
**必须与作业 A 完全相同**。只要有一格不同，就说明补丁影响了不该影响的东西。
注意实体诊断的 alias 分支已修（预播 Aliaser），所以 `alias*` 臂的 tent 在两份产物里
都已经是对的，可以直接比。
"""
import io
import json
import sys

A = r'D:\CodexData\tmp\privtrim-review-20261015\fixres\aliaslink_qwen\cells.json'
B = r'D:\CodexData\tmp\privtrim-review-20261015\fixres\aliasq_qwen\cells.json'
ARMS_ALIAS = ('alias', 'alias_budget', 'alias_budget_link')

a = json.load(io.open(A, encoding='utf-8'))
b = json.load(io.open(B, encoding='utf-8'))
ka, kb = a['all'], b['all']
comm = sorted(set(ka) & set(kb), key=lambda k: (k.split('|')[0], k.split('|')[1]))


def cell(v):
    n = max(1, v['n'])
    return (v['f1'] / n, v['em'] / n, v['tok'] / n, v['ans'] / n, v['coll'] / n, v['tent'] / n)


# ---- 不变量：非别名臂必须逐格相同 ----
bad = []
for k in comm:
    arm = k.split('|')[0]
    if arm in ARMS_ALIAS:
        continue
    if cell(ka[k]) != cell(kb[k]):
        bad.append((k, cell(ka[k]), cell(kb[k])))
print('INVARIANT :: 非别名臂共 %d 格，与作业 A 不同的 %d 格 %s'
      % (len([k for k in comm if k.split('|')[0] not in ARMS_ALIAS]), len(bad),
         bad[:3] if bad else 'OK'))

# ---- 别名臂：前后对比 ----
out = []
for k in comm:
    arm = k.split('|')[0]
    if arm not in ARMS_ALIAS:
        continue
    fa, ea, ta, ga, ca, na = cell(ka[k])
    fb, eb, tb, gb, cb, nb = cell(kb[k])
    out.append('%s  F1 %.4f -> %.4f (%+.4f)  EM %.4f -> %.4f  tok %.0f -> %.0f  '
               'gic %.3f -> %.3f  coll %.3f -> %.3f  tent %.3f -> %.3f'
               % (k, fa, fb, fb - fa, ea, eb, ta, tb, ga, gb, ca, cb, na, nb))
print('ALIAS ARMS :: ' + ' ;; '.join(out))

# ---- 汇总 ----
for tag, o in (('A(no-flag)', ka), ('r5(alias-question)', kb)):
    tot = sum(v['f1'] for v in o.values())
    n = sum(v['n'] for v in o.values())
    print('TOTAL %-18s arms=%d  mean_F1=%.4f' % (tag, len(o), tot / max(1, n)))

# ---- 按 case_kind ----
for tag, o in (('A', a), ('r5', b)):
    bk = o.get('by_case_kind') or {}
    for kind in ('multi_entity', 'needle'):
        parts = []
        for arm in ARMS_ALIAS:
            for k in sorted(bk):
                if k.startswith(kind + '|' + arm + '|0.50'):
                    v = bk[k]
                    parts.append('%s F1=%.4f' % (arm, v['f1'] / max(1, v['n'])))
        if parts:
            print('BY-KIND %s %-13s %s' % (tag, kind, ' | '.join(parts)))
