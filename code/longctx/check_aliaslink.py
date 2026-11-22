"""核对作业 10602422 的产物：alias_budget_link 是否同时做到高效用 + 零泄漏 + 不毁身份。

单行输出，避免显示层把表格折叠掉。
"""
import io
import json
import sys

P = r'D:\CodexData\tmp\privtrim-review-20261015\fixres\aliaslink_qwen\cells.json'
o = json.load(io.open(P, encoding='utf-8'))
print('meta: cases=%s arms=%s budgets=%s model=%s' % (
    o.get('cases'), o.get('arms'), o.get('budgets'),
    (o.get('model') or '')[-40:]))

allb = o['all']
parts = []
for k in sorted(allb):
    v = allb[k]
    n = max(1, v['n'])
    parts.append('%s[n=%d F1=%.4f EM=%.4f tok=%.0f gic=%.3f leakV=%.2f%% leakF=%.2f%% out=%d coll=%.3f tent=%.3f]' % (
        k, v['n'], v['f1'] / n, v['em'] / n, v['tok'] / n, v['ans'] / n,
        100.0 * v['cv'] / max(1, v['cn']), 100.0 * v['cf'] / max(1, v['cn']),
        v['ov'], v['coll'] / n, v['tent'] / n))
print('ALL :: ' + ' || '.join(parts))

# 预算对账：超标与欠用
ba = o.get('budget_accounting') or {}
if ba:
    over = [(k, v) for k, v in ba.items() if v.get('devmax_signed', 0) > 0]
    print('BUDGET :: cells=%d  overshoot_cells=%d  %s'
          % (len(ba), len(over), over[:4] if over else 'none'))

# 逐例：alias_budget vs alias_budget_link 的差异（自检锚点保护是否生效）
per = [json.loads(l) for l in io.open(
    r'D:\CodexData\tmp\privtrim-review-20261015\fixres\aliaslink_qwen\percase.jsonl',
    encoding='utf-8') if l.strip()]
by = {}
for r in per:
    by[(r['arm'], r.get('ratio'), r['case_key'])] = r
diff = same = 0
link_more = 0
for (arm, ratio, key), r in by.items():
    if arm != 'alias_budget' or ratio != 0.5:
        continue
    o2 = by.get(('alias_budget_link', 0.5, key))
    if not o2:
        continue
    if r['pred'] != o2['pred']:
        diff += 1
    else:
        same += 1
    if o2['tgt_entity_present'] > r['tgt_entity_present']:
        link_more += 1
print('SELFCHECK :: alias_budget vs alias_budget_link @0.5: pred_differ=%d pred_same=%d link_keeps_more_entity=%d'
      % (diff, same, link_more))

# 按案例类型
bk = o.get('by_case_kind') or {}
for kind in sorted({k[0] for k in bk}):
    row = []
    for k in sorted(bk):
        if k[0] != kind:
            continue
        v = bk[k]
        n = max(1, v['n'])
        row.append('%s@%s F1=%.4f' % (k[1], k[2], v['f1'] / n))
    print('KIND %s :: %s' % (kind, ' | '.join(row)))
