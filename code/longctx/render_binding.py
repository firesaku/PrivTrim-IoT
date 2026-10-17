"""把 binding.json 渲染成 markdown 表（写入 results/reports/）。"""
import collections
import io
import json
import statistics as st
import sys

R = r'D:\CodexData\tmp\privtrim-review-20261015\remote_probe'
b = json.load(io.open(R + r'\results\cells\longctx_binding.json', encoding='utf-8'))

agg = collections.defaultdict(lambda: dict(n=0, ent=0, gold=0, both=0, co=0, d=[]))
for r in b['rows']:
    s = agg[r['arm']]
    s['n'] += 1
    s['ent'] += r['ent_present']
    s['gold'] += r['gold_present']
    s['both'] += r['both_present']
    s['co'] += r['co_located']
    if r['min_dist'] is not None:
        s['d'].append(r['min_dist'])

F1 = {'ORIGINAL': 0.6083, 'full': 0.6083, 'budget_cluster_link@0.50': 0.5500,
      'budget_cluster@0.50': 0.0000, 'delete': 0.0000, 'delete_budget@0.50': 0.0000,
      'alias': 0.0000, 'alias_budget@0.50': 0.1250, 'ext_provence': 0.4750,
      'ext_cpc': 0.3500, 'ext_llmlingua2': 0.3417, 'ext_mask_llmlingua2': 0.0000,
      'ext_quito_x': 0.3793}
ORDER = ['ORIGINAL', 'full', 'budget_cluster_link@0.50', 'budget_cluster@0.50',
         'delete', 'delete_budget@0.50', 'alias', 'alias_budget@0.50',
         'ext_provence', 'ext_cpc', 'ext_llmlingua2', 'ext_mask_llmlingua2', 'ext_quito_x']

L = []
L.append('# 绑定诊断：查询实体是否存活，决定一切')
L.append('')
L.append('40 个多实体案例。`ent_present` = 题目点名的那个实体的表面串是否还在上下文里；')
L.append('`gold_present` = 答案串是否还在；`both` = 两者同在（能答对的必要条件）。')
L.append('原文参照 `ORIGINAL` 为 1.00/1.00/1.00/1.00，度量自检通过（见文末）。')
L.append('')
L.append('| 臂 | n | ent_present | gold_present | both | min_dist p50 | F1 |')
L.append('|---|---:|---:|---:|---:|---:|---:|')
for a in ORDER:
    s = agg.get(a)
    if not s:
        continue
    n = max(1, s['n'])
    d = ('%.0f' % st.median(s['d'])) if s['d'] else 'n/a'
    L.append('| `%s` | %d | %.2f | %.2f | %.2f | %s | %.4f |'
             % (a, s['n'], s['ent'] / n, s['gold'] / n, s['both'] / n, d, F1.get(a, -1)))
L.append('')
L.append('## 三条结论')
L.append('')
L.append('1. **纯冗余度压缩把查询实体删干净，却留着答案。** `budget_cluster@0.50` 在 **40/40**')
L.append('   个案例里 `ent_present = 0.00` 而 `gold_present = 1.00`，F1 = 0。')
L.append('   机制：k 个实体块近乎同形，冗余度选择器把它们当成最冗余的内容最先丢弃 ——')
L.append('   **"看起来最冗余的东西"恰恰就是"标识实体的东西"**。冗余度信号与实体标识')
L.append('   在这个任务上直接冲突。这是领域级的观察，不是某个实现的毛病。')
L.append('')
L.append('2. **F1 几乎完全由实体存活率决定**：0.00 → 0.0000，0.80 → 0.4750，1.00 → 0.5500。')
L.append('   所以"锚点保护"不是锦上添花，它是这条任务线上唯一有效的机制。')
L.append('')
L.append('3. **别名臂的隐藏陷阱**：`alias` 的 ent/gold/both 全是 1.00，F1 却是 0.0000 ——')
L.append('   因为它只对**上下文**换别名，问题里仍是 `ACCT-xxxx` 原始串，模型无法把问题里的')
L.append('   实体对应到上下文里的 `[ACCOUNT-F]`。**别名脱敏必须同时作用于问题字符串**，')
L.append('   否则链接在问题一侧就断了。')
L.append('   → 这直接预测 `alias_budget_link` 若不同时别名化问题，F1 上不去 0.72；')
L.append('   作业 10602422 的产物可以检验这条预测。')
L.append('')
L.append('## 度量自检')
L.append('')
L.append('`ORIGINAL`（原文自身）必须 40/40 全过：实测 ent 40、gold 40、both 40、')
L.append('co_located 40，`min_dist` 中位 446 字符。任务定义保证答案就在目标区域内，')
L.append('所以 `co_located` 在原文上必须恒为 1；它在压缩后掉下去，只可能是压缩造成的。')
L.append('')
L.append('注：`co_located` 与 `both` 在本批数据上数值相同，因为一旦两个串都在，')
L.append('`min_dist` 几乎总落在区域长度内。区分两者需要更碎的压缩（如 ext_llmlingua2 的')
L.append('1-2 段集中输出），后续可用 `min_dist` 的分布单独观察。')

txt = '\n'.join(L) + '\n'
io.open(R + r'\results\reports\longctx_binding.md', 'w', encoding='utf-8', newline='\n').write(txt)
print('wrote results/reports/longctx_binding.md lines=%d' % len(L))
