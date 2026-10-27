"""Type B：同一批 LongBench 上下文，但问题是**引用注入 PII** 的合成问题。

目的：验证第 11 节提出的解释——
"自建集上 +0.15 的优势来自'问题点名敏感标识符'这一任务设计"。

做法：
  - 上下文**完全复用**（含已注入的 device-log 块），不重新生成、不重新压缩
  - 从上下文里正则解析出 {"device": X, "account": Y} 记录
  - 问题改为 "Which device is associated with account Y?"，gold = X
  -> 此时问题引用了 canary，link（问题同步别名）就有事可做了

预期：若 link 增益恢复为正，则第 11 节的解释成立；
      若仍为 0，说明公开上下文上即使问题引用实体，机制也不起作用。
"""
import json, io, re, random

RD = '/dpc/kuex0095/100069961_ku/privtrim'
SRC = RD + '/data/public_lb.jsonl'
OUT = RD + '/data/public_lb_typeb.jsonl'

REC = re.compile(r'\{"device": "([^"]+)", "account": "([^"]+)", "contact": "([^"]+)", "phone": "([^"]+)"\}')

rng = random.Random(20261027)
rows, skipped = [], 0
for line in io.open(SRC, encoding='utf-8'):
    if not line.strip():
        continue
    c = json.loads(line)
    recs = REC.findall(c['context'])
    if not recs:
        skipped += 1
        continue
    # 随机挑一条记录提问，避免总是第一条带来的位置偏差
    dev, acct, email, phone = recs[rng.randrange(len(recs))]
    q = 'Which device is associated with account %s?' % acct
    cc = dict(c)
    cc['key'] = c['key']                      # 关键：key 不变，才能复用已压缩产物
    cc['question'] = q
    cc['answers'] = [dev]
    cc['case_kind'] = 'single'
    cc['target_entity'] = acct          # 关键：link 机制靠它定位要保留的实体区域
    cc['_type'] = 'B'
    cc['_orig_question'] = c['question']
    rows.append(cc)

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('parsed %d cases (skipped %d without device log) -> %s' % (len(rows), skipped, OUT))
# 自检：答案确实在上下文里，且是可回答的
ok = sum(1 for r in rows if r['answers'][0] in r['context'])
print('答案串在上下文中的比例: %d/%d = %.1f%%' % (ok, len(rows), 100.0 * ok / max(1, len(rows))))
print('\n样例:')
for r in rows[:3]:
    print('  key      :', r['key'])
    print('  question :', r['question'])
    print('  gold     :', r['answers'][0])
    print('  原问题   :', r['_orig_question'][:70])
