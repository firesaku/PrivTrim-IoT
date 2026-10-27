"""Type C：在公开上下文上构造「锚点单元 ≠ 答案承载单元」的两跳任务。

为什么需要它：
  自建集上 link 带来 +0.33 的增益，但公开基准上两臂完全相同。
  代码级原因是 protect_of 只保护"除答案承载单元之外的锚点单元"，
  而 Type A/B 里账号与设备同处一行（同一个 unit），保护无增量。

构造（每个案例追加两个块，每行是一个 unit）:
  ----- account directory -----
  account ACCT-h1 belongs to user U-h1
  ...（5 行）
  ----- user registry -----
  user U-h1 operates device sensor-x1
  ...（5 行）

问题: "Which device does account ACCT-hk operate?"  gold = sensor-xk
  -> 锚点 ACCT-hk 在 directory 行（不含 gold）
  -> 答案 sensor-xk 在 registry 行（不含锚点）
  -> 必须两跳，link 保护锚点行才有意义

不需要外部方法压缩（本实验只比较我们自己的 4 个臂）。
"""
import json, io, hashlib, random, os

RD = '/path/to/project'
LB = RD + '/data/longbench/data'
TASKS = ['multifieldqa_en', 'qasper', 'hotpotqa']
OUT = RD + '/data/public_lb_typec.jsonl'


def build(per_task=100, seed=20261028):
    rng = random.Random(seed)
    rows = []
    for task in TASKS:
        src = [json.loads(x) for x in io.open('%s/%s.jsonl' % (LB, task), encoding='utf-8')
               if x.strip()][:per_task]
        for i, r in enumerate(src):
            key = 'lbc-%s-%04d' % (task, i)
            h = hashlib.sha256(key.encode()).hexdigest()
            accs, users, devs = [], [], []
            for k in range(5):
                hh = h[k * 4:(k + 1) * 4]
                accs.append('ACCT-%s' % hh)
                # 用户名与设备名必须与账号哈希**无关**，否则模型可纯字符串推断，
                # 不必做检索，会虚高所有方法并稀释 link 的效应。
                users.append('U-%06x' % rng.randrange(16 ** 6))
                devs.append('sensor_%06x_%02d' % (rng.randrange(16 ** 6), rng.randrange(64)))
            tgt = rng.randrange(5)

            lines = ['', '----- account directory -----']
            for a, u in zip(accs, users):
                lines.append('account %s belongs to user %s' % (a, u))
            lines.append('----- user registry -----')
            for u, d in zip(users, devs):
                lines.append('user %s operates device %s' % (u, d))
            ctx = r['context'] + '\n'.join(lines) + '\n'

            rows.append(dict(
                key=key, source=task, case_kind='single', dataset='LongBench-C',
                context=ctx,
                question='Which device does account %s operate?' % accs[tgt],
                answers=[devs[tgt]],
                target_entity=accs[tgt],
                canaries=[dict(type='account', value=a, marker=a, fragment_length=8)
                          for a in accs],
            ))
    with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    return rows


rows = build()
print('wrote %d -> %s' % (len(rows), OUT))

# 自检 1: gold 在上下文里
ok = sum(1 for r in rows if r['answers'][0] in r['context'])
print('gold 在上下文中: %d/%d' % (ok, len(rows)))

# 自检 2: 关键 —— 锚点行里不含 gold，答案行里不含锚点（保证分属不同 unit）
def split_units(t):
    import re
    parts = re.split(r'(?<=[.!?])\s+|\n+', t)
    return [p.strip() for p in parts if len(p.strip()) >= 2]

sep = 0
for r in rows:
    us = split_units(r['context'])
    a_units = [u for u in us if r['target_entity'] in u]
    g_units = [u for u in us if r['answers'][0] in u]
    if a_units and g_units and not (set(a_units) & set(g_units)):
        sep += 1
print('锚点单元与答案单元分离: %d/%d = %.1f%%' % (sep, len(rows), 100.0 * sep / len(rows)))
print('\n样例:')
r = rows[0]
print('  question :', r['question'])
print('  gold     :', r['answers'][0])
print('  target   :', r['target_entity'])
tail = r['context'][-320:]
print('  注入块结尾:')
for ln in tail.splitlines()[-8:]:
    print('   ', ln)
