"""为 DAC 生成静态分片计划。

为什么单独给 DAC 做计划:
  - dac 加载只要 1.6-3.0 秒（实测），所以把任务切碎几乎零代价；
  - 但 dac 每例 35.9s、p95 81.8s，方差极大，切大块会整块塞不进碎片化的空洞；
  - 实测运行中作业有 8 个在 15 分钟内结束 -> 切成 ~5 例/任务（约 9-15 分钟）最合身。

范围: 只补 396 例对照集里 DAC 还缺的部分（现有 217），
      让「6 个方法在同一批案例上」这张表能成立。全量 2332 例的 dac 需 27 GPU-h，
      不在本次范围（见 limitation）。

产物: data/dac_plan.json  {"<task>": {"method":"dac","keys":[...]}}
"""
import json, io, os

RD = '/path/to/project'
EXT = RD + '/out/ext_full'
PER = 5          # 每任务案例数
PLAN = RD + '/data/dac_plan.json'

base = set(f[:-5] for f in os.listdir(os.path.join(EXT, 'quito_x')) if f.endswith('.json'))
dac_done = set(f[:-5] for f in os.listdir(os.path.join(EXT, 'dac')) if f.endswith('.json'))

# 对照集 = longctx_full 的 396 例
comp = set()
for x in io.open(RD + '/data/longctx_full.jsonl', encoding='utf-8'):
    if x.strip():
        comp.add(json.loads(x)['key'])

todo = sorted(comp - dac_done)
print('对照集 396 例: %d' % len(comp))
print('dac 已完成    : %d' % len(comp & dac_done))
print('待补          : %d' % len(todo))

plan = {}
for i in range(0, len(todo), PER):
    plan[str(len(plan))] = {'method': 'dac', 'keys': todo[i:i + PER]}

# 自检：并集必须恰好等于 todo，且无重复
flat = [k for p in plan.values() for k in p['keys']]
assert len(flat) == len(set(flat)) == len(todo), 'plan 自检失败'
assert set(flat) == set(todo), 'plan 覆盖不全'

print('分片数        : %d  (每片 %d 例)' % (len(plan), PER))
with io.open(PLAN, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(plan, f, ensure_ascii=False)
print('自检 PASS ->', PLAN)
