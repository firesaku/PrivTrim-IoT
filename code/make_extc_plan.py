"""在提交前把 30 个压缩分片的案例分配**静态**算好，消除运行期竞态。

问题背景:
  原 slurm 里每个 task 各自现算 todo = [未压缩的案例] 然后 todo[S::N]。
  各 task 启动时刻不同 -> todo 长度不同 -> 同一 S 切到的是不同列表，
  会互相覆盖并漏掉案例（幂等跳过只能防重复，防不了漏跑）。

本脚本一次性生成 data/extc_plan.json:
  {"<task_id>": {"method": <m>, "keys": [case_key, ...]}, ...}
每个 task 只读自己的那份，与启动先后无关。
"""
import json, io, os

RD = '/path/to/project'
EXT = RD + '/out/ext_full'
PLAN = RD + '/data/extc_plan.json'

# task 区间 -> (method, nshards)，与 slurm 的映射一致
LAYOUT = [('quito_x', 4), ('provence', 4), ('cpc', 10), ('llmlingua2', 6), ('mask_llmlingua2', 6)]

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
print('total cases: %d' % len(cases))

plan = {}
tid = 0
for method, n in LAYOUT:
    d = os.path.join(EXT, method)
    done = set(f[:-5] for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else set()
    todo = [c['key'] for c in cases if c['key'] not in done]
    print('%-18s done=%4d  todo=%4d  -> %d shards' % (method, len(done), len(todo), n))
    for s in range(n):
        plan[str(tid)] = {'method': method, 'keys': todo[s::n]}
        tid += 1

# 自检：每个方法的各分片并集必须恰好等于 todo，且无重复
ok = True
for method, n in LAYOUT:
    parts = [plan[str(i)] for i in range(30) if plan[str(i)]['method'] == method]
    union = [k for p in parts for k in p['keys']]
    if len(union) != len(set(union)):
        print('!! %s 有重复' % method); ok = False
for method, n in LAYOUT:
    d = os.path.join(EXT, method)
    done = set(f[:-5] for f in os.listdir(d) if f.endswith('.json'))
    expect = set(c['key'] for c in cases) - done
    got = set(k for p in plan.values() if p['method'] == method for k in p['keys'])
    if got != expect:
        print('!! %s 覆盖不完整: 缺 %d 例' % (method, len(expect - got))); ok = False

sizes = [len(plan[str(i)]['keys']) for i in range(30)]
print('\ntasks=%d  size min=%d max=%d  total=%d' % (len(plan), min(sizes), max(sizes), sum(sizes)))
print('self-check:', 'PASS' if ok else 'FAIL')

with io.open(PLAN, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(plan, f, ensure_ascii=False)
print('wrote', PLAN, os.path.getsize(PLAN), 'bytes')
