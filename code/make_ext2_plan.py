"""为新增外部基线生成静态分片计划（提交前算好，避免运行期竞态）。

用法:
  python make_ext2_plan.py <method> <nshards>

产物 data/ext2_plan_<method>.json: {"<task>": {"method": m, "keys": [...]}}
自带自检：各分片并集 == 待压缩集合，且无重复。
"""
import json, io, os, sys

RD = '/path/to/project'
EXT = RD + '/out/ext_full'

if len(sys.argv) < 3:
    raise SystemExit('usage: make_ext2_plan.py <method> <nshards>')
method = sys.argv[1]
nsh = int(sys.argv[2])

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
allkeys = [c['key'] for c in cases]

d = os.path.join(EXT, method)
done = set(f[:-5] for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else set()
todo = [k for k in allkeys if k not in done]

plan = {str(i): {'method': method, 'keys': todo[i::nsh]} for i in range(nsh)}

flat = [k for p in plan.values() for k in p['keys']]
assert len(flat) == len(set(flat)), 'plan 有重复'
assert set(flat) == set(todo), 'plan 覆盖不全'

out = RD + '/data/ext2_plan_%s.json' % method
with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(plan, f, ensure_ascii=False)

sizes = [len(p['keys']) for p in plan.values()]
print('method=%s  total=%d  done=%d  todo=%d  shards=%d  size(min/max)=%d/%d'
      % (method, len(allkeys), len(done), len(todo), nsh, min(sizes), max(sizes)))
print('自检 PASS ->', out)
