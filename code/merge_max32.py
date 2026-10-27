"""Merge the 32 max32 shards into one result set + aggregate tables.

用法（在登录节点）：
  cd /dpc/kuex0095/100069961_ku/privtrim
  env/bin/python merge_max32.py

幂等：分片不全时只报告缺哪些，不写合并文件。
"""
import json, io, os, sys, collections

RD = '/dpc/kuex0095/100069961_ku/privtrim'
N = 32

rows, missing, per_shard = [], [], {}
for s in range(N):
    p = '%s/out/max32_s%d_percase.jsonl' % (RD, s)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        missing.append(s)
        continue
    n = 0
    for x in io.open(p, encoding='utf-8'):
        if x.strip():
            rows.append(json.loads(x)); n += 1
    per_shard[s] = n

print('shards present : %d/%d' % (N - len(missing), N))
print('shards missing : %s' % (missing if missing else 'none'))
if per_shard:
    vals = sorted(per_shard.values())
    print('records/shard  : min=%d max=%d' % (vals[0], vals[-1]))
print('total records  : %d' % len(rows))
if not rows:
    sys.exit(0)
print('record keys    : %s' % sorted(rows[0].keys()))
keys = set(r.get('case_key') for r in rows)
print('distinct cases : %d' % len(keys))

if missing:
    print('\nINCOMPLETE - 不写合并文件。等 %d 个分片跑完再执行本脚本。' % len(missing))
    sys.exit(0)

out = RD + '/out/max32_percase.jsonl'
with io.open(out, 'w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('merged -> %s' % out)


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


def table(L, title):
    print('\n=== %s (%d cases) ===' % (title, len(set(r.get('case_key') for r in L))))
    print('%-24s %6s %7s %8s %8s %9s %9s' % ('arm', 'ratio', 'n', 'F1', 'EM', 'tgt_pres', 'leak'))
    print('-' * 78)
    agg = collections.defaultdict(list)
    for r in L:
        agg[(r.get('arm'), r.get('ratio'))].append(r)
    for k in sorted(agg, key=lambda x: (str(x[0]), -1 if x[1] is None else float(x[1]))):
        v = agg[k]; n = len(v)
        f1 = sum(num(r, 'f1') for r in v) / n
        em = sum(num(r, 'em') for r in v) / n
        tp = sum(num(r, 'tgt_entity_present') for r in v) / n
        lk = sum(1 for r in v
                 if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
        print('%-24s %6s %7d %8.4f %8.4f %9.3f %9.3f'
              % (k[0], k[1], n, f1, em, tp, lk))


table(rows, 'ALL')
for kind in sorted(set(r.get('case_kind') for r in rows if r.get('case_kind'))):
    table([r for r in rows if r.get('case_kind') == kind], kind)
