"""把 out/ext_full/<method>/<key>.json 汇总成 e2e 需要的 data/ext_arms_all.jsonl。

格式（由 e2e_longctx_r5full.py:673 读取）:
    {"key": <case_key>, "method": <arm 后缀>, "text": <压缩后上下文>}
e2e 会把 method 变成 arm 名 `ext_<method>`。

复用约定:
  - 每个方法产 matched / native 两个变体（ext_compress.py 的 `matched` 与 `native`）
  - `full` 不是真压缩，直接用原始 context，作为 ext_full 参照臂
  - 幂等：每次全量重扫，输出覆盖写；压缩跑到哪，这里就能出到哪
"""
import json, io, os, sys, collections

RD = '/path/to/project'
EXT = RD + '/out/ext_full'
MAXJ = RD + '/data/longctx_max.jsonl'
# 允许调用方指定输出路径，避免多个并发 task 同时写同一个文件
OUT = sys.argv[1] if len(sys.argv) > 1 else RD + '/data/ext_arms_all.jsonl'
# 第二个参数可选：只收这些方法（逗号分隔）。用于"只补新增臂"的增量评测，
# 避免把已经评过的 12 个臂再算一遍。
ONLY = [x.strip() for x in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[2].strip() else None
METHODS = ['quito_x', 'provence', 'cpc', 'llmlingua2', 'mask_llmlingua2', 'dac',
           'longllmlingua', 'selective_context', 'intact', 'ner_redact']
if ONLY:
    METHODS = [m for m in METHODS if m in ONLY]
    print('ONLY filter -> %s' % METHODS)

cases = [json.loads(x) for x in io.open(MAXJ, encoding='utf-8') if x.strip()]
print('cases in longctx_max : %d' % len(cases))

# --- 结构自检：打印一个样本的字段，便于核对 ---
sample = None
for m in METHODS:
    d = os.path.join(EXT, m)
    if os.path.isdir(d):
        fs = [f for f in os.listdir(d) if f.endswith('.json')]
        if fs:
            sample = (m, os.path.join(d, fs[0]))
            break
if sample:
    o = json.load(io.open(sample[1], encoding='utf-8'))
    print('sample %s/%s keys: %s' % (sample[0], os.path.basename(sample[1]), sorted(o.keys())))
    for k in ('status', 'matched', 'native', 'original_tokens', 'target_tokens'):
        v = o.get(k)
        if isinstance(v, dict):
            print('   %-14s -> %s (tokens=%s)' % (k, sorted(v.keys()), v.get('tokens')))
        else:
            print('   %-14s -> %r' % (k, v))

rows = []
cov = collections.defaultdict(int)
errors = collections.Counter()

for m in METHODS:
    d = os.path.join(EXT, m)
    if not os.path.isdir(d):
        print('%-16s MISSING DIR' % m); continue
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        key = f[:-5]
        try:
            o = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        except Exception:
            errors[m + '/unreadable'] += 1
            continue
        if o.get('status') != 'ok':
            errors[m + '/' + str(o.get('status'))] += 1
            continue
        mt = (o.get('matched') or {}).get('text')
        nt = (o.get('native') or {}).get('text')
        if mt:
            rows.append({'key': key, 'method': m, 'text': mt}); cov[m] += 1
        if nt:
            rows.append({'key': key, 'method': m + '_native', 'text': nt}); cov[m + '_native'] += 1

# full 参照臂：原始上下文，不需要 GPU
for c in cases:
    if c.get('context'):
        rows.append({'key': c['key'], 'method': 'full', 'text': c['context']}); cov['full'] += 1

with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('\nwrote %s  rows=%d' % (OUT, len(rows)))
print('%-20s %8s' % ('arm', 'cases'))
for k in sorted(cov):
    print('%-20s %8d' % (k, cov[k]))
if errors:
    print('\nnon-ok records:')
    for k, v in errors.most_common():
        print('   %-30s %d' % (k, v))

# 覆盖率：有多少案例至少有一个外部方法
by_case = collections.defaultdict(set)
for r in rows:
    by_case[r['key']].add(r['method'])
non_full = {k: v for k, v in by_case.items() if v - {'full'}}
print('\ncases with >=1 external arm : %d / %d (%.1f%%)'
      % (len(non_full), len(cases), 100.0 * len(non_full) / len(cases)))
