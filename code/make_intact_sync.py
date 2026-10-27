"""② INTACT + 问题同步：把 INTACT 的替代映射同样应用到 question 上。

为什么需要：INTACT 把上下文里的 ACCT-xxx 换成 CODE_0，但 question 仍按
ACCT-xxx 提问，模型只能答 UNKNOWN -> F1 = 0。这是"referential dangling"。

本脚本从 INTACT 的输出（entity_mentions 里的 span_text 与 replacements_*）
构造 {原值 -> 替代值} 映射，重写 question，产出：
  data/intact_sync_cases.jsonl   —— 与原案例同 key，但 question 已同步
之后用既有的 --ext-arms 机制（上下文仍用 INTACT 的匿名化文本）评测即可。

注意：这一步**不重跑 INTACT**，只改评测输入，成本几乎为零。
"""
import json, io, glob, collections, sys

RD = '/dpc/kuex0095/100069961_ku/privtrim'

cases = {json.loads(x)['key']: json.loads(x)
         for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()}

# 1) 从 INTACT 输出构造 每篇文档的 原值 -> 替代值
maps = {}
for p in sorted(glob.glob(RD + '/data/intact_sample_s*_out.json')):
    for d in json.load(io.open(p, encoding='utf-8')):
        m = {}
        ann = list(d['annotations'].values())[0]
        for ent in ann['entity_mentions']:
            reps = None
            for k in ('replacements_mistral', 'replacements_rulebased'):
                v = ent.get(k)
                if v:
                    reps = v if isinstance(v, list) else [v]
                    break
            if reps and reps[0]:
                m[ent['span_text']] = reps[0]
        maps[d['doc_id']] = m

# 2) 这些文档对应哪些"已应用 INTACT"的 key（apply_intact.py 写过 out/ext_full/intact/）
import os
done = set(f[:-5] for f in os.listdir(RD + '/out/ext_full/intact') if f.endswith('.json'))

out_rows = []
changed = 0
multi = collections.Counter()
for key in sorted(done):
    c = cases.get(key)
    if not c:
        continue
    q = c['question']
    m = maps.get(key) or {}
    # 长串优先替换，避免短串先命中把长串拆坏
    for orig in sorted(m, key=len, reverse=True):
        if orig and orig in q:
            q = q.replace(orig, m[orig])
            multi[orig] += 1
    if q != c['question']:
        changed += 1
    cc = dict(c)
    cc['question'] = q
    cc['_original_question'] = c['question']
    out_rows.append(cc)

with io.open(RD + '/data/intact_sync_cases.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('cases with INTACT output : %d' % len(done))
print('cases written            : %d' % len(out_rows))
print('questions actually changed: %d (%.1f%%)' % (changed, 100.0 * changed / max(1, len(out_rows))))
print('\n=== 同步样例 ===')
shown = 0
for r in out_rows:
    if r['question'] != r['_original_question']:
        print('  BEFORE: %s' % r['_original_question'][:110])
        print('  AFTER : %s' % r['question'][:110])
        print()
        shown += 1
        if shown >= 3:
            break
