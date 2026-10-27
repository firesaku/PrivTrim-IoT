import json, io, re, collections

RD = '/dpc/kuex0095/100069961_ku/privtrim'
cs = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]

FLOAT = re.compile(r'-?\d+\.\d+')
INT = re.compile(r'(?<![\w.-])\d{1,7}(?![\w.])')

by_src = collections.defaultdict(lambda: {'n': 0, 'has_float': 0, 'numeric_ans': 0,
                                          'has_attr': 0, 'q_num': 0})
samples = []

for c in cs:
    s = c.get('source')
    d = by_src[s]
    d['n'] += 1
    ctx = c['context']
    fl = FLOAT.findall(ctx)
    if fl:
        d['has_float'] += 1
    ans = c.get('answers') or []
    a0 = str(ans[0]) if ans else ''
    if re.fullmatch(r'-?\d+(\.\d+)?', a0.strip()):
        d['numeric_ans'] += 1
    av = c.get('attribute_values')
    if av:
        d['has_attr'] += 1
    if re.search(r'\bhow many\b|\bhow much\b|\bwhat (?:is|was) the (?:temperature|reading|value|number)\b',
                 c['question'], re.I):
        d['q_num'] += 1
    if fl and len(samples) < 3:
        samples.append((c['key'], c['question'][:110], a0[:40], fl[:6]))

print('%-20s %6s %9s %11s %9s %8s' % ('source', 'n', 'has_float', 'numeric_ans', 'has_attr', 'q_numeric'))
print('-' * 70)
for s, d in sorted(by_src.items(), key=lambda x: -x[1]['n']):
    print('%-20s %6d %9d %11d %9d %8d' % (s, d['n'], d['has_float'], d['numeric_ans'],
                                          d['has_attr'], d['q_num']))

print('\n=== 带浮点数的上下文样例 ===')
for k, q, a, fl in samples:
    print('  %-36s ans=%-22s floats=%s' % (k[:36], a, fl))
    print('     Q: %s' % q)

print('\n=== 数值答案的分布（按 source）===')
for s in sorted(by_src):
    ex = [str((c.get('answers') or [''])[0])[:20] for c in cs
          if c.get('source') == s and re.fullmatch(r'-?\d+(\.\d+)?', str((c.get('answers') or [''])[0]).strip())]
    if ex:
        print('  %-20s n=%d  例: %s' % (s, len(ex), ex[:5]))
