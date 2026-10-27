"""把 LongBench 案例转成 Perception Compressor 期待的格式。

它要的格式（读自 PerceptionCompressor.py 的 main）:
  qa_data.jsonl    每行 {"question": str, "ctxs": [{"title": str, "text": str}, ...]}
  query_dict.json  {question: [引导子问题, ...]}   —— 代码取 [:3]

注意：
  - 上下文仍然来自 LongBench，按段落切分为多个 ctx（它本身是"多文档排序+压缩"框架）
  - 注入的 device-log 块保留在上下文中（泄漏指标才有对象）
"""
import json, io, re

RD = '/dpc/kuex0095/100069961_ku/privtrim'
SRC = RD + '/data/public_lb.jsonl'          # Type A：LongBench 原问题
OUT_QA = RD + '/data/pc_qa.jsonl'
OUT_QD = RD + '/data/pc_query_dict_questions.json'   # 待生成引导问题的清单

MAX_CTX = 12          # 切成多少段（段太少排序无意义，太多 PPL 计算变慢）
MIN_LEN = 40


def split_ctx(text):
    # 先按空行分段；不足则按句子均分
    parts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(parts) < 3:
        sents = re.split(r'(?<=[.!?])\s+', text)
        n = max(1, len(sents) // 30)
        parts = [''.join(sents[i:i + n]) for i in range(0, len(sents), n)]
    # 合并到不超过 MAX_CTX 段
    while len(parts) > MAX_CTX:
        merged, step = [], len(parts) // MAX_CTX
        for i in range(0, len(parts), step + 1):
            merged.append('\n'.join(parts[i:i + step + 1]))
        parts = merged
    parts = [p for p in parts if len(p) >= MIN_LEN] or [text]
    return parts


rows, qs = [], []
for line in io.open(SRC, encoding='utf-8'):
    if not line.strip():
        continue
    c = json.loads(line)
    parts = split_ctx(c['context'])
    rows.append({
        'key': c['key'],
        'question': c['question'],
        'ctxs': [{'title': 'part%d' % i, 'text': p} for i, p in enumerate(parts)],
    })
    qs.append(c['question'])

with io.open(OUT_QA, 'w', encoding='utf-8', newline='\n') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with io.open(OUT_QD, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(sorted(set(qs)), f, ensure_ascii=False, indent=0)

print('qa_data  -> %s  (%d cases)' % (OUT_QA, len(rows)))
print('待生成引导问题的问题数 -> %d' % len(set(qs)))
ns = [len(r['ctxs']) for r in rows]
print('每例 ctx 段数: min=%d max=%d avg=%.1f' % (min(ns), max(ns), sum(ns) / len(ns)))
print('\n样例:')
r = rows[0]
print('  question:', r['question'])
print('  ctxs   :', len(r['ctxs']), '段，长度 =', [len(x['text']) for x in r['ctxs']][:6])
