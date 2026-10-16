"""
生成"分散式"敏感注入样本集（与连续同构块对照）
输入：crossdata/frozen_samples.jsonl（块式注入）
输出：crossdata/frozen_samples_scattered.jsonl

做法：把 context 里的 "Private contact record:" 整块剥离，再把三行分别插到
      prefix / 1-3 处 / middle / suffix 四个不同位置，使敏感项不再同构相邻。
"""
import io, json, os, re, hashlib

D = r"D:\CodexData\tmp\privtrim-review-20261015\crossdata"
SRC = os.path.join(D, 'frozen_samples.jsonl')
DST = os.path.join(D, 'frozen_samples_scattered.jsonl')

BLOCK_RE = re.compile(r'Private contact record:\s*\n(?:.*\n)*?Account ID:\s*[^\n]*\n?', re.M)

recs = [json.loads(l) for l in io.open(SRC, encoding='utf-8') if l.strip()]
out = []
for r in recs:
    ctx = r['context']
    bare = BLOCK_RE.sub('', ctx).strip()
    bare = re.sub(r'\n{3,}', '\n\n', bare)
    if len(bare) < 200:
        bare = ctx  # 兜底：剥离失败则保留原样
    lines = [
        "Contact email: %s" % r['canaries'][0]['value'],
        "Contact phone: %s" % r['canaries'][1]['value'],
        "Account ID: %s" % r['canaries'][2]['value'],
    ]
    # 四个插入点
    n = len(bare)
    pts = sorted({0, n // 4, n // 2, (3 * n) // 4, n})
    def snap(p):
        # 吸附到最近的换行/句末，避免切断句子
        for d in range(0, 80):
            for q in (p + d, p - d):
                if 0 <= q < n and (q == 0 or bare[q - 1] in '.!?\n'):
                    return q
        return p
    pts = [snap(p) for p in pts]
    pts = sorted(set(pts))
    if len(pts) < 4:
        pts = [0, n // 3, (2 * n) // 3, n]
    pieces = []
    prev = 0
    for i, p in enumerate(pts):
        pieces.append(bare[prev:p])
        if i < len(lines):
            pieces.append("\n" + lines[i] + "\n")
        prev = p
    pieces.append(bare[prev:])
    scattered = re.sub(r'\n{3,}', '\n\n', "".join(pieces)).strip()

    rr = dict(r)
    rr['context'] = scattered
    rr['canary_position'] = 'scattered'
    rr['sid'] = r['sid'] + '_sc'
    out.append(rr)

with io.open(DST, 'w', encoding='utf-8') as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

h = hashlib.sha256(io.open(DST, 'rb').read()).hexdigest()
print("records:", len(out))
print("sha256:", h)
print("out:", DST)
# 抽查一条
s = out[0]['context']
print("--- sample head ---")
print(s[:300].replace("\n", " | "))
