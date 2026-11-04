"""
V2b 决定性测试（比例驱动 + 随机对照）
CPU only, 0 model calls.

对每条样本:
  把 context 切成 unit -> n-gram TF-IDF 向量 (numpy) -> 余弦相似度矩阵
  贪心合并：按相似度从高到低合并 unit，直到合并掉 ratio*n 个 unit
  对照：同样数量的合并，但配对随机
  损伤指标（每个 size>=2 的簇）:
    hit_answer : 簇内既有"含答案跨度"的 unit，又有不含的
    mix_sens   : 簇内既有命中敏感标识符的 unit，又有不含的
    fuse_2sec  : 簇内融合了两个不同敏感标识符  <- 最危险（把两个秘密混成一类）
"""
import io, json, os, re, math, argparse, random
from collections import defaultdict, Counter
import numpy as np

def grams(s, k=3):
    s = re.sub(r'[^a-z0-9]', '', s.lower())
    if len(s) < k:
        return [s] if s else []
    return [s[i:i+k] for i in range(len(s) - k + 1)]

def split_units(text, cap):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = [p.strip() for p in parts if len(p.strip()) >= 12]
    if not out:
        out = [text[:600]]
    return out[:cap]

def build_matrix(units):
    docs = [grams(u) for u in units]
    vocab = {}
    for d in docs:
        for g in set(d):
            if g not in vocab:
                vocab[g] = len(vocab)
    n, V = len(units), max(len(vocab), 1)
    df = np.zeros(V, dtype=np.float64)
    M = np.zeros((n, V), dtype=np.float64)
    for i, d in enumerate(docs):
        c = Counter(d)
        for g, f in c.items():
            M[i, vocab[g]] = f
        for g in set(d):
            df[vocab[g]] += 1
    idf = np.log((n + 1) / (df + 1)) + 1.0
    M *= idf
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return M / norm

def greedy_clusters(S, ratio, rng=None, random_mode=False):
    n = S.shape[0]
    target = int(ratio * n)
    if target < 2:
        return []
    if random_mode:
        idx = list(range(n)); rng.shuffle(idx)
        pairs = [(idx[i], idx[i+1]) for i in range(0, 2*(target//2), 2)]
    else:
        iu = np.triu_indices(n, k=1)
        sims = S[iu]
        order = np.argsort(-sims)
        pairs = []
        for k in order:
            if len(pairs) * 2 >= target:
                break
            pairs.append((int(iu[0][k]), int(iu[1][k])))
    used = set(); parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i, j in pairs:
        if i in used or j in used:
            continue
        union(i, j); used.add(i); used.add(j)
    g = defaultdict(list)
    for i in range(n):
        g[find(i)].append(i)
    return [v for v in g.values() if len(v) >= 2]

def exposure_hit(secret, text):
    n = re.sub(r'[^a-z0-9]', '', text.lower())
    if re.sub(r'[^a-z0-9]', '', secret['value'].lower()) in n:
        return True
    m = re.sub(r'[^a-z0-9]', '', secret['marker'].lower())
    k = secret['fragment_length']
    return any(m[i:i+k] in n for i in range(len(m) - k + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratios', default='0.25,0.5')
    ap.add_argument('--cap', type=int, default=150)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    recs = [json.loads(l) for l in io.open(a.samples, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    ratios = [float(x) for x in a.ratios.split(',')]

    # res[mode][ratio][source] = [clusters, hitAns, mixSens, fuse2]
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])))
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))

    for r in recs:
        units = split_units(r['context'], a.cap)
        if len(units) < 8:
            continue
        S = build_matrix(units) @ build_matrix(units).T
        np.fill_diagonal(S, -1.0)

        key = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
        ans_units = set()
        for ans in (r.get('answers') or []):
            kk = key(str(ans))
            if not kk: continue
            for i, u in enumerate(units):
                if kk in key(u):
                    ans_units.add(i)
        unit_sec = [{ci for ci, c in enumerate(r['canaries']) if exposure_hit(c, u)} for u in units]

        for mode in ('similarity', 'random'):
            for ratio in ratios:
                clusters = greedy_clusters(S, ratio, rng=rng, random_mode=(mode == 'random'))
                bucket = res[mode][ratio][r['source']]
                ab = agg[mode][ratio]
                for cl in clusters:
                    has_ans = any(i in ans_units for i in cl)
                    no_ans = any(i not in ans_units for i in cl)
                    ha = 1 if (ans_units and has_ans and no_ans) else 0
                    any_s = any(unit_sec[i] for i in cl)
                    any_ns = any(not unit_sec[i] for i in cl)
                    ms = 1 if (any_s and any_ns) else 0
                    fused = set()
                    for i in cl:
                        fused |= unit_sec[i]
                    f2 = 1 if len(fused) >= 2 else 0
                    for b in (bucket, ab):
                        b[0] += 1; b[1] += ha; b[2] += ms; b[3] += f2

    L = []
    w = L.append
    w("V2b FEASIBILITY: similarity-based merging vs random merging (same merge count)")
    w("samples=%d cap_units=%d ratios=%s seed=%d" % (len(recs), a.cap, ratios, a.seed))
    w("metrics = %% of merged clusters (size>=2) that ...")
    w("  hit_answer = mix answer-span unit with non-answer unit")
    w("  mix_sens   = mix sensitive unit with non-sensitive unit")
    w("  fuse_2sec  = fuse TWO DIFFERENT secrets into one cluster (worst)")
    w("")
    for mode in ('similarity', 'random'):
        w("### mode = %s" % mode)
        w("%-8s %-18s %9s %11s %11s %11s" % ("ratio", "source", "clusters", "hit_answer", "mix_sens", "fuse_2sec"))
        for ratio in ratios:
            for src in sorted(res[mode][ratio]):
                b = res[mode][ratio][src]
                if b[0] == 0: continue
                w("%-8.2f %-18s %9d %10.1f%% %10.1f%% %10.1f%%" % (ratio, src, b[0], 100*b[1]/b[0], 100*b[2]/b[0], 100*b[3]/b[0]))
            b = agg[mode][ratio]
            if b[0]:
                w("%-8.2f %-18s %9d %10.1f%% %10.1f%% %10.1f%%" % (ratio, "== ALL ==", b[0], 100*b[1]/b[0], 100*b[2]/b[0], 100*b[3]/b[0]))
            w("")
        w("")
    w("### 判定")
    for ratio in ratios:
        s = agg['similarity'][ratio]; r = agg['random'][ratio]
        if s[0] and r[0]:
            w("ratio=%.2f : sim hitAns=%.1f%% vs rand %.1f%% | sim fuse2=%.1f%% vs rand %.1f%% -> %s" % (
                ratio, 100*s[1]/s[0], 100*r[1]/r[0], 100*s[3]/s[0], 100*r[3]/r[0],
                "相似度合并在制造额外损伤" if (s[1]/s[0] - r[1]/r[0]) > 0.02 or (s[3]/s[0] - r[3]/r[0]) > 0.02 else "与随机无实质差别"))
    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    print("wrote", a.out)
    for ratio in ratios:
        s = agg['similarity'][ratio]; r = agg['random'][ratio]
        if s[0]:
            print("ratio=%.2f SIM  clusters=%d hitAns=%.1f%% mixSens=%.1f%% fuse2=%.1f%%" % (ratio, s[0], 100*s[1]/s[0], 100*s[2]/s[0], 100*s[3]/s[0]))
        if r[0]:
            print("ratio=%.2f RAND clusters=%d hitAns=%.1f%% mixSens=%.1f%% fuse2=%.1f%%" % (ratio, r[0], 100*r[1]/r[0], 100*r[2]/r[0], 100*r[3]/r[0]))

if __name__ == '__main__':
    main()
