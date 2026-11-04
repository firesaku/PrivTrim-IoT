"""
V2 决定性测试：距离有界聚类会不会"吃掉"效用关键跨度或敏感跨度？
CPU only, 0 model calls。

对每条样本：
  1) 把 context 切成 unit（句 / 行）
  2) 用字符 n-gram 余弦相似度做距离有界聚类（DBSCAN 式，阈值 tau）
  3) 检查每个簇：
     - 是否同时含 "答案跨度所在 unit" 与 "不含答案的 unit"   -> merge_hits_answer
     - 是否同时含 "敏感标识符" 与 "不含敏感标识符的 unit"     -> merge_mixes_sensitive
     - 是否同时含 两个不同敏感标识符的 unit                  -> merge_fuses_two_secrets（最危险）
  4) 报告比例，并按 tau / 来源分层

这是一个**下界**：字符 n-gram 比真嵌入保守，真嵌入聚类只会合得更多。
"""
import io, json, os, re, sys, math, hashlib, argparse
from collections import defaultdict, Counter

def build_idf(units):
    df = Counter()
    for u in units:
        for g in grams(u):
            df[g] += 1
    n = max(len(units), 1)
    return {g: math.log((n + 1) / (c + 1)) + 1.0 for g, c in df.items()}

def grams(s, k=3):
    s = re.sub(r'[^a-z0-9]', '', s.lower())
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i+k] for i in range(len(s) - k + 1)}

def vec(u, idf):
    g = grams(u)
    return {x: idf.get(x, 1.0) for x in g}

def cos(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb + 1e-12)

def split_units(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = [p.strip() for p in parts if len(p.strip()) >= 12]
    return out or [text[:400]]

def bounded_cluster(vecs, tau):
    n = len(vecs)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i in range(n):
        for j in range(i + 1, n):
            if cos(vecs[i], vecs[j]) >= tau:
                union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())

def exposure_hit(secret, text):
    norm = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
    n = norm(text)
    if norm(secret['value']) in n:
        return True
    m = norm(secret['marker']); k = secret['fragment_length']
    return any(m[i:i+k] in n for i in range(len(m)-k+1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--taus', default='0.5,0.65,0.8')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--maxtokens', type=int, default=400)
    a = ap.parse_args()

    recs = []
    with io.open(a.samples, encoding='utf-8') as fh:
        for line in fh:
            recs.append(json.loads(line))
    if a.limit:
        recs = recs[:a.limit]

    taus = [float(x) for x in a.taus.split(',')]
    res = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # tau -> source -> [n, hitAns, mixSens, fuseTwo]
    tot = defaultdict(lambda: [0, 0, 0, 0])

    for ri, r in enumerate(recs):
        units = split_units(r['context'])[:a.maxtokens]
        if len(units) < 4:
            continue
        idf = build_idf(units)
        vecs = [vec(u, idf) for u in units]

        # 答案跨度落在哪些 unit
        ans_units = set()
        for ans in r.get('answers') or []:
            if not ans:
                continue
            key = re.sub(r'[^a-z0-9]', '', str(ans).lower())
            if not key:
                continue
            for i, u in enumerate(units):
                if key in re.sub(r'[^a-z0-9]', '', u.lower()):
                    ans_units.add(i)

        # 每条 unit 命中的敏感标识符
        unit_sec = []
        for u in units:
            hit = {ci for ci, c in enumerate(r['canaries']) if exposure_hit(c, u)}
            unit_sec.append(hit)

        for tau in taus:
            clusters = bounded_cluster(vecs, tau)
            for cl in clusters:
                if len(cl) < 2:
                    continue
                src = r['source']
                # 1) 簇内混合了"含答案"与"不含答案"
                has_ans = any(i in ans_units for i in cl)
                no_ans = any(i not in ans_units for i in cl)
                hit_ans = 1 if (ans_units and has_ans and no_ans) else 0
                # 2) 簇内混合了 敏感 与 非敏感
                any_s = any(unit_sec[i] for i in cl)
                any_ns = any(not unit_sec[i] for i in cl)
                mix_s = 1 if (any_s and any_ns) else 0
                # 3) 簇内融合了两个不同敏感标识符
                fused = set()
                for i in cl:
                    fused |= unit_sec[i]
                fuse2 = 1 if len(fused) >= 2 else 0

                for key in (('__ALL__',), (src,)):
                    pass
                res[tau][src][0] += 1; res[tau][src][1] += hit_ans
                res[tau][src][2] += mix_s;  res[tau][src][3] += fuse2
                tot[(tau,)][0] += 1; tot[(tau,)][1] += hit_ans
                tot[(tau,)][2] += mix_s;  tot[(tau,)][3] += fuse2

    L = []
    w = L.append
    w("V2 FEASIBILITY: does distance-bounded clustering merge utility-critical or sensitive spans?")
    w("samples=%d  max_units/ctx=%d  taus=%s" % (len(recs), a.maxtokens, taus))
    w("metric = fraction of MERGED CLUSTERS (size>=2) that ...")
    w("")
    w("%-6s %-18s %8s %12s %12s %12s" % ("tau", "source", "clusters", "hit_answer", "mix_sens", "fuse_2sec"))
    for tau in taus:
        for src in sorted(res[tau]):
            n, ha, ms, f2 = res[tau][src]
            if n == 0: continue
            w("%-6.2f %-18s %8d %11.1f%% %11.1f%% %11.1f%%" % (tau, src, n, 100*ha/n, 100*ms/n, 100*f2/n))
        n, ha, ms, f2 = tot[(tau,)]
        w("%-6.2f %-18s %8d %11.1f%% %11.1f%% %11.1f%%" % (tau, "== ALL ==", n, 100*ha/n, 100*ms/n, 100*f2/n))
        w("")
    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    print("wrote", a.out)
    for tau in taus:
        n, ha, ms, f2 = tot[(tau,)]
        print("tau=%.2f clusters=%d hitAns=%.1f%% mixSens=%.1f%% fuse2=%.1f%%" % (tau, n, 100*ha/n, 100*ms/n, 100*f2/n))

if __name__ == '__main__':
    main()
