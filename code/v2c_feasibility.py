"""
V2c：距离有界 / 相似度贪心合并的可行性测试（HPC 版）
CPU only, 0 model calls, 只依赖 numpy。

相比 V2b 的三处加固：
  1) 随机对照拆成两个：random_adjacent（原版）与 random_far（非相邻，排除"相邻即跨边界"的伪影）
  2) 三个合并比例：0.125 / 0.25 / 0.50
  3) 每来源、每比例分层 + 输出 JSON 供后续合并统计

指标（每个 size>=2 的簇）：
  hit_answer : 簇内既有含答案跨度的 unit，又有不含的
  mix_sens   : 簇内既有命中敏感标识符的 unit，又有不含的
  fuse_2sec  : 簇内融合了两个不同敏感标识符（最危险）
"""
import io, json, os, re, argparse, random
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
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return M / nrm

def make_clusters(n, pairs):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    used = set()
    for i, j in pairs:
        if i in used or j in used:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
        used.add(i); used.add(j)
    g = defaultdict(list)
    for i in range(n):
        g[find(i)].append(i)
    return [v for v in g.values() if len(v) >= 2]

def pairs_similarity(S, target):
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    order = np.argsort(-sims)
    out = []
    for k in order:
        if len(out) * 2 >= target:
            break
        out.append((int(iu[0][k]), int(iu[1][k])))
    return out

def pairs_random(n, target, rng, min_gap=0):
    idx = list(range(n)); rng.shuffle(idx)
    out = []
    for k in range(0, n - 1, 2):
        i, j = idx[k], idx[k + 1]
        if abs(i - j) <= min_gap:
            continue
        out.append((i, j))
        if len(out) * 2 >= target:
            break
    return out

def exposure_hit(secret, text):
    n = re.sub(r'[^a-z0-9]', '', text.lower())
    if re.sub(r'[^a-z0-9]', '', secret['value'].lower()) in n:
        return True
    m = re.sub(r'[^a-z0-9]', '', secret['marker'].lower())
    k = secret['fragment_length']
    return any(m[i:i+k] in n for i in range(len(m) - k + 1))

MODES = ('similarity', 'random_adjacent', 'random_far')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--ratios', default='0.125,0.25,0.5')
    ap.add_argument('--cap', type=int, default=150)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    recs = [json.loads(l) for l in io.open(a.samples, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    ratios = [float(x) for x in a.ratios.split(',')]

    cells = defaultdict(lambda: [0, 0, 0, 0])          # (mode, ratio, source) -> [n, ha, ms, f2]
    n_used = 0
    for r in recs:
        units = split_units(r['context'], a.cap)
        if len(units) < 8:
            continue
        n_used += 1
        M = build_matrix(units)
        S = M @ M.T
        np.fill_diagonal(S, -1.0)

        key = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
        ans_units = set()
        for ans in (r.get('answers') or []):
            kk = key(str(ans))
            if not kk:
                continue
            for i, u in enumerate(units):
                if kk in key(u):
                    ans_units.add(i)
        unit_sec = [{ci for ci, c in enumerate(r['canaries']) if exposure_hit(c, u)} for u in units]

        n = len(units)
        for ratio in ratios:
            target = max(int(ratio * n), 2)
            plans = {
                'similarity':      pairs_similarity(S, target),
                'random_adjacent': pairs_random(n, target, rng, min_gap=0),
                'random_far':      pairs_random(n, target, rng, min_gap=2),
            }
            for mode in MODES:
                cl_list = make_clusters(n, plans[mode])
                b = cells[(mode, ratio, r['source'])]
                for cl in cl_list:
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
                    b[0] += 1; b[1] += ha; b[2] += ms; b[3] += f2

    L = []
    w = L.append
    w("V2c FEASIBILITY (HPC) — similarity greedy merge vs two random controls")
    w("samples_file=%s  records_scored=%d  cap_units=%d  ratios=%s  seed=%d" % (
        os.path.basename(a.samples), n_used, a.cap, ratios, a.seed))
    w("")
    w("%-16s %-8s %-18s %9s %11s %11s %11s" % ("mode", "ratio", "source", "clusters", "hit_answer", "mix_sens", "fuse_2sec"))
    for mode in MODES:
        for ratio in ratios:
            srcs = sorted({k[2] for k in cells if k[0] == mode and k[1] == ratio})
            tot = [0, 0, 0, 0]
            for src in srcs:
                b = cells[(mode, ratio, src)]
                for i in range(4):
                    tot[i] += b[i]
                if b[0]:
                    w("%-16s %-8.3f %-18s %9d %10.1f%% %10.1f%% %10.1f%%" % (
                        mode, ratio, src, b[0], 100*b[1]/b[0], 100*b[2]/b[0], 100*b[3]/b[0]))
            if tot[0]:
                w("%-16s %-8.3f %-18s %9d %10.1f%% %10.1f%% %10.1f%%" % (
                    mode, ratio, "== ALL ==", tot[0], 100*tot[1]/tot[0], 100*tot[2]/tot[0], 100*tot[3]/tot[0]))
            w("")
    w("### VERDICT")
    for ratio in ratios:
        def agg(mode):
            t = [0, 0, 0, 0]
            for k, b in cells.items():
                if k[0] == mode and k[1] == ratio:
                    for i in range(4):
                        t[i] += b[i]
            return t
        s, ra, rf = agg('similarity'), agg('random_adjacent'), agg('random_far')
        if not (s[0] and ra[0] and rf[0]):
            continue
        w("ratio=%.3f | hit_answer sim=%.1f%% adj=%.1f%% far=%.1f%% | mix_sens sim=%.1f%% adj=%.1f%% far=%.1f%% | fuse2 sim=%.1f%% adj=%.1f%% far=%.1f%%" % (
            ratio,
            100*s[1]/s[0], 100*ra[1]/ra[0], 100*rf[1]/rf[0],
            100*s[2]/s[0], 100*ra[2]/ra[0], 100*rf[2]/rf[0],
            100*s[3]/s[0], 100*ra[3]/ra[0], 100*rf[3]/rf[0]))

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    io.open(a.json, 'w', encoding='utf-8').write(json.dumps(
        {"records_scored": n_used, "cap": a.cap, "ratios": ratios, "seed": a.seed,
         "cells": {"%s|%s|%s" % k: v for k, v in cells.items()}}, ensure_ascii=False, indent=1))
    print("wrote", a.out, "and", a.json)
    for ratio in ratios:
        for mode in MODES:
            t = [0, 0, 0, 0]
            for k, b in cells.items():
                if k[0] == mode and k[1] == ratio:
                    for i in range(4):
                        t[i] += b[i]
            if t[0]:
                print("  %-16s r=%.3f clusters=%6d hitAns=%5.1f%% mixSens=%5.1f%% fuse2=%4.1f%%" % (
                    mode, ratio, t[0], 100*t[1]/t[0], 100*t[2]/t[0], 100*t[3]/t[0]))

if __name__ == '__main__':
    main()
