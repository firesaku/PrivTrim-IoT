"""
V2d（GPU 版）：真语义嵌入下的合并可行性 + d_sens 约束版
单卡。用本地 llmlingua-2-xlm-roberta-large 编码器做 mean-pool 句嵌入（离线，不下载）。

对比 5 个模式：
  sim_emb      : 真嵌入余弦 -> 贪心合并
  sim_char3    : 字符 3-gram TF-IDF（复算，与 HPC CPU 版对齐）
  cons_emb     : 真嵌入 + d_sens 硬约束（禁止敏感与非敏感 unit 合并）
  random_adj   : 随机相邻
  random_far   : 随机非相邻

指标同 V2c：hit_answer / mix_sens / fuse_2sec
"""
import io, json, os, re, sys, time, argparse, random
from collections import defaultdict, Counter
import numpy as np

def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- 文本处理 ----------------
def split_units(text, cap):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    out = [p.strip() for p in parts if len(p.strip()) >= 12]
    if not out:
        out = [text[:600]]
    return out[:cap]

def grams(s, k=3):
    s = re.sub(r'[^a-z0-9]', '', s.lower())
    if len(s) < k:
        return [s] if s else []
    return [s[i:i+k] for i in range(len(s) - k + 1)]

def char_matrix(units):
    docs = [grams(u) for u in units]
    vocab = {}
    for d in docs:
        for g in set(d):
            if g not in vocab:
                vocab[g] = len(vocab)
    n, V = len(units), max(len(vocab), 1)
    df = np.zeros(V); M = np.zeros((n, V))
    for i, d in enumerate(docs):
        c = Counter(d)
        for g, f in c.items():
            M[i, vocab[g]] = f
        for g in set(d):
            df[vocab[g]] += 1
    idf = np.log((n + 1) / (df + 1)) + 1.0
    M *= idf
    nn = np.linalg.norm(M, axis=1, keepdims=True); nn[nn == 0] = 1.0
    return M / nn

# ---------------- 聚类 ----------------
def pairs_similarity(S, target):
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-S[iu])
    out = []
    for k in order:
        if len(out) * 2 >= target:
            break
        out.append((int(iu[0][k]), int(iu[1][k])))
    return out

def pairs_constrained(S, target, unit_sec):
    """d_sens 硬约束：不允许把"命中敏感"的 unit 与"未命中"的 unit 合并。"""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-S[iu])
    out = []
    for k in order:
        if len(out) * 2 >= target:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        if bool(unit_sec[i]) != bool(unit_sec[j]):
            continue
        out.append((i, j))
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

def exposure_hit(secret, text):
    n = re.sub(r'[^a-z0-9]', '', text.lower())
    if re.sub(r'[^a-z0-9]', '', secret['value'].lower()) in n:
        return True
    m = re.sub(r'[^a-z0-9]', '', secret['marker'].lower())
    k = secret['fragment_length']
    return any(m[i:i+k] in n for i in range(len(m) - k + 1))

MODES = ('sim_emb', 'sim_char3', 'cons_emb', 'random_adj', 'random_far')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--model', required=True, help='本地编码器目录')
    ap.add_argument('--ratios', default='0.125,0.25,0.5')
    ap.add_argument('--cap', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--maxlen', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log("device", dev, "cuda_count", torch.cuda.device_count())
    if dev == 'cuda':
        log("gpu", torch.cuda.get_device_name(0))

    log("loading encoder:", a.model)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    try:
        enc = AutoModel.from_pretrained(a.model, local_files_only=True)
    except Exception as e:
        log("AutoModel failed (%r), trying XLMRobertaModel" % (e,))
        from transformers import XLMRobertaModel
        enc = XLMRobertaModel.from_pretrained(a.model, local_files_only=True)
    enc.eval().to(dev)

    @torch.no_grad()
    def embed(texts):
        outs = []
        for i in range(0, len(texts), a.batch):
            b = texts[i:i + a.batch]
            t = tok(b, padding=True, truncation=True, max_length=a.maxlen, return_tensors='pt').to(dev)
            h = enc(**t).last_hidden_state
            mask = t['attention_mask'].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v, dim=-1)
            outs.append(v.float().cpu().numpy())
        return np.vstack(outs)

    recs = [json.loads(l) for l in io.open(a.samples, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    ratios = [float(x) for x in a.ratios.split(',')]

    cells = defaultdict(lambda: [0, 0, 0, 0])
    n_used = 0
    t0 = time.time()
    for ri, r in enumerate(recs):
        units = split_units(r['context'], a.cap)
        if len(units) < 8:
            continue
        n_used += 1
        if n_used % 25 == 0:
            log("progress %d/%d  %.1fs" % (n_used, len(recs), time.time() - t0))

        E = embed(units)
        S = E @ E.T
        np.fill_diagonal(S, -1.0)
        Mc = char_matrix(units)
        Sc = Mc @ Mc.T
        np.fill_diagonal(Sc, -1.0)

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
                'sim_emb':    pairs_similarity(S, target),
                'sim_char3':  pairs_similarity(Sc, target),
                'cons_emb':   pairs_constrained(S, target, unit_sec),
                'random_adj': pairs_random(n, target, rng, min_gap=0),
                'random_far': pairs_random(n, target, rng, min_gap=2),
            }
            for mode in MODES:
                b = cells[(mode, ratio, r['source'])]
                for cl in make_clusters(n, plans[mode]):
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

    def agg(mode, ratio):
        t = [0, 0, 0, 0]
        for k, b in cells.items():
            if k[0] == mode and k[1] == ratio:
                for i in range(4):
                    t[i] += b[i]
        return t

    L = []
    w = L.append
    w("V2d GPU FEASIBILITY — real semantic embeddings vs char-3gram vs random")
    w("samples=%s records=%d cap=%d ratios=%s model=%s device=%s" % (
        os.path.basename(a.samples), n_used, a.cap, ratios, os.path.basename(str(a.model).rstrip('/')), dev))
    w("")
    w("%-12s %-7s %9s %11s %11s %11s" % ("mode", "ratio", "clusters", "hit_answer", "mix_sens", "fuse_2sec"))
    for mode in MODES:
        for ratio in ratios:
            t = agg(mode, ratio)
            if t[0]:
                w("%-12s %-7.3f %9d %10.1f%% %10.1f%% %10.1f%%" % (
                    mode, ratio, t[0], 100*t[1]/t[0], 100*t[2]/t[0], 100*t[3]/t[0]))
        w("")
    w("### VERDICT")
    for ratio in ratios:
        parts = []
        for mode in MODES:
            t = agg(mode, ratio)
            if t[0]:
                parts.append("%s: hit=%.1f%% mix=%.1f%% fuse=%.1f%%" % (
                    mode, 100*t[1]/t[0], 100*t[2]/t[0], 100*t[3]/t[0]))
        w("ratio=%.3f | %s" % (ratio, " || ".join(parts)))

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    io.open(a.json, 'w', encoding='utf-8').write(json.dumps(
        {"records_scored": n_used, "cap": a.cap, "ratios": ratios, "model": a.model, "device": dev,
         "cells": {"%s|%s|%s" % k: v for k, v in cells.items()}}, ensure_ascii=False, indent=1))
    log("wrote", a.out)
    print("\n".join(L))

if __name__ == '__main__':
    main()
