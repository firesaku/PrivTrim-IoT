"""
V3：多配置实验批（单卡/任务，作业数组）
configs:
  scatter   : 分散式注入，测 d_sens 约束是否仍冗余
  guard     : 连续块注入，隔离 d_imp（重要性护栏）与 d_sens 各自贡献
  gran      : 粒度扫描（句 / 行 / 3 句窗口）
  bound     : 距离有界聚类（tau 扫描）+ 簇大小上限
每个配置独立输出报告与 cells.json
"""
import io, json, os, re, sys, time, argparse, random
from collections import defaultdict, Counter
import numpy as np

def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- unit 切分 ----------------
def split_units(text, cap, gran="sentence"):
    if gran == "sentence":
        parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    elif gran == "line":
        parts = text.split("\n")
    elif gran == "win3":
        sents = re.split(r'(?<=[.!?])\s+|\n+', text)
        parts = [" ".join(sents[i:i+3]) for i in range(0, len(sents), 3)]
    elif gran == "charwin":
        # 固定字符窗：不依赖换行（本语料在抽样时已把空白归一化）
        W = 200
        parts = [text[i:i + W] for i in range(0, len(text), W)]
    else:
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

# ---------------- 配对策略 ----------------
def pairs_similarity(S, target, rules=None, max_size=2, tau=None):
    """rules: list of boolean arrays; 若任一 rule 上 i,j 取值不同则禁止合并。
       max_size: 簇大小上限（2 = 纯两两配对）; tau: 相似度下界。"""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    order = np.argsort(-sims)
    parent = list(range(n)); size = [1] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    up = np.asarray(rules) if rules else None
    merged = 0
    for k in order:
        if merged >= target:
            break
        if tau is not None and sims[k] < tau:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if size[ri] + size[rj] > max_size:
            continue
        if up is not None and np.any(up[:, i] != up[:, j]):
            continue
        parent[rj] = ri
        size[ri] += size[rj]
        merged += 2
    g = defaultdict(list)
    for i in range(n):
        g[find(i)].append(i)
    return [v for v in g.values() if len(v) >= 2]

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

def clusters_from_pairs(n, pairs):
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

# ---------------- 配置 ----------------
def cfg_scatter():
    return dict(samples='frozen_samples_scattered.jsonl', granularity='sentence',
                ratios=[0.125, 0.25, 0.5],
                modes=['sim_emb', 'cons_emb', 'imp_emb', 'cons_imp_emb', 'random_adj', 'random_far'])

def cfg_guard():
    return dict(samples='frozen_samples.jsonl', granularity='sentence',
                ratios=[0.125, 0.25, 0.5],
                modes=['sim_emb', 'cons_emb', 'imp_emb', 'cons_imp_emb', 'random_adj'])

def cfg_gran():
    # 与 guard 配置同样本、同模式子集，唯一差别是 unit 粒度：
    # guard = sentence，本配置 = charwin(200 字符窗)
    return dict(samples='frozen_samples.jsonl', granularity='charwin',
                ratios=[0.125, 0.25, 0.5],
                modes=['sim_emb', 'imp_emb', 'cons_imp_emb', 'random_adj'])

def cfg_bound():
    return dict(samples='frozen_samples_scattered.jsonl', granularity='sentence',
                ratios=[0.25], modes=['sim_emb', 'bound_emb', 'rand4_emb'],
                max_size=4)

CONFIGS = {'scatter': cfg_scatter, 'guard': cfg_guard, 'gran': cfg_gran, 'bound': cfg_bound}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=sorted(CONFIGS))
    ap.add_argument('--datadir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--cap', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--maxlen', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20261015)
    a = ap.parse_args()
    C = CONFIGS[a.config]()
    rng = random.Random(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log("config", a.config, "device", dev, "modes", C['modes'], "ratios", C['ratios'])
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    try:
        enc = AutoModel.from_pretrained(a.model, local_files_only=True)
    except Exception:
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
            outs.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.vstack(outs)

    sp = os.path.join(a.datadir, C['samples'])
    recs = [json.loads(l) for l in io.open(sp, encoding='utf-8') if l.strip()]
    cells = defaultdict(lambda: [0, 0, 0, 0])
    n_used = 0
    t0 = time.time()
    for r in recs:
        units = split_units(r['context'], a.cap, C['granularity'])
        if len(units) < 8:
            continue
        n_used += 1
        if n_used % 50 == 0:
            log("progress %d  %.1fs" % (n_used, time.time() - t0))
        E = embed(units)
        S = E @ E.T
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
        sens_flag = np.array([bool(unit_sec[i]) for i in range(n)])
        ans_flag = np.array([i in ans_units for i in range(n)])
        for ratio in C['ratios']:
            target = max(int(ratio * n), 2)
            for mode in C['modes']:
                if mode == 'sim_emb':
                    cl = pairs_similarity(S, target, None, 2)
                elif mode == 'cons_emb':
                    cl = pairs_similarity(S, target, [sens_flag], 2)
                elif mode == 'imp_emb':
                    cl = pairs_similarity(S, target, [ans_flag], 2)
                elif mode == 'cons_imp_emb':
                    cl = pairs_similarity(S, target, [sens_flag, ans_flag], 2)
                elif mode == 'bound_emb':
                    cl = pairs_similarity(S, target, None, C.get('max_size', 4), tau=0.75)
                elif mode == 'rand4_emb':
                    cl = clusters_from_pairs(n, pairs_random(n, target, rng, min_gap=2))
                else:
                    cl = clusters_from_pairs(n, pairs_random(n, target, rng,
                                                             min_gap=(2 if mode == 'random_far' else 0)))
                b = cells[(mode, ratio)]
                for cu in cl:
                    has_ans = any(i in ans_units for i in cu)
                    no_ans = any(i not in ans_units for i in cu)
                    ha = 1 if (ans_units and has_ans and no_ans) else 0
                    any_s = any(unit_sec[i] for i in cu)
                    any_ns = any(not unit_sec[i] for i in cu)
                    ms = 1 if (any_s and any_ns) else 0
                    fused = set()
                    for i in cu:
                        fused |= unit_sec[i]
                    f2 = 1 if len(fused) >= 2 else 0
                    b[0] += 1; b[1] += ha; b[2] += ms; b[3] += f2

    L = []
    w = L.append
    w("V3 CONFIG=%s  samples=%s  granularity=%s  records=%d  cap=%d  seed=%d  dev=%s" % (
        a.config, C['samples'], C['granularity'], n_used, a.cap, a.seed, dev))
    w("")
    w("%-14s %-8s %9s %11s %11s %11s" % ("mode", "ratio", "clusters", "hit_answer", "mix_sens", "fuse_2sec"))
    for mode in C['modes']:
        for ratio in C['ratios']:
            b = cells[(mode, ratio)]
            if b[0]:
                w("%-14s %-8.3f %9d %10.1f%% %10.1f%% %10.1f%%" % (
                    mode, ratio, b[0], 100*b[1]/b[0], 100*b[2]/b[0], 100*b[3]/b[0]))
        w("")
    outp = os.path.join(a.outdir, 'v3_%s_report.txt' % a.config)
    outj = os.path.join(a.outdir, 'v3_%s_cells.json' % a.config)
    io.open(outp, 'w', encoding='utf-8').write("\n".join(L))
    io.open(outj, 'w', encoding='utf-8').write(json.dumps(
        {"config": a.config, "granularity": C['granularity'], "records": n_used,
         "cap": a.cap, "modes": C['modes'], "ratios": C['ratios'],
         "cells": {"%s|%s" % k: v for k, v in cells.items()}}, ensure_ascii=False, indent=1))
    log("wrote", outp)
    print("\n".join(L))

if __name__ == '__main__':
    main()
