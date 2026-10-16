"""
V3：多配置实验批（单卡/任务，作业数组）
configs:
  scatter   : 分散式注入，测 d_sens 约束是否仍冗余
  guard     : 连续块注入，隔离 d_imp（重要性护栏）与 d_sens 各自贡献
  gran      : 粒度扫描（句 / 行 / 3 句窗口）
  bound     : 距离有界聚类（tau 扫描）+ 簇大小上限
每个配置独立输出报告与 cells.json

—— 审计修订（2026-10-16，findings #1–#3；旧数字必须仍可复现，故一律保留开关）——
#1 表示层：旧版 embed() 取 enc(**t).last_hidden_state（末层）。本项目声称本实验是 PACT 的
   距离有界聚类迁移，而 PACT 用早期层隐状态；两层的簇结构不同，所以旧数字是在「表示层与
   PACT 不一致」的前提下得到的，不能当作 PACT 迁移证据。现默认改为早期层
   max(1, n_layers // 6)（--layer early，层数由 model.config 推出，随编码器自适应；
   层号按 hidden_states 下标计：0 = 词嵌入输出，k = 第 k 层输出，n_layers-1 = 末层）。
   --layer last 仍走 last_hidden_state 且不请求 hidden_states（调用参数与旧实现逐字相同），
   用于逐位复现已发布数字。
#2 随机对照：旧 rand4_emb 用 pairs_random（只连 idx[k] 与 idx[k+1]，簇结构恒为 size 2），
   而 bound_emb 允许 max_size=4，于是「选择准则」与「簇大小」两个因素被混在一起，
   无法支撑「是距离界而不是簇太小压住了 mixing」这一归因。现默认改用
   clusters_random_sized（打乱后贪心地把至多 max_size 个未使用 unit 组成一簇，
   合并记账与 pairs_similarity 完全一致 → 两臂请求的预算相同，唯一差别是选择准则）；
   --rand-legacy 恢复旧 size-2 配对，用于复现已发布数字。
#3 可审计性：tau / max_size / 编码器名 / 所选层 全部写入报告头与 JSON（旧版只有
   config/samples/granularity/records/cap/seed/dev/modes/ratios，表格无法复核）。
   同时明确 tau 只约束合并边（edge-level，见 pairs_similarity 注释）：链式合并可以让
   相似度低于 tau 的两个 unit 进入同一簇，因此 tau 不是簇直径上界，
   「同簇成员两两相似度 >= tau」这类表述是错的。
#4 合并匹配（2026-10-16 第二轮修正）：上一轮只匹配了簇大小上限，没匹配**合并次数**。GPU 实测
   bound 配置 bound_emb 1370 簇 / rand4_emb 888 簇 —— 随机臂合并得更多，因为 tau 本身是约束
   （找不到 sim >= tau 的边就提前停），于是「混得少」可能只是「合并得少」，与已修的
   pairs_similarity 欠合并问题同类。现默认让随机臂逐记录×ratio 停在与 bound 臂**完全相同的
   union 数**上：bound 臂经 stats 回报真实 union 数，随机臂用 stop_unions= 精确对齐（两侧都数
   真实 union，不用 target 估算）。--rand-stop matched（默认）/ budget（上一版：按 target 预算，
   簇大小匹配但合并次数可能多于 bound 臂）；--rand-legacy 仍是已发布数字的复现路径。
   报告增 unions 列（--rand-legacy 下不加该列，以保持已发布表格逐字不变）、每臂 clusters/unions
   汇总行、merge_parity 行（逐记录×ratio 的 union 相等计数；不等则逐条列出，不平均掉），
   JSON 增 arm_unions / merge_parity / merge_parity_mismatch / rand_stop / rand_min_gap。
—— 复现已发布数字：--layer last --rand-legacy（其余参数不变）
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
def pairs_similarity(S, target, rules=None, max_size=2, tau=None, stats=None):
    """rules: list of boolean arrays; 若任一 rule 上 i,j 取值不同则禁止合并。
       max_size: 簇大小上限（2 = 纯两两配对）; tau: 相似度下界。
       target: 合并预算；每接受一次 union 记 2（merged += 2），merged >= target 即停
       （故未受限时 union 次数 = ceil(target/2)；tau/上限受限时会更少）。
       stats: 可选 dict，回填真实计数 {'unions': 接受的 union 数, 'clusters', 'units_used', 'budget'}；
              审计 #4 的合并匹配就是靠它把 bound 臂的真实 union 数传给随机臂。

       tau 语义（重要，勿过度解读）：tau 只在「当前这条合并边」上检查
       （sims[k] < tau 即 break；order 按相似度降序，所以 break 不会漏掉更高的边）。
       它保证每条被采用的边 sim >= tau，但**不保证簇内任意两成员** sim >= tau：
       A-B、B-C 两条边各自 >= tau 时，链式合并会把 A、B、C 放进同一簇，
       而 sim(A,C) 可以远低于 tau（见 verify/test_v3_controls.py 的构造用例）。
       即 tau 是 edge-level（单条合并边）保证，不是簇直径 / 全对相似度下界。
       任何「同簇成员两两相似度 >= tau」的说法都是错的。"""
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
    cl = [v for v in g.values() if len(v) >= 2]
    if stats is not None:
        # 审计 #4：回报**真实**接受的 union 数（每次 union 记 2 → unions = merged // 2），
        # 供合并匹配的随机臂与报告/JSON 使用；不是用 target 估算出来的值
        stats['unions'] = merged // 2
        stats['clusters'] = len(cl)
        stats['units_used'] = sum(len(c) for c in cl)
        stats['budget'] = target
    return cl

def pairs_random(n, target, rng, min_gap=0):
    # 注意：只连 idx[k] 与 idx[k+1]，配出的对互斥 → 簇结构恒为 size 2。
    # scatter/guard/gran 的 random_adj / random_far 走这里；
    # bound 配置的 rand4_emb 旧行为也走这里，需 --rand-legacy 才回到此路径。
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

def clusters_random_sized(n, target, rng, max_size=4, min_gap=0, stop_unions=None, stats=None):
    """大小匹配 + 合并匹配的随机对照（bound 配置的 rand4_emb 默认走这里，审计 #2/#4）：
       打乱 unit 顺序后，贪心地把至多 max_size 个「尚未被前面各簇取走」的 unit 组成一簇。
       合并记账与 pairs_similarity 完全一致：每并入一个 unit 视为一次 union，merged += 2；
       inner/outer 都在每次 union **之前**检查停止条件，故既不会少做也不会超做。
       停止条件：stop_unions 给出时恰好做 stop_unions 次 union（审计 #4 合并匹配：用 bound 臂
       同一记录同一个 ratio 的真实 union 数）；否则按 target 预算（merged >= target 即停，
       未受限时 union 次数 = ceil(target/2)，即上一版行为）。
       stats: 可选 dict，回填真实计数 {'unions', 'clusters', 'units_used', 'stop'}。
       min_gap>0 时不并入与簇内已有 unit 原始下标相距 <= min_gap 的 unit（CLI 默认 --rand-min-gap 2，
       与旧 random_far 同口径；函数默认 0 = 不限制，便于单测与敏感性对照）。"""
    limit = 2 * int(stop_unions) if stop_unions is not None else int(target)
    order = list(range(n)); rng.shuffle(order)
    out = []; merged = 0; k = 0
    while merged < limit and k < n:
        grp = [order[k]]; k += 1
        while len(grp) < max_size and merged < limit and k < n:
            i = order[k]; k += 1
            if min_gap and any(abs(i - j) <= min_gap for j in grp):
                continue
            grp.append(i); merged += 2
        if len(grp) >= 2:
            out.append(grp)
    if stats is not None:
        stats['unions'] = merged // 2
        stats['clusters'] = len(out)
        stats['units_used'] = sum(len(g) for g in out)
        stats['stop'] = ('stop_unions=%d' % stop_unions) if stop_unions is not None else ('target=%d' % target)
    return out

def unions_of(cl):
    """簇列表对应的真实 union 次数（森林恒等式：Σ(|c|-1)）。pairs_similarity 与
       clusters_random_sized 用自己的内部计数器（stats['unions']），这里只给
       clusters_from_pairs/pairs_random 那条旧随机路径兜底；单测已交叉验证两者相等。"""
    return sum(len(c) - 1 for c in cl)

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

# ---------------- 表示层选择（纯逻辑，可脱离模型单测，审计 #1） ----------------
def n_layers_of(model, default=12):
    """从 model.config 读编码器层数，兼容 num_hidden_layers / n_layers / num_layers。"""
    c = getattr(model, 'config', None)
    for k in ('num_hidden_layers', 'n_layers', 'num_layers'):
        v = getattr(c, k, None)
        if isinstance(v, int) and v > 0:
            return v
    return default

def resolve_layer(n_layers, layer='early', layer_idx=None):
    """返回 (层下标, 来源)。层号按 hidden_states 计：0 = 词嵌入输出，k = 第 k 层输出，
       n_layers-1 = 末层。'last' = 末层（--layer last，复现已发布数字）；
       'early' = max(1, n_layers // 6)（默认，贴近 PACT 的早期层，随编码器自适应）；
       显式 layer_idx 优先（与 --layer 冲突时以它为准），并截断到 [0, n_layers-1]。"""
    if layer_idx is not None:
        return max(0, min(int(layer_idx), n_layers - 1)), 'idx'
    if layer == 'last':
        return n_layers - 1, 'last'
    return min(max(1, n_layers // 6), n_layers - 1), 'early'

def pick_hidden_state(out, layer_idx, n_layers):
    """按层取隐状态：末层直接返回 last_hidden_state（与旧 embed() 取值逐位一致），
       其余取 hidden_states[layer_idx]（调用方需已传 output_hidden_states=True）。"""
    if layer_idx >= n_layers - 1:
        return out.last_hidden_state
    return out.hidden_states[layer_idx]

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
    # max_size: 簇大小上限；tau: 合并边相似度下界，仅 bound_emb 使用（旧版是 dispatch 里的字面量，
    # 表格因此无法复核 —— 审计 #3）。tau 为 edge-level 保证，见 pairs_similarity 注释：
    # 链式合并后同簇内可能存在相似度 < tau 的成员对，故 tau 不是簇直径上界。
    # rand4_emb 与之共用 max_size，两臂只在「按相似度选」还是「随机选」上不同（选择准则 = 唯一变量）。
    return dict(samples='frozen_samples_scattered.jsonl', granularity='sentence',
                ratios=[0.25], modes=['sim_emb', 'bound_emb', 'rand4_emb'],
                max_size=4, tau=0.75)

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
    ap.add_argument('--layer', choices=['early', 'last'], default='early',
                    help="表示层：early = max(1, n_layers//6) 早期层（默认，PACT 口径）；"
                         "last = 末层（旧行为，复现已发布数字）")
    ap.add_argument('--layer-idx', type=int, default=None,
                    help="显式层下标（0 = 词嵌入输出，n_layers-1 = 末层）；给出时优先于 --layer")
    ap.add_argument('--rand-legacy', action='store_true',
                    help="rand4_emb 用旧的 size-2 随机配对（复现已发布数字；此时 --rand-stop/"
                         "--rand-min-gap 均不生效，报告表格式也冻结为已发布列集合）")
    ap.add_argument('--rand-stop', choices=['matched', 'budget'], default='matched',
                    help="rand4_emb 的停止条件（审计 #4）：matched = 与该记录、该 ratio 下 "
                         "bound_emb 的**真实 union 数**精确对齐（默认；两侧都数真实 union，"
                         "不用 target 估算，消除「合并次数」混淆）；budget = 上一版行为"
                         "（按 target 预算停，簇大小匹配但合并次数可能多于 bound 臂）")
    ap.add_argument('--rand-min-gap', type=int, default=2,
                    help="clusters_random_sized 的最小索引间隔：2 = 沿用旧 random_far 的结构"
                         "（不配相邻 unit，偏保守）；0 = 允许相邻配对。"
                         "两个值都跑并都报，不用默认值去挑对自己有利的结果")
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

    # 表示层（审计 #1）：默认早期层（PACT 口径），--layer last 复现旧末层路径
    n_layers = n_layers_of(enc)
    if a.layer_idx is not None and a.layer == 'last':
        log("warn: --layer last 与 --layer-idx 同时给出，采用 --layer-idx=%d" % a.layer_idx)
    layer_idx, layer_src = resolve_layer(n_layers, a.layer, a.layer_idx)
    need_hs = (layer_idx < n_layers - 1)   # 末层不需要 hidden_states，保持旧调用参数不变
    log("model", a.model, "n_layers", n_layers, "layer_idx", layer_idx, "layer_src", layer_src)

    @torch.no_grad()
    def embed(texts):
        outs = []
        for i in range(0, len(texts), a.batch):
            b = texts[i:i + a.batch]
            t = tok(b, padding=True, truncation=True, max_length=a.maxlen, return_tensors='pt').to(dev)
            if need_hs:
                o = enc(**t, output_hidden_states=True)
            else:
                o = enc(**t)               # 旧路径：不请求 hidden_states，逐位复现已发布数字
            h = pick_hidden_state(o, layer_idx, n_layers)
            mask = t['attention_mask'].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            outs.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
        return np.vstack(outs)

    sp = os.path.join(a.datadir, C['samples'])
    recs = [json.loads(l) for l in io.open(sp, encoding='utf-8') if l.strip()]
    cells = defaultdict(lambda: [0, 0, 0, 0])
    arm_u = defaultdict(int)          # 审计 #4：每臂每 ratio 的真实 union 数（跨记录累加）
    parity_n = parity_eq = 0          # 合并配平检查：逐记录×ratio 比较 rand4_emb 与 bound_emb
    mismatch = []
    n_used = 0
    t0 = time.time()
    for line_idx, r in enumerate(recs):
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
            # 审计 #4：合并匹配需要 bound 臂在**同一条记录、同一个 ratio** 下的真实 union 数。
            # 与 modes 的排列顺序无关地先算一次并缓存，循环里命中 bound_emb 时复用（不重复计算）。
            bound_ref = None
            if (not a.rand_legacy) and 'bound_emb' in C['modes'] and 'rand4_emb' in C['modes']:
                _bst = {}
                bound_ref = (pairs_similarity(S, target, None, C.get('max_size', 4),
                                              tau=C.get('tau', 0.75), stats=_bst), _bst)
            for mode in C['modes']:
                st = None
                if mode == 'sim_emb':
                    st = {}; cl = pairs_similarity(S, target, None, 2, stats=st)
                elif mode == 'cons_emb':
                    st = {}; cl = pairs_similarity(S, target, [sens_flag], 2, stats=st)
                elif mode == 'imp_emb':
                    st = {}; cl = pairs_similarity(S, target, [ans_flag], 2, stats=st)
                elif mode == 'cons_imp_emb':
                    st = {}; cl = pairs_similarity(S, target, [sens_flag, ans_flag], 2, stats=st)
                elif mode == 'bound_emb':
                    # tau / max_size 取自配置（见 cfg_bound），并写入报告头与 JSON（审计 #3）；
                    # tau 是 edge-level 下界（见 pairs_similarity 注释），不是簇直径上界
                    if bound_ref is not None:
                        cl, st = bound_ref
                    else:
                        st = {}
                        cl = pairs_similarity(S, target, None, C.get('max_size', 4),
                                              tau=C.get('tau', 0.75), stats=st)
                elif mode == 'rand4_emb':
                    if a.rand_legacy:
                        # 旧行为：size-2 随机配对（仅用于复现已发布数字，无 union 计数器）
                        cl = clusters_from_pairs(n, pairs_random(n, target, rng, min_gap=2))
                    else:
                        # 审计 #2/#4：同 max_size、同 min_gap；matched 模式再用 bound 臂的真实
                        # union 数收口，使两臂请求/消耗的合并数完全一致（选择准则 = 唯一变量）
                        st = {}
                        cl = clusters_random_sized(n, target, rng, C.get('max_size', 4),
                                                   min_gap=a.rand_min_gap,
                                                   stop_unions=(bound_ref[1]['unions']
                                                                if (bound_ref and a.rand_stop == 'matched') else None),
                                                   stats=st)
                else:
                    cl = clusters_from_pairs(n, pairs_random(n, target, rng,
                                                             min_gap=(2 if mode == 'random_far' else 0)))
                # 真实 union 数：pairs_similarity / clusters_random_sized 用内部计数器，
                # random_adj/random_far（旧路径）用森林恒等式兜底
                u = st['unions'] if (st is not None and 'unions' in st) else unions_of(cl)
                arm_u[(mode, ratio)] += u
                # 合并配平：逐记录×ratio 记录两臂 union 是否相等；不等则逐条留痕，不平均掉
                if mode == 'rand4_emb' and bound_ref is not None:
                    parity_n += 1
                    if u == bound_ref[1]['unions']:
                        parity_eq += 1
                    else:
                        mismatch.append({'line': line_idx, 'used': n_used, 'ratio': ratio,
                                         'bound_unions': bound_ref[1]['unions'], 'rand_unions': u})
                        log("merge-parity mismatch line=%d used#%d ratio=%.3f bound_unions=%d rand_unions=%d"
                            % (line_idx, n_used, ratio, bound_ref[1]['unions'], u))
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
    rand_mode = ("legacy_size2" if a.rand_legacy else
                 ("sized_matched_unions" if a.rand_stop == 'matched' else "sized_target_budget"))
    w("V3 CONFIG=%s  samples=%s  granularity=%s  records=%d  cap=%d  seed=%d  dev=%s" % (
        a.config, C['samples'], C['granularity'], n_used, a.cap, a.seed, dev))
    # 审计 #3：tau / max_size / 编码器 / 所选层必须随结果落盘，否则表格无法复核
    # max_size 取配置值；非 bound 配置各臂都是两两配对，故回落值 2 即其真实簇大小上限
    w("model=%s  layer_idx=%d  n_layers=%d  layer=%s  tau=%s  max_size=%s  rand=%s  rand_stop=%s  rand_min_gap=%s" % (
        a.model, layer_idx, n_layers, layer_src,
        ("%.3f" % C['tau']) if C.get('tau') is not None else "-",
        C.get('max_size', 2), rand_mode, a.rand_stop, a.rand_min_gap))
    w("note: tau 只约束合并边(edge-level)，链式合并可使同簇成员对的相似度 < tau，故 tau 不是簇直径上界；tau=- 表示该配置不设下界")
    w("note: max_size 为该配置声明的簇大小上限（bound=4，作用于 bound_emb/rand4_emb；sim_emb 等两两配对臂恒为 2；其余配置也全是两两配对）")
    # 审计 #4：每臂的簇数与真实 union 数必须落盘；合并配平检查逐记录×ratio 报告，不等要逐条可见
    for m in C['modes']:
        w("arm %-14s %s" % (m, "  ".join("ratio=%.3f clusters=%d unions=%d" % (
            r, cells[(m, r)][0], arm_u[(m, r)]) for r in C['ratios'])))
    if parity_n:
        w("merge_parity(rand4_emb vs bound_emb, 逐记录×ratio): equal=%d/%d (%.1f%%)  unequal=%d  rand_stop=%s  rand_min_gap=%s"
          % (parity_eq, parity_n, 100.0 * parity_eq / parity_n, parity_n - parity_eq,
             a.rand_stop, a.rand_min_gap))
        for mm in mismatch[:10]:
            w("  mismatch: line=%d used#%d ratio=%.3f bound_unions=%d rand_unions=%d"
              % (mm['line'], mm['used'], mm['ratio'], mm['bound_unions'], mm['rand_unions']))
        if len(mismatch) > 10:
            w("  ...(+%d more，全量见 JSON merge_parity_mismatch)" % (len(mismatch) - 10))
    else:
        w("merge_parity: n/a%s" % ("（--rand-legacy：冻结的已发布行为，不对齐合并数）" if a.rand_legacy
                                   else "（本配置没有 bound_emb/rand4_emb 对照）"))
    w("")
    # --rand-legacy 冻结已发布表格的列集合；unions 列只在修正后的模式里加（数值仍在 arm 行与 JSON）
    hdr = "%-14s %-8s %9s %11s %11s %11s"; row = "%-14s %-8.3f %9d %10.1f%% %10.1f%% %10.1f%%"
    cols = ("mode", "ratio", "clusters", "hit_answer", "mix_sens", "fuse_2sec")
    if not a.rand_legacy:
        hdr += " %9s"; row += " %9d"; cols = cols + ("unions",)
    w(hdr % cols)
    for mode in C['modes']:
        for ratio in C['ratios']:
            b = cells[(mode, ratio)]
            if b[0]:
                vals = (mode, ratio, b[0], 100*b[1]/b[0], 100*b[2]/b[0], 100*b[3]/b[0])
                w(row % (vals if a.rand_legacy else vals + (arm_u[(mode, ratio)],)))
        w("")
    outp = os.path.join(a.outdir, 'v3_%s_report.txt' % a.config)
    outj = os.path.join(a.outdir, 'v3_%s_cells.json' % a.config)
    io.open(outp, 'w', encoding='utf-8').write("\n".join(L))
    io.open(outj, 'w', encoding='utf-8').write(json.dumps(
        {"config": a.config, "granularity": C['granularity'], "records": n_used,
         "cap": a.cap, "modes": C['modes'], "ratios": C['ratios'],
         "model": a.model, "layer_idx": layer_idx, "n_layers": n_layers, "layer": layer_src,
         "tau": C.get('tau'), "max_size": C.get('max_size', 2),
         "rand_mode": rand_mode, "rand_stop": a.rand_stop, "rand_min_gap": a.rand_min_gap,
         "tau_semantics": "edge-level：tau 只在下一条合并边上检查，链式合并可使同簇成员对的相似度 < tau；tau 不是簇直径上界（tau=null 表示该配置不设下界）",
         "max_size_semantics": "配置声明的簇大小上限：bound=4（作用于 bound_emb/rand4_emb；sim_emb 等两两配对臂恒为 2）；其余配置也全是两两配对=2",
         "arm_unions": {"%s|%s" % k: v for k, v in arm_u.items()},
         "merge_parity": {"unit": "record×ratio", "mode_pair": ["rand4_emb", "bound_emb"],
                          "total": parity_n, "equal": parity_eq, "unequal": parity_n - parity_eq,
                          "equal_pct": (100.0 * parity_eq / parity_n) if parity_n else None,
                          "rand_stop": a.rand_stop, "rand_min_gap": a.rand_min_gap,
                          "note": "matched 模式下应 100% 相等；unequal>0 时逐条见 merge_parity_mismatch"},
         "merge_parity_mismatch": mismatch,
         "cells": {"%s|%s" % k: v for k, v in cells.items()}}, ensure_ascii=False, indent=1))
    log("wrote", outp)
    print("\n".join(L))

if __name__ == '__main__':
    main()
