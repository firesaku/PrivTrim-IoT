r"""v3_batch.py 审计修订（#1 层选择 / #2 大小匹配随机对照 / #3 tau 语义）的本地单测。
纯 numpy，不下载也不加载任何模型；被 import 的 v3_batch 模块级只依赖 numpy。

运行（Windows PowerShell）：
  $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
  python D:\CodexData\tmp\privtrim-review-20261015\verify\test_v3_controls.py
断言失败 -> 非零退出码。
"""
import importlib.util, math, os, random, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3PATH = os.path.join(os.path.dirname(HERE), 'hpc', 'v3_batch.py')
spec = importlib.util.spec_from_file_location('v3_batch_under_test', V3PATH)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
SRC = open(V3PATH, encoding='utf-8').read()

n_ok = 0
def check(name, cond, extra=''):
    global n_ok
    if not cond:
        print('  FAIL %s %s' % (name, extra)); sys.exit(1)
    n_ok += 1
    print('  ok   %-52s %s' % (name, extra))

def merges(cl):
    """union 次数。pairs_similarity 只做无环 union（ri==rj 跳过）→ Σ(|c|-1) = union 数。"""
    return sum(len(c) - 1 for c in cl)

# ---------------- #1 层选择 ----------------
print('#1 层选择 resolve_layer / pick_hidden_state / n_layers_of')
check('early(24层, XLM-R large 默认)', v3.resolve_layer(24) == (4, 'early'), v3.resolve_layer(24))
check('last(24层) = 末层下标 23', v3.resolve_layer(24, 'last') == (23, 'last'))
check('显式 layer_idx 优先', v3.resolve_layer(24, 'early', 7) == (7, 'idx'))
check('layer_idx 与 last 冲突时取 idx', v3.resolve_layer(24, 'last', 7) == (7, 'idx'))
check('越界截断到 n_layers-1', v3.resolve_layer(24, 'early', 999) == (23, 'idx'))
check('负值截断到 0', v3.resolve_layer(24, 'early', -5) == (0, 'idx'))
check('1 层模型两个模式都落到 0', v3.resolve_layer(1, 'early') == (0, 'early') and v3.resolve_layer(1, 'last') == (0, 'last'))
check('6 层 -> max(1, 1) = 1', v3.resolve_layer(6) == (1, 'early'), v3.resolve_layer(6))
check('12 层 -> 2（随编码器自适应）', v3.resolve_layer(12) == (2, 'early'), v3.resolve_layer(12))

class _Cfg:
    def __init__(self, **kw): self.__dict__.update(kw)
class _Model:
    """stub：只提供 .config（n_layers_of 的真实入参是 encoder 对象）"""
    def __init__(self, cfg): self.config = cfg
class _Out:
    """stub：模拟 encoder 输出（hidden_states 长度 = n_layers+1，最后一个 == last_hidden_state）"""
    def __init__(self, n_layers, D=3, seed=0):
        rng = np.random.RandomState(seed)
        self.hidden_states = tuple(rng.randn(D) for _ in range(n_layers + 1))
        self.last_hidden_state = self.hidden_states[-1]
check('n_layers_of: num_hidden_layers', v3.n_layers_of(_Model(_Cfg(num_hidden_layers=24))) == 24)
check('n_layers_of: n_layers 别名', v3.n_layers_of(_Model(_Cfg(n_layers=12))) == 12)
check('n_layers_of: 读不到时回落 default',
      v3.n_layers_of(_Model(_Cfg(foo=1)), default=12) == 12
      and v3.n_layers_of(None, default=12) == 12)

o = _Out(24)
check('pick: 末层取 last_hidden_state（同一对象）',
      v3.pick_hidden_state(o, 23, 24) is o.last_hidden_state)
check('pick: 早层取 hidden_states[idx]（同一对象）',
      v3.pick_hidden_state(o, 4, 24) is o.hidden_states[4])
check('pick: hidden_states[-1] 与 last_hidden_state 数值一致',
      bool(np.array_equal(o.hidden_states[24], o.last_hidden_state)))
# 源码级守卫：末层分支必须仍是旧的 enc(**t) 调用（--layer last 复现已发布数字的前提）
has_plain = re.search(r'^\s*o = enc\(\*\*t\)\s*(#.*)?$', SRC, re.M) is not None
has_hs = 'o = enc(**t, output_hidden_states=True)' in SRC
check('源码守卫: 末层走 enc(**t)，早层才请求 hidden_states', has_plain and has_hs,
      'plain=%s hidden_states=%s' % (has_plain, has_hs))

# ---------------- #2 大小匹配随机对照 ----------------
print('#2 clusters_random_sized：可达 size=max_size，且与 bound 臂合并预算一致')
COMBOS = [(100, 25, 4), (64, 64, 4), (37, 9, 4), (50, 2, 4), (80, 17, 3), (80, 40, 2)]
all_ok = True
for n, target, ms in COMBOS:
    # 全同高相似度 -> bound 臂不受 tau / cap 限制，是「预算」口径的干净对照
    S = np.full((n, n), 0.99); np.fill_diagonal(S, -1.0)
    cl_b = v3.pairs_similarity(S, target, None, ms, tau=0.75)
    cl_r = v3.clusters_random_sized(n, target, random.Random(20261015), ms)
    mb, mr, want = merges(cl_b), merges(cl_r), math.ceil(target / 2)
    flat = [i for g in cl_r for i in g]
    same = (mb == mr == want) and len(flat) == len(set(flat)) and all(len(g) <= ms for g in cl_r)
    all_ok &= same
    print('       n=%-4d target=%-3d max_size=%d | bound unions=%-3d rand unions=%-3d ceil(t/2)=%-3d '
          'rand sizes=%s' % (n, target, ms, mb, mr, want, sorted({len(g) for g in cl_r})))
check('两臂 union 次数相同且 = ceil(target/2)，簇内 unit 互不重复、size <= max_size', all_ok)

n, target, ms = 100, 25, 4
S = np.full((n, n), 0.99); np.fill_diagonal(S, -1.0)
cl_r = v3.clusters_random_sized(n, target, random.Random(7), ms)
check('(a) 新随机对照能产出 size = max_size 的簇', max(len(g) for g in cl_r) == ms,
      'max_size=%d 实际最大簇=%d' % (ms, max(len(g) for g in cl_r)))
cl_r2 = v3.clusters_random_sized(64, 64, random.Random(11), 4)
check('(a) 预算充足时 size-4 簇占多数', sorted(len(g) for g in cl_r2) == [3] + [4] * 10,
      str(sorted(len(g) for g in cl_r2)))
check('(b) 预算不足时末簇按预算截断（size 3）而非超发', merges(cl_r2) == math.ceil(64 / 2) == 32)

# 旧对照（--rand-legacy）：结构上恒为 size 2 —— 审计发现 #2 的证据
legacy = v3.clusters_from_pairs(n, v3.pairs_random(n, target, random.Random(7), min_gap=2))
check('旧 rand4_emb 路径恒为 size 2（无法构成 size-4 簇）',
      len(legacy) > 0 and all(len(c) == 2 for c in legacy),
      '簇数=%d 最大簇=%d' % (len(legacy), max(len(c) for c in legacy)))

# tau 挡住相似度时：bound 臂 0 次合并，随机对照仍按预算出簇（对照不看相似度，这是它的定义）。
# 注意：这是 budget 模式（--rand-stop budget / stop_unions=None）的行为；matched 模式见下方 #4。
S_low = np.full((n, n), 0.5); np.fill_diagonal(S_low, -1.0)
cl_b_low = v3.pairs_similarity(S_low, target, None, ms, tau=0.75)
check('tau 全线不满足 -> bound 臂 0 簇；budget 模式随机对照仍消耗预算',
      merges(cl_b_low) == 0 and merges(cl_r) == math.ceil(target / 2),
      'bound=%d rand=%d' % (merges(cl_b_low), merges(cl_r)))

# ---------------- #3 tau 是 edge-level，不是簇直径上界 ----------------
print('#3 tau 语义：链式合并可把相似度 < tau 的两点放进同一簇')
tau = 0.75
S3 = np.array([[1.0, 0.90, 0.10],
               [0.90, 1.0, 0.90],
               [0.10, 0.90, 1.0]]); np.fill_diagonal(S3, -1.0)
cl3 = v3.pairs_similarity(S3, 4, None, 4, tau=tau)   # target=4 -> 允许两次 union
check('构造用例：{0,1,2} 同簇，但 sim(0,2)=0.10 < tau', cl3 == [[0, 1, 2]], str(cl3))
check('簇内最小成对相似度 < tau -> tau 不是簇直径下界',
      min(S3[i, j] for c in cl3 for i in c for j in c if i < j) < tau)

# ---------------- #4 合并匹配：stop_unions ----------------
print('#4 合并匹配：随机臂恰好做 bound 臂同样多的 union（真实计数，不用 target 估算）')

# 停止条件精确性：stop_unions=k -> 恰好 k 次 union（不差一，且 0 也不多跑）
exact = True
for k in range(0, 9):
    st = {}
    cl_k = v3.clusters_random_sized(200, 50, random.Random(100 + k), 4, stop_unions=k, stats=st)
    exact &= (st['unions'] == k == merges(cl_k) == v3.unions_of(cl_k) and st['clusters'] == len(cl_k))
check('stop_unions=k 精确执行 k 次 union（k=0..8，不差一）', exact)

def mk_case(n, blocks, base=0.1):
    """blocks: [(start, end, sim)] 块内相似度；块间 base。用于构造不同受限结构。"""
    S = np.full((n, n), base)
    for a, b, s in blocks:
        S[a:b, a:b] = s
    np.fill_diagonal(S, -1.0)
    return S

CASES = [
    ('全同高相似(预算受限)', mk_case(80, [(0, 80, 0.99)]), 20, 4),
    ('tau 全线不满足', mk_case(80, [], 0.5), 20, 4),
    ('只有 6 个 unit 相似', mk_case(40, [(0, 6, 0.90)]), 20, 4),
    ('分块 + max_size=2', mk_case(60, [(0, 20, 0.95)]), 30, 2),
    ('小预算 ratio=0.125', mk_case(100, [(0, 40, 0.90)]), 12, 4),
]
all_match = True
for name, S, target, ms in CASES:
    sb = {}; cl_b = v3.pairs_similarity(S, target, None, ms, tau=0.75, stats=sb)
    sr = {}; cl_r = v3.clusters_random_sized(S.shape[0], target, random.Random(9), ms,
                                            stop_unions=sb['unions'], stats=sr)
    ok = (sr['unions'] == sb['unions']                       # 两臂真实 union 数完全相同
          and merges(cl_b) == sb['unions'] == merges(cl_r) == sr['unions'])   # 报告口径 = 簇结构
    all_match &= ok
    print('       %-20s bound unions=%2d  rand(matched) unions=%2d  budget 模式会做=%2d  '
          '簇数 bound=%3d rand=%3d' % (name, sb['unions'], sr['unions'], math.ceil(target / 2),
                                       len(cl_b), len(cl_r)))
check('matched：5 种受限结构下两臂 union 数逐 case 完全相同', all_match)

# GPU 上 1370 vs 888 的成因：tau 是约束 -> bound 臂提前停，而默认（matched）随机臂不再超支
sb2 = {}; v3.pairs_similarity(S_low, target, None, ms, tau=0.75, stats=sb2)
sr2 = {}; cl_r_b = v3.clusters_random_sized(n, target, random.Random(3), ms,
                                           stop_unions=sb2['unions'], stats=sr2)
sr3 = {}; v3.clusters_random_sized(n, target, random.Random(3), ms, stats=sr3)
check('tau 全线不满足：matched 随机臂 0 union（= bound 臂），budget 模式仍超支',
      sb2['unions'] == 0 and sr2['unions'] == 0 and len(cl_r_b) == 0
      and sr3['unions'] == math.ceil(target / 2),
      'bound=%d matched=%d budget=%d' % (sb2['unions'], sr2['unions'], sr3['unions']))

# unit 不够时只做得到的次数、绝不超发（差额会在报告的 merge_parity 里显示为不等）
sr4 = {}; cl_r4 = v3.clusters_random_sized(4, 100, random.Random(1), 4, stop_unions=10, stats=sr4)
check('unit 不足时不超发（4 个 unit 最多 3 次 union，达不到 10 就少做）',
      sr4['unions'] == 3 == merges(cl_r4) and sr4['unions'] <= 10)

# budget 模式 = 上一版行为，未被改动
sr5 = {}; v3.clusters_random_sized(200, 50, random.Random(2), 4, stats=sr5)
check('budget 模式仍 = ceil(target/2)（上一版行为未改动）', sr5['unions'] == math.ceil(50 / 2) == 25)

# stats 交叉校验：内部计数器 == 森林恒等式 Σ(|c|-1)（报告列用前者，两者必须一致）
sb6 = {}; cl_b6 = v3.pairs_similarity(mk_case(60, [(0, 30, 0.95)]), 20, None, 4, tau=0.75, stats=sb6)
sr6 = {}; cl_r6 = v3.clusters_random_sized(60, 20, random.Random(4), 4, stop_unions=sb6['unions'], stats=sr6)
check('stats[unions] 与 Σ(|c|-1)（unions_of）一致，两臂都校验',
      sb6['unions'] == merges(cl_b6) == v3.unions_of(cl_b6)
      and sr6['unions'] == merges(cl_r6) == v3.unions_of(cl_r6) and sb6['unions'] == sr6['unions'],
      'bound=%d rand=%d' % (sb6['unions'], sr6['unions']))

print('\nALL %d CHECKS PASSED' % n_ok)