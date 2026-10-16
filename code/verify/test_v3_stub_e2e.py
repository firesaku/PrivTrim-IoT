r"""v3_batch.py 审计修订的 stub 端到端 A/B 测试（不需要下载/加载真实编码器）。

做法：用 numpy 伪造最小可用的 torch / transformers 接口（FakeTensor / FakeTokenizer /
FakeModel，24 层），让新旧两份 v3_batch.py 在**同一份合成冻结数据**上跑完 bound 配置，
然后逐字节比较报告数据行与 cells.json。旧文件只读引用，不修改；基线路径现查现认
（repo / repo_staging / remote_probe 里第一个仍是 288 行原版的副本，见 _pristine()）：
  OLD = 未修改的已发布实现（md5 dd0b65ec6e）
  NEW = ..\hpc\v3_batch.py                 （本次审计修订）

断言：
  1) NEW --layer last --rand-legacy 的报告数据行 / cells 与 OLD 完全一致（末层 + 旧随机对照 = 旧行为）
  2) NEW 默认（--layer early，大小匹配随机对照）把 model / layer_idx / n_layers / tau /
     max_size / rand_mode 写进报告头与 JSON
  3) --layer-idx 7 端到端生效（JSON layer_idx=7, layer=idx）
  4) 默认随机对照确实改变了 rand4_emb 的簇（与旧 size-2 对照不同）
注意：stub 的隐状态只按 topic 分组，因此本测试不能证明「早层 vs 末层簇结构不同」——
那需要真实编码器（本地无法验证）。

运行：python D:\CodexData\tmp\privtrim-review-20261015\verify\test_v3_stub_e2e.py
"""
import importlib.util, io, json, os, random, re, shutil, sys, zlib
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OLD_PATH = os.path.join(BASE, 'remote_probe', 'code', 'v3_batch.py')
NEW_PATH = os.path.join(BASE, 'hpc', 'v3_batch.py')
RUN = os.path.join(HERE, '_stub_run')
N_LAYERS, DIM = 24, 8
MODEL_NAME = 'fake-encoder-24l'

def _pristine(p):
    """已发布实现（288 行、无审计修订）的判据：不含 layer_idx / clusters_random_sized。"""
    if not os.path.exists(p):
        return False
    t = io.open(p, encoding='utf-8', errors='replace').read()
    return ('layer_idx' not in t and 'clusters_random_sized' not in t and 'rand4_emb' in t
            and 'last_hidden_state' in t)

# 注意：上游会把改过的文件同步到 remote_probe\code\，所以基线必须现查现认，不能写死路径
OLD_CANDS = [os.path.join(BASE, 'repo', 'PrivTrim-IoT', 'code', 'v3_batch.py'),
             os.path.join(BASE, 'repo_staging', 'extracted', 'v3_batch.py'),
             OLD_PATH]
OLD_PATH = next((p for p in OLD_CANDS if _pristine(p)), None)
if OLD_PATH is None:
    print('FAIL: 找不到未修改的已发布实现副本（%s 都已被同步覆盖），A/B 无法进行' % ', '.join(OLD_CANDS))
    sys.exit(1)
print('A/B 基线（未修改的已发布实现）: %s' % os.path.relpath(OLD_PATH, BASE))

# ---------------- 伪造 torch / transformers ----------------
class FakeTensor:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def unsqueeze(self, d): return FakeTensor(np.expand_dims(self.a, d))
    def float(self): return self
    def sum(self, dim): return FakeTensor(self.a.sum(axis=dim))
    def clamp(self, min=0.0): return FakeTensor(np.clip(self.a, min, None))
    def cpu(self): return self
    def to(self, dev): return self
    def numpy(self): return self.a
    def __mul__(self, o): return FakeTensor(self.a * (o.a if isinstance(o, FakeTensor) else o))
    __rmul__ = __mul__
    def __truediv__(self, o): return FakeTensor(self.a / (o.a if isinstance(o, FakeTensor) else o))

class FakeBatch(dict):
    def to(self, dev): return self

class FakeTok:
    """把文本映射成「词表 id」：id 决定隐状态。
       带 ZZT<k>ZZ 标签 -> id = 100000+k（同标签 = 同一向量 -> 相似度 1.0）；
       无标签 -> id = 200000 + crc32%4（原构造：4 组 topic，组内相似度 1.0）；
       标签 20/21 是特意构造的「跨层 tau 边界对」(见 FakeModel)。"""
    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors='pt'):
        def _id(t):
            m = re.search(r'ZZT(\d+)ZZ', t)
            return 100000 + int(m.group(1)) if m else (200000 + zlib.crc32(t.encode('utf-8')) % 4)
        ids = [[_id(t)] * 6 for t in texts]
        T = max(len(x) for x in ids)
        ii = np.zeros((len(ids), T)); am = np.zeros((len(ids), T))
        for b, x in enumerate(ids):
            ii[b, :len(x)] = x; am[b, :len(x)] = 1.0
        return FakeBatch(input_ids=FakeTensor(ii), attention_mask=FakeTensor(am))

def _stub_vec(tid, k):
    """每个 (id, 层) 一个确定性的单位向量：不同 id -> 32 维随机方向（实测两两 |cos| < 0.5，
       远低于 tau=0.75）；同 id -> 同一向量（相似度 1.0）。
       100020/100021 是一对「跨层移动的相似度」：sim = cos(0.13+0.05k)，
       k=4 时 0.946 >= tau（可合并），k=23 时 0.287 < tau（被 tau 挡住）
       —— 用来在 stub 上证明「换层会改变 tau 边界处的簇结构」（审计 #1 的前提）。"""
    if tid == 100020:
        v = np.zeros(DIM); v[0] = 1.0; return v
    if tid == 100021:
        th = 0.13 + 0.05 * k
        v = np.zeros(DIM); v[0] = np.cos(th); v[1] = np.sin(th); return v
    rng = np.random.RandomState((int(tid) * 7919 + k * 104729) % (2 ** 31 - 1))
    v = rng.randn(DIM)
    return v / np.linalg.norm(v)

class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=N_LAYERS)
    def eval(self): return self
    def to(self, dev): return self
    def __call__(self, input_ids=None, attention_mask=None, **kw):
        ii = input_ids.a.astype(np.int64) if isinstance(input_ids, FakeTensor) else np.asarray(input_ids)
        hs = []
        for k in range(N_LAYERS + 1):
            H = np.zeros(ii.shape + (DIM,))
            for tid in np.unique(ii):
                if tid == 0:            # padding
                    continue
                H[ii == tid] = _stub_vec(int(tid), k)
            hs.append(FakeTensor(H))
        return SimpleNamespace(hidden_states=tuple(hs), last_hidden_state=hs[-1])

class _NoGrad:
    def __call__(self, f): return f
    def __enter__(self): return None
    def __exit__(self, *a): return False

def _normalize(v, dim=-1):
    a = v.a
    return FakeTensor(a / np.maximum(np.linalg.norm(a, axis=dim, keepdims=True), 1e-12))

def _tok_from_pretrained(*a, **k): return FakeTok()
def _model_from_pretrained(*a, **k): return FakeModel()

torch_mod = SimpleNamespace(
    no_grad=_NoGrad,
    cuda=SimpleNamespace(is_available=lambda: False),
    nn=SimpleNamespace(functional=SimpleNamespace(normalize=_normalize)))
tf_mod = SimpleNamespace(
    AutoTokenizer=SimpleNamespace(from_pretrained=_tok_from_pretrained),
    AutoModel=SimpleNamespace(from_pretrained=_model_from_pretrained))
sys.modules['torch'] = torch_mod
sys.modules['transformers'] = tf_mod

# ---------------- 合成冻结数据 ----------------
shutil.rmtree(RUN, ignore_errors=True)
DATA = os.path.join(RUN, 'data'); os.makedirs(DATA)
recs = []
# 记录 0：12 个 unit 同标签 -> 相似度全 1.0，bound 臂能花完预算（matched 与 budget 一致）
# 记录 1：只有 unit0/unit1 同标签、其余互不相似 -> tau 下只有 1 条可合并边，bound 臂花不完预算
#         （budget 模式会超支 -> merge_parity 不等，必须逐条可见；matched 模式两臂相等）
# 记录 2：无标签（4 组 topic，组内 1.0）-> 每组 3 个 unit，bound 臂花完预算
# 记录 3：一对「跨层 tau 边界」unit（ZZT20ZZ/ZZT21ZZ）+ 10 个互不相似 -> k=4 可合并、k=23 被 tau 挡住
TAGS = [['ZZT1ZZ'] * 12,
        ['ZZT2ZZ', 'ZZT2ZZ'] + ['ZZT%dZZ' % (3 + i) for i in range(10)],
        [''] * 12,
        ['ZZT20ZZ', 'ZZT21ZZ'] + ['ZZT%dZZ' % (3 + i) for i in range(10)]]
for r in range(4):
    sents = ["Unit %02d of record %d topic %d carries filler words for the audit run. %s"
             % (i, r, i % 4, TAGS[r][i]) for i in range(12)]
    recs.append({'context': " ".join(sents),
                 'answers': ["unit03ofrecord%dtopic3" % r],
                 'canaries': [{'value': "topic 0 carries", 'marker': 'auditcanarymarker0000',
                               'fragment_length': 12}]})
blob = "\n".join(json.dumps(x) for x in recs) + "\n"
io.open(os.path.join(DATA, 'frozen_samples_scattered.jsonl'), 'w', encoding='utf-8').write(blob)
io.open(os.path.join(DATA, 'frozen_samples.jsonl'), 'w', encoding='utf-8').write(blob)   # guard/gran 用

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def run(mod, outdir, extra, cfg='bound'):
    argv = [NEW_PATH, '--config', cfg, '--datadir', DATA, '--outdir', outdir,
            '--model', MODEL_NAME] + extra
    old_argv, sys.argv = sys.argv, argv
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    rep = io.open(os.path.join(outdir, 'v3_%s_report.txt' % cfg), encoding='utf-8').read().splitlines()
    js = json.loads(io.open(os.path.join(outdir, 'v3_%s_cells.json' % cfg), encoding='utf-8').read())
    return rep, js

MODE_NAMES = ('sim_emb', 'cons_emb', 'imp_emb', 'cons_imp_emb', 'random_adj', 'random_far',
              'bound_emb', 'rand4_emb')
def data_rows(rep):
    return [l for l in rep if l.startswith('mode ') or l.split(' ')[0] in MODE_NAMES]

def core(row):
    """已发布列（mode ratio clusters hit_answer mix_sens fuse_2sec），忽略审计 #4 新增的 unions 列"""
    return row.split()[:6]

n_ok = 0
def check(name, cond, extra=''):
    global n_ok
    if not cond:
        print('  FAIL %s %s' % (name, extra)); sys.exit(1)
    n_ok += 1
    print('  ok   %-58s %s' % (name, extra))

old = load('v3_old', OLD_PATH)
new = load('v3_new', NEW_PATH)
rep_o, js_o = run(old, os.path.join(RUN, 'out_old'), [])
rep_l, js_l = run(new, os.path.join(RUN, 'out_legacy'), ['--layer', 'last', '--rand-legacy'])
rep_e, js_e = run(new, os.path.join(RUN, 'out_early'), [])
rep_i, js_i = run(new, os.path.join(RUN, 'out_idx7'), ['--layer-idx', '7'])
rep_el, js_el = run(new, os.path.join(RUN, 'out_early_legacy'), ['--layer', 'early', '--rand-legacy'])
rep_go, js_go = run(old, os.path.join(RUN, 'out_guard_old'), [], cfg='guard')
rep_gn, js_gn = run(new, os.path.join(RUN, 'out_guard_new'), ['--layer', 'last'], cfg='guard')
rep_b, js_b = run(new, os.path.join(RUN, 'out_budget'), ['--rand-stop', 'budget'])
rep_g0, js_g0 = run(new, os.path.join(RUN, 'out_gap0'), ['--rand-min-gap', '0'])

print('--- A/B：旧实现 vs 新实现 --layer last --rand-legacy ---')
print('OLD rows: %s' % data_rows(rep_o))
print('NEW rows: %s' % data_rows(rep_l))
check('报告数据行逐字一致（末层 + 旧 size-2 对照 = 旧行为）', data_rows(rep_o) == data_rows(rep_l))
check('报告首行（config/samples/.../dev）一致', rep_o[0] == rep_l[0], rep_o[0])
check('cells.json 的 cells 完全一致', js_o['cells'] == js_l['cells'])
check('非 cells 字段（modes/ratios/records）一致',
      all(js_o[k] == js_l[k] for k in ('config', 'granularity', 'records', 'cap', 'modes', 'ratios')))
check('旧文件 JSON 无新键（对照基线未被污染）', 'layer_idx' not in js_o and 'tau' not in js_o)

check('非 bound 配置（guard）在 --layer last 下已发布列逐字一致（差异仅新增 unions 列）',
      [core(r) for r in data_rows(rep_go)] == [core(r) for r in data_rows(rep_gn)]
      and len(data_rows(rep_go)) > 3
      and not any(r.rstrip().endswith('unions') for r in data_rows(rep_go))
      and any(r.rstrip().endswith('unions') for r in data_rows(rep_gn)),
      '%d rows' % len(data_rows(rep_go)))
check('非 bound 配置（guard）cells 完全一致（含 random_adj 的 rng 消耗顺序）',
      js_go['cells'] == js_gn['cells'])

print('--- 审计 #3：新产物的可复核元数据 ---')
for tag, rep, js, want in (('legacy', rep_l, js_l, (23, 'last', 'legacy_size2', 'matched', 2)),
                           ('early', rep_e, js_e, (4, 'early', 'sized_matched_unions', 'matched', 2)),
                           ('idx7', rep_i, js_i, (7, 'idx', 'sized_matched_unions', 'matched', 2)),
                           ('budget', rep_b, js_b, (4, 'early', 'sized_target_budget', 'budget', 2)),
                           ('gap0', rep_g0, js_g0, (4, 'early', 'sized_matched_unions', 'matched', 0))):
    idx, src, rand, stop, gap = want
    meta = rep[1]
    ok = (js['layer_idx'] == idx and js['layer'] == src and js['n_layers'] == N_LAYERS
          and js['model'] == MODEL_NAME and js['tau'] == 0.75 and js['max_size'] == 4
          and js['rand_mode'] == rand
          and js['rand_stop'] == stop and js['rand_min_gap'] == gap
          and ('layer_idx=%d' % idx) in meta and ('tau=0.750' in meta) and ('max_size=4' in meta)
          and ('rand_stop=%s' % stop) in meta and ('rand_min_gap=%d' % gap) in meta
          and (MODEL_NAME in meta) and ('rand=%s' % rand) in meta)
    check('%-7s 报告头/JSON 记录 model+layer_idx+tau+max_size+rand(+stop/gap)' % tag, ok, meta[:96] + ' ...')
check('报告含 tau edge-level 说明行', any(l.startswith('note: tau 只约束合并边') for l in rep_e))
check('JSON 含 tau_semantics / rand_mode',
      'edge-level' in js_e['tau_semantics'] and js_e['rand_mode'] == 'sized_matched_unions')

print('--- 审计 #2：默认随机对照 vs 旧 size-2 对照 ---')
check('默认 rand4_emb 与旧对照的簇数不同（选择准则之外的簇大小已改变）',
      js_e['cells']['rand4_emb|0.25'] != js_l['cells']['rand4_emb|0.25'],
      'new=%s old=%s' % (js_e['cells']['rand4_emb|0.25'], js_l['cells']['rand4_emb|0.25']))
check('同一层内改随机对照不波及 sim_emb/bound_emb 臂（非随机臂未被波及）',
      js_e['cells']['sim_emb|0.25'] == js_el['cells']['sim_emb|0.25']
      and js_e['cells']['bound_emb|0.25'] == js_el['cells']['bound_emb|0.25'],
      'sim=%s bound=%s' % (js_e['cells']['sim_emb|0.25'], js_e['cells']['bound_emb|0.25']))
check('层选择改变 tau 边界处的簇结构（记录 3：k=4 可合并、k=23 被 tau 挡住；#1 的前提）',
      js_el['cells']['bound_emb|0.25'] != js_l['cells']['bound_emb|0.25'],
      'early=%s last=%s' % (js_el['cells']['bound_emb|0.25'], js_l['cells']['bound_emb|0.25']))

print('--- 审计 #4：合并匹配（matched 默认）---')
mp = js_e['merge_parity']
check('matched：merge_parity 逐记录×ratio 100% 相等、无 mismatch 条目',
      mp['total'] == 4 and mp['equal'] == 4 and mp['unequal'] == 0 and mp['equal_pct'] == 100.0
      and js_e['merge_parity_mismatch'] == [],
      'total=%d equal=%d unequal=%d' % (mp['total'], mp['equal'], mp['unequal']))
check('matched：两臂真实 union 数相同（arm_unions 落盘，不再只报簇数）',
      js_e['arm_unions']['bound_emb|0.25'] == js_e['arm_unions']['rand4_emb|0.25'],
      'bound=%d rand=%d' % (js_e['arm_unions']['bound_emb|0.25'], js_e['arm_unions']['rand4_emb|0.25']))
check('matched：报告含 unions 列、每臂 clusters/unions 行、merge_parity 行',
      any(l.startswith('mode ') and l.rstrip().endswith('unions') for l in rep_e)
      and any(l.startswith('arm bound_emb') and 'unions=' in l for l in rep_e)
      and any(l.startswith('arm rand4_emb') and 'unions=' in l for l in rep_e)
      and any(l.startswith('merge_parity(rand4_emb vs bound_emb') for l in rep_e))
check('--rand-legacy 冻结已发布表格式（表头无 unions 列）且 parity 标 n/a',
      any(l.startswith('mode ') and not l.rstrip().endswith('unions') for l in rep_l)
      and any(l.startswith('merge_parity: n/a') for l in rep_l)
      and js_l['merge_parity']['total'] == 0)
check('budget 模式（上一版行为）：超支被逐条记录，不平均掉',
      js_b['merge_parity']['unequal'] >= 1
      and len(js_b['merge_parity_mismatch']) == js_b['merge_parity']['unequal']
      and all(m['rand_unions'] > m['bound_unions'] for m in js_b['merge_parity_mismatch'])
      and any(l.strip().startswith('mismatch: line=') for l in rep_b)
      and js_b['arm_unions']['rand4_emb|0.25'] > js_b['arm_unions']['bound_emb|0.25'],
      'unequal=%d/%d rand=%d bound=%d' % (js_b['merge_parity']['unequal'], js_b['merge_parity']['total'],
                                          js_b['arm_unions']['rand4_emb|0.25'], js_b['arm_unions']['bound_emb|0.25']))
check('budget 模式超支只体现在 unions 上、簇数可能恰好相同（只看簇数会漏掉它，故两者都必须报）',
      js_b['arm_unions']['rand4_emb|0.25'] > js_e['arm_unions']['rand4_emb|0.25']
      and js_b['cells']['rand4_emb|0.25'] == js_e['cells']['rand4_emb|0.25'],
      'unions budget=%d matched=%d，而簇数两边都是 %s'
      % (js_b['arm_unions']['rand4_emb|0.25'], js_e['arm_unions']['rand4_emb|0.25'],
         js_e['cells']['rand4_emb|0.25']))
check('--rand-min-gap 0 也走 matched 且 parity 相等（min_gap 不影响配平）',
      js_g0['rand_min_gap'] == 0 and js_g0['merge_parity']['equal'] == js_g0['merge_parity']['total'],
      'gap=%d equal=%d/%d' % (js_g0['rand_min_gap'], js_g0['merge_parity']['equal'], js_g0['merge_parity']['total']))

print('\nALL %d CHECKS PASSED   (artifacts: %s)' % (n_ok, RUN))
