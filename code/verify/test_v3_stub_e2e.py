r"""v3_batch.py 审计修订的 stub 端到端 A/B 测试（不需要下载/加载真实编码器）。

做法：用 numpy 伪造最小可用的 torch / transformers 接口（FakeTensor / FakeTokenizer /
FakeModel，24 层），让新旧两份 v3_batch.py 在**同一份合成冻结数据**上跑完 bound 配置，
然后逐字节比较报告数据行与 cells.json。旧文件只读引用，不修改：
  OLD = ..\remote_probe\code\v3_batch.py   （= 已发布数字所用的实现，md5 dd0b65ec6e）
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
import importlib.util, io, json, os, random, shutil, sys, zlib
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OLD_PATH = os.path.join(BASE, 'remote_probe', 'code', 'v3_batch.py')
NEW_PATH = os.path.join(BASE, 'hpc', 'v3_batch.py')
RUN = os.path.join(HERE, '_stub_run')
N_LAYERS, DIM = 24, 8
MODEL_NAME = 'fake-encoder-24l'

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
    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors='pt'):
        # 每个 unit 的「词表 id」= 文本 topic（crc32%4），同 topic 的 unit 得到相同隐状态
        ids = [[zlib.crc32(t.encode('utf-8')) % 4] * 6 for t in texts]
        T = max(len(x) for x in ids)
        ii = np.zeros((len(ids), T)); am = np.zeros((len(ids), T))
        for b, x in enumerate(ids):
            ii[b, :len(x)] = x; am[b, :len(x)] = 1.0
        return FakeBatch(input_ids=FakeTensor(ii), attention_mask=FakeTensor(am))

class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=N_LAYERS)
    def eval(self): return self
    def to(self, dev): return self
    def __call__(self, input_ids=None, attention_mask=None, **kw):
        ii = input_ids.a if isinstance(input_ids, FakeTensor) else np.asarray(input_ids)
        d = np.arange(DIM)[None, None, :]
        hs = tuple(FakeTensor(np.sin(0.13 * ii[..., None] + 0.7 * k + d)) for k in range(N_LAYERS + 1))
        return SimpleNamespace(hidden_states=hs, last_hidden_state=hs[-1])

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
for r in range(3):
    sents = ["Unit %02d of record %d topic %d carries filler words for the audit run." % (i, r, i % 4)
             for i in range(12)]
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

print('--- A/B：旧实现 vs 新实现 --layer last --rand-legacy ---')
print('OLD rows: %s' % data_rows(rep_o))
print('NEW rows: %s' % data_rows(rep_l))
check('报告数据行逐字一致（末层 + 旧 size-2 对照 = 旧行为）', data_rows(rep_o) == data_rows(rep_l))
check('报告首行（config/samples/.../dev）一致', rep_o[0] == rep_l[0], rep_o[0])
check('cells.json 的 cells 完全一致', js_o['cells'] == js_l['cells'])
check('非 cells 字段（modes/ratios/records）一致',
      all(js_o[k] == js_l[k] for k in ('config', 'granularity', 'records', 'cap', 'modes', 'ratios')))
check('旧文件 JSON 无新键（对照基线未被污染）', 'layer_idx' not in js_o and 'tau' not in js_o)

check('非 bound 配置（guard）在 --layer last 下报告数据行也逐字一致',
      data_rows(rep_go) == data_rows(rep_gn) and len(data_rows(rep_go)) > 3,
      '%d rows' % len(data_rows(rep_go)))
check('非 bound 配置（guard）cells 完全一致（含 random_adj 的 rng 消耗顺序）',
      js_go['cells'] == js_gn['cells'])

print('--- 审计 #3：新产物的可复核元数据 ---')
for tag, rep, js, want in (('legacy', rep_l, js_l, (23, 'last', 'legacy_size2')),
                           ('early', rep_e, js_e, (4, 'early', 'sized_random')),
                           ('idx7', rep_i, js_i, (7, 'idx', 'sized_random'))):
    idx, src, rand = want
    meta = rep[1]
    ok = (js['layer_idx'] == idx and js['layer'] == src and js['n_layers'] == N_LAYERS
          and js['model'] == MODEL_NAME and js['tau'] == 0.75 and js['max_size'] == 4
          and js['rand_mode'] == rand
          and ('layer_idx=%d' % idx) in meta and ('tau=0.750' in meta) and ('max_size=4' in meta)
          and (MODEL_NAME in meta) and ('rand=%s' % rand) in meta)
    check('%-7s 报告头/JSON 记录 model+layer_idx+tau+max_size+rand' % tag, ok, meta[:96] + ' ...')
check('报告含 tau edge-level 说明行', any(l.startswith('note: tau 只约束合并边') for l in rep_e))
check('JSON 含 tau_semantics / rand_mode', 'edge-level' in js_e['tau_semantics'] and js_e['rand_mode'] == 'sized_random')

print('--- 审计 #2：默认随机对照 vs 旧 size-2 对照 ---')
check('默认 rand4_emb 与旧对照的簇数不同（选择准则之外的簇大小已改变）',
      js_e['cells']['rand4_emb|0.25'] != js_l['cells']['rand4_emb|0.25'],
      'new=%s old=%s' % (js_e['cells']['rand4_emb|0.25'], js_l['cells']['rand4_emb|0.25']))
check('同一层内改随机对照不波及 sim_emb/bound_emb 臂（非随机臂未被波及）',
      js_e['cells']['sim_emb|0.25'] == js_el['cells']['sim_emb|0.25']
      and js_e['cells']['bound_emb|0.25'] == js_el['cells']['bound_emb|0.25'],
      'sim=%s bound=%s' % (js_e['cells']['sim_emb|0.25'], js_e['cells']['bound_emb|0.25']))
check('层选择确实改变簇结构（stub 级证据，#1 的前提；真实大小需 GPU 跑真编码器）',
      js_el['cells']['sim_emb|0.25'] != js_l['cells']['sim_emb|0.25'],
      'early=%s last=%s' % (js_el['cells']['sim_emb|0.25'], js_l['cells']['sim_emb|0.25']))

print('\nALL %d CHECKS PASSED   (artifacts: %s)' % (n_ok, RUN))
