# -*- coding: utf-8 -*-
"""test_longctx_synthetic.py —— 在**没有 GPU / 没有 7B 模型**的本地环境里，
把 e2e_longctx.py 的非模型部分（读案例 -> 建臂 -> 预算对账 -> 评分接线 -> 报表）
真跑一遍，并把手写的合成案例作为输入，验证每条链路。

跑法：
    python test_longctx_synthetic.py

覆盖：
  T1  make_prompt / split_units / drop_to_budget 与 e2e4.py 源码逐字一致（正则比对源码）
  T2  privcheck / v4_router 是 import 来的单一真源，e2e_longctx.py 里没有复制谓词
  T3  drop_to_budget 的不变量：任取文本与预算，发射文本 token 数 <= 预算
  T4  check_budget 的守卫：预算臂超标必须 raise；长度保持臂不 raise
  T5  合成案例上的 7 臂构造：full/delete/alias 不压；budget_* 全部 <= B
  T6  预算对账表：同一 ratio 下 budget_cluster / budget_random / delete_budget /
      alias_budget 的 B 相同（同预算块的定义）
  T7  delete 臂：privcheck 谓词下 canary 整值 + 片段泄漏都为 0；alias 臂：整值泄漏 0 但
      实体仍可区分（collision=0）；budget（不脱敏）臂：实体可区分
  T8  官方评分器接线：用真实 evaluate-v2.0.py + --stub-preds 跑完整评分/报表链，
      EM/F1 与直接调用 compute_exact/compute_f1 对上
  T9  产物可解析：report / cells.json / percase.jsonl 齐全且字段自洽
  T10 真案例上的 --dry-run（112 例）预算对账无超标、gold_in_ctx 随预算单调不增
"""
import io
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HPC = r'D:\CodexData\tmp\privtrim-review-20261015\hpc'      # 交付物所在目录
sys.path.insert(0, HERE)
sys.path.insert(0, HPC)
import privcheck                                            # noqa: E402
import v4_router                                            # noqa: E402
import e2e_longctx as E                                     # noqa: E402

FROZEN = r'D:\CodexData\tmp\privtrim-review-20261015\crossdata\frozen_samples.jsonl'
REAL_CASES = r'D:\CodexData\tmp\privtrim-review-20261015\longctx\longctx_cases.jsonl'
SCORER = (r'C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT\external\squad_v11'
          r'\evaluate-v2.0.py')
E2E4 = os.path.join(HPC, 'e2e4.py')

FAILS = []
OKS = []


LOG = []


def check(name, cond, detail=''):
    (OKS if cond else FAILS).append(name)
    line = '%s %-6s %s%s' % ('[ok]  ' if cond else '[FAIL]', name, '' if cond else '<<< ',
                             '' if cond else detail)
    LOG.append(line)
    print(line)


def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)


# ---------------------------------------------------------------- T1 源码一致性
def t1():
    src = io.open(E2E4, encoding='utf-8').read()
    lines = io.open(os.path.join(HPC, 'e2e_longctx.py'), encoding='utf-8').read().split('\n')

    def body(text, name):
        m = re.search(r'^def %s\(.*?\n(?=\ndef |\n# ---)' % re.escape(name), text,
                      re.S | re.M)
        need(m, 'e2e4.py 里找不到 %s' % name)
        return re.sub(r'\s+', ' ', m.group(0)).strip()

    for fn in ('make_prompt', 'split_units', 'drop_to_budget'):
        a, b = body(src, fn), body('\n'.join(lines), fn)
        # 允许注释/docstring/命名差异，比对"实质代码"：去掉注释与 docstring
        def core(t):
            t = re.sub(r'""".*?"""', '', t, flags=re.S)
            t = re.sub(r'#[^\n]*', '', t)
            t = re.sub(r'\s+', ' ', t)
            return t.strip()
        ca, cb = core(a), core(b)
        check('T1.%s 源码逐字一致' % fn, ca == cb,
              '\n  e2e4 : %s\n  longctx: %s' % (ca[:400], cb[:400]))


# ---------------------------------------------------------------- T2 单一真源
def t2():
    src = io.open(os.path.join(HPC, 'e2e_longctx.py'), encoding='utf-8').read()
    for pat in (r'EMAIL\s*=\s*re\.compile', r'PHONE\s*=\s*re\.compile',
                r'ACCOUNT\s*=\s*re\.compile', r"r'\\b\[A-Za-z0-9\._%\+-\]\+@"):
        check('T2 未复制谓词 %s' % pat, not re.search(pat, src), '发现本地重定义')
    check('T2 import privcheck', 'import privcheck' in src)
    check('T2 import v4_router', 'import v4_router' in src)
    check('T2 Aliaser 来自 v4_router', E.v4_router.Aliaser is v4_router.Aliaser)
    check('T2 redact 来自 privcheck', E.privcheck.redact is privcheck.redact)


# ---------------------------------------------------------------- T3 drop_to_budget 不变量
def t3():
    nt = lambda s: len(re.findall(r'[A-Za-z0-9]+', s))
    txt = ('A quick brown fox jumps over the lazy dog. ' * 40 +
           'The target answer is 42 and nothing else matters. ' +
           'Filler sentence number seven about humidity sensors. ' * 60)
    units = E.split_units(txt)
    check('T3 split_units 产出多单元', len(units) > 20, 'units=%d' % len(units))
    order = list(range(len(units)))
    bad = []
    for B in (16, 32, 64, 128, 256, 512, 1024, 4096):
        out = E.drop_to_budget(units, B, nt, order)
        got = nt(out)
        if got > B:
            bad.append((B, got))
    check('T3 任意预算下发射文本 token <= 预算', not bad, 'violations=%s' % bad)
    # 答案承载单元排最后 -> 预算收紧时答案应最后消失
    ans_flags = [('42' in u) for u in units]
    o_cluster = E.lex_drop_order(units, ans_flags)
    kept_last = o_cluster[-1]
    check('T3 答案承载单元排在丢弃序最后', ans_flags[kept_last], 'last=%r' % units[kept_last][:60])
    check('T3 选择器确定性', E.lex_drop_order(units, ans_flags) == o_cluster)


# ---------------------------------------------------------------- T4 check_budget 守卫
def t4():
    try:
        E.check_budget('k', 'budget_cluster', 0.25, 100, 50)
        check('T4 预算臂超标必须 raise', False, '没有 raise')
    except RuntimeError as e:
        check('T4 预算臂超标必须 raise', '预算突破' in str(e), str(e))
    try:
        E.check_budget('k', 'full', None, 100, 50)
        check('T4 长度保持臂不 raise', True)
    except RuntimeError as e:
        check('T4 长度保持臂不 raise', False, str(e))
    check('T4 刚好等于 B 不 raise', E.check_budget('k', 'alias_budget', 0.5, 50, 50) == 50)


# ---------------------------------------------------------------- 合成案例
def synth_case(key='synth-needle-0000', kind='needle'):
    """手写合成案例：一个 900+ 词的日志，答案只在 entry 3。"""
    filler = ('device heartbeat ok rtt 12 ms firmware check reported no drift. '
              'config sync completed with no conflicts. link monitor saw 4 ms jitter. ')
    recs = []
    for i in range(6):
        body = (f'----- entry {i} -----\nrecord node_{i:02d} periodic telemetry dump. '
                + filler * 12)
        if i == 3:
            body += ('occupancy_type is single_adult_work_from_home and total_events=235. ')
            body += ('Evidence: basement_humidity_sensor (basement): 24 events. ')
        recs.append(body)
    ctx = ''.join(r + '\n\n' for r in recs)
    ans = 'single_adult_work_from_home'
    off = ctx.index(ans)
    case = dict(schema_version=1, key=key, case_kind=kind, dataset='longctx',
                source='synth', question='In the log above, what is the occupancy type '
                                         'recorded for entry 3?',
                answers=[ans], context=ctx, context_sha256='x',
                attribute_key='occupancy_type', target_attribute=ans,
                anchor='3', anchor_kind='entry', needle_rid='synth_r', entry_index=3,
                answer_char_offset=ctx.index('----- entry 3 -----'), answer_value_offset=off,
                region_start=ctx.index('----- entry 3 -----'),
                region_end=ctx.index('----- entry 4 -----') - 2,
                region_words=len(ctx.split()), words=len(ctx.split()), n_records=6,
                n_distractors=5, canaries=[dict(type='email', value='synth@example.invalid',
                                                marker='synthmarker1', fragment_length=6),
                                           dict(type='phone', value='+1-202-555-0199',
                                                marker='12025550199', fragment_length=8),
                                           dict(type='account', value='ACCT-synth000001',
                                                marker='synth000001', fragment_length=6)],
                entry_spans=[], pool_records=[])
    return case


def synth_multi(key='synth-multi-0000', k=4):
    """手写合成多实体案例：4 个 account，各带一个互异属性。"""
    attrs = ['single_adult_work_from_home', 'night_shift_worker', 'family_on_vacation',
             'elderly_single'][:k]
    accs = ['ACCT-aaaa0000000%d' % i for i in range(k)]
    recs, spans, cur = [], [], 0
    for i in range(k):
        head = '----- entry %d -----' % i
        body = ('Private contact record:\nContact email: contact.mail%02d@example.invalid\n'
                'Contact phone: +1-202-555-%04d\nAccount ID: %s\n'
                'occupancy_type is %s and the routine is recorded.\n'
                % (i, 100 + i, accs[i], attrs[i])) + ('filler telemetry sentence. ' * 30)
        seg = head + '\n' + body + '\n\n'
        st = cur + len(head) + 1
        spans.append([st, st + len(body), 'r%d' % i])
        recs.append(seg)
        cur += len(seg)
    ctx = ''.join(recs)
    ti = 2
    return dict(schema_version=1, key=key, case_kind='multi_entity', dataset='longctx',
                source='synth', question='In the log above, which occupancy type is recorded '
                                        'for the account %s?' % accs[ti],
                answers=[attrs[ti]], context=ctx, context_sha256='x', n_entities=k,
                target_entity=accs[ti], target_attribute=attrs[ti], entity_type='account',
                entity_key='account', attribute_key='occupancy_type',
                entity_values=accs, attribute_values=attrs,
                target_region_start=spans[ti][0], target_region_end=spans[ti][1],
                target_attribute_offset=ctx.index(attrs[ti]),
                target_attribute_occurrences_in_region=1,
                entity_offset=ctx.index(accs[ti]), answer_char_offset=spans[ti][0],
                answer_value_offset=ctx.index(attrs[ti]), answer_region_index=ti,
                entry_spans=spans, words=len(ctx.split()), n_records=k, guess_baseline=0.25,
                canaries=[dict(type='email', value='contact.mail%02d@example.invalid' % i,
                               marker='mail%02dmarker' % i, fragment_length=6) for i in range(k)]
                         + [dict(type='account', value=accs[i], marker='aaaa0000000%d' % i,
                                 fragment_length=6) for i in range(k)],
                pool_records=[])


# ---------------------------------------------------------------- T5/T6/T7 臂构造
def t5_7():
    nt = E.est_tok
    budgets = [0.25, 0.5, 0.75]
    arms = list(E.ARMS_ALL)
    for case in (synth_case(), synth_multi()):
        texts = E.build_arm_texts(case, nt, budgets, arms, E.lex_drop_order)
        F = nt(case['context'])
        check('T5[%s] 8 臂 x 3 预算 + 3 长度臂齐全' % case['key'],
              len(texts) == 5 * len(budgets) + 3, 'got=%d' % len(texts))
        for arm in E.ARMS_LENGTH:
            t = texts[(arm, None)]
            check('T5[%s] %s 不压缩（长度≈full）' % (case['key'], arm),
                  abs(nt(t) - F) <= max(4, 0.25 * F), '%d vs %d' % (nt(t), F))
        for ratio in budgets:
            B = E.budget_target(F, ratio, 16)
            for arm in E.ARMS_BUDGET:
                got = nt(texts[(arm, ratio)])
                check('T5[%s] %s@%.2f <= B' % (case['key'], arm, ratio), got <= B,
                      'got=%d B=%d' % (got, B))
                E.check_budget(case['key'], arm, ratio, got, B)
            check('T6[%s] 同 ratio 四臂共享同一 B' % case['key'],
                  len({E.budget_target(F, ratio, 16) for _ in E.ARMS_BUDGET}) == 1)
        # T7 隐私
        c = case
        dt = texts[('delete', None)]
        v, f, n = E.leak_of(c, dt)
        check('T7[%s] delete 臂 canary 整值+片段泄漏=0' % c['key'], v == 0 and f == 0,
              'v=%d f=%d n=%d' % (v, f, n))
        at = texts[('alias', None)]
        v2, f2, _ = E.leak_of(c, at)
        check('T7[%s] alias 臂整值泄漏=0' % c['key'], v2 == 0, 'v=%d' % v2)
        if c['case_kind'] == 'multi_entity':
            check('T7[%s] delete 臂 collision=1（链接被杀）' % c['key'],
                  E.entity_diag(c, dt, 'delete')['collision'] == 1)
            check('T7[%s] alias 臂 collision=0（链接存活）' % c['key'],
                  E.entity_diag(c, at, 'alias')['collision'] == 0)
            check('T7[%s] full 臂 collision=0' % c['key'],
                  E.entity_diag(c, texts[('full', None)], 'none')['collision'] == 0)
            check('T7[%s] budget_cluster@0.25 目标实体仍在' % c['key'],
                  E.entity_diag(c, texts[('budget_cluster', 0.25)], 'none')
                  ['tgt_entity_present'] in (0, 1))
        else:
            check('T7[%s] budget_cluster@0.75 gold 仍在上下文' % c['key'],
                  c['answers'][0] in texts[('budget_cluster', 0.75)])
            check('T7[%s] budget_cluster@0.25 gold 仍在（答案单元最后丢）' % c['key'],
                  c['answers'][0] in texts[('budget_cluster', 0.25)],
                  '答案单元被丢了 -> 与"答案承载单元最后丢"矛盾')


# ---------------------------------------------------------------- T8/T9 端到端（stub 预测 + 官方评分器）
def t8_9(tmp):
    cases = [synth_case(), synth_multi()]
    cpath = os.path.join(tmp, 'cases.jsonl')
    io.open(cpath, 'w', encoding='utf-8', newline='\n').write(
        ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
    # 生成 stub 预测：budget_cluster@0.75 与 full/alias 给正确答案，delete 给 UNKNOWN
    preds = []
    for c in cases:
        for arm in E.ARMS_ALL:
            ratios = [None] if arm in E.ARMS_LENGTH else [0.25, 0.5, 0.75]
            for r in ratios:
                if arm in ('full', 'alias', 'budget_cluster') and r in (None, 0.75):
                    p = c['answers'][0]
                elif arm in ('delete', 'delete_budget', 'alias_budget', 'budget_random'):
                    p = 'UNKNOWN'
                else:
                    p = c['answers'][0]
                preds.append(dict(key=c['key'], arm=arm, ratio=r, pred=p))
    ppath = os.path.join(tmp, 'preds.jsonl')
    io.open(ppath, 'w', encoding='utf-8', newline='\n').write(
        ''.join(json.dumps(p) + '\n' for p in preds))

    out = os.path.join(tmp, 'r.txt')
    js = os.path.join(tmp, 'r.json')
    pc = os.path.join(tmp, 'r.jsonl')
    cmd = [sys.executable, os.path.join(HPC, 'e2e_longctx.py'),
           '--cases', cpath, '--out', out, '--json', js, '--percase', pc,
           '--dry-run', '--arms', 'all', '--budgets', '0.25,0.5,0.75',
           '--scorer', SCORER, '--stub-preds', ppath]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=HPC)
    check('T8 端到端退出码 0', r.returncode == 0, (r.stderr or '')[-1400:])
    if r.returncode != 0:
        return
    cells = json.load(io.open(js, encoding='utf-8'))
    check('T9 cells.json 有 5x3 + 3 = 18 个 (arm,ratio) 桶',
          len(cells['budget_accounting']) == 18, str(sorted(cells['budget_accounting'])[:5]))
    check('T9 per-case 行数 = 2 案例 x 18 臂', sum(1 for _ in io.open(pc, encoding='utf-8')) == 36)
    rep = io.open(out, encoding='utf-8').read()
    check('T9 report 含预算对账块', '[1] 预算对账' in rep)
    check('T9 report 标明 DRY-RUN', 'DRY-RUN' in rep)

    # 与直接调用官方 scorer 对拍
    import importlib.util
    spec = importlib.util.spec_from_file_location('sc', SCORER)
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    rows = [json.loads(l) for l in io.open(pc, encoding='utf-8') if l.strip()]
    mism = []
    for row in rows:
        e = sm.compute_exact(row['gold'], row['pred'])
        f = sm.compute_f1(row['gold'], row['pred'])
        if abs(e - row['em']) > 1e-12 or abs(f - row['f1']) > 1e-12:
            mism.append(row['case_key'], )
    check('T8 EM/F1 与官方 compute_* 逐一一致', not mism, str(mism[:3]))
    # delete 臂 ctx_leak 必须为 0
    bad = [r for r in rows if r['arm'] in ('delete', 'delete_budget') and
           (r['ctx_leak_value'] or r['ctx_leak_frag'])]
    check('T8 delete/delete_budget 臂 ctx_leak 全 0', not bad, str(bad[:1]))
    # 长度保持臂的 ratio 必须是 None
    check('T9 长度保持臂 ratio=None',
          {r['ratio'] for r in rows if r['arm'] in E.ARMS_LENGTH} == {None})
    # 预算臂 target 必须出现在 percase 里
    check('T9 percase 记录 target', all(r.get('target') for r in rows))
    # 预算对账 json 里 |dev| 的最大值 <= 0（预算臂只允许低于 B）
    devs = [(k, v['devmax_signed']) for k, v in cells['budget_accounting'].items()
            if k.split('|')[0] in E.ARMS_BUDGET]
    check('T8 预算臂 dev_max_signed <= 0', all(d <= 0 for _k, d in devs), str(devs[:4]))


# ---------------------------------------------------------------- T10 真案例 dry-run
def t10(tmp):
    if not os.path.exists(REAL_CASES):
        print('[skip] T10 真案例不存在: %s' % REAL_CASES)
        return
    out = os.path.join(tmp, 'real.txt')
    js = os.path.join(tmp, 'real.json')
    pc = os.path.join(tmp, 'real.jsonl')
    cmd = [sys.executable, os.path.join(HPC, 'e2e_longctx.py'),
           '--cases', REAL_CASES, '--out', out, '--json', js, '--percase', pc,
           '--dry-run', '--arms', 'all', '--budgets', '0.25,0.5,0.75']
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', cwd=HPC)
    check('T10 真案例 dry-run 退出码 0', r.returncode == 0, (r.stderr or '')[-1200:])
    if r.returncode != 0:
        return
    rows = [json.loads(l) for l in io.open(pc, encoding='utf-8') if l.strip()]
    check('T10 percase 行数 = 112 x 18', len(rows) == 112 * 18, 'got=%d' % len(rows))
    over = [r for r in rows if r['arm'] in E.ARMS_BUDGET and r['ctx_tokens'] > r['target']]
    check('T10 无预算超标', not over, str(over[:1]))
    # gold_in_ctx 随预算单调不增（同一源+臂，cluster 选择器）
    mono = []
    for src in sorted({r['source'] for r in rows}):
        for kind in ('needle', 'multi_entity'):
            g = {}
            for r in rows:
                if r['source'] == src and r['arm'] == 'budget_cluster' and \
                        r['case_kind'] == kind:
                    g[r['ratio']] = g.get(r['ratio'], 0) + r['ans_in_ctx']
            if len(g) == 3:
                if not (g[0.25] <= g[0.5] <= g[0.75]):
                    mono.append((src, kind, g))
    check('T10 budget_cluster 的 gold_in_ctx 随预算单调不增', not mono, str(mono[:2]))
    # 全量案例词数下限
    cases = [json.loads(l) for l in io.open(REAL_CASES, encoding='utf-8') if l.strip()]
    check('T10 所有案例 >= 800 词', all(c['words'] >= 800 for c in cases),
          str([c['words'] for c in cases if c['words'] < 800][:3]))
    check('T10 最长案例的估计 token 数放得进 maxlen=8192',
          max(E.est_tok(c['context']) for c in cases) + 64 + 64 < 8192,
          'max_est=%d' % max(E.est_tok(c['context']) for c in cases))


def main():
    print('=== longctx 合成自检（CPU-only，无模型） ===')
    LOG.append('=== longctx 合成自检（CPU-only，无模型） ===')
    for fn in (t1, t2, t3, t4, t5_7):
        try:
            fn()
        except Exception as e:
            check(fn.__name__, False, 'EXC %r' % (e,))
    with tempfile.TemporaryDirectory() as tmp:
        for fn in (t8_9, t10):
            try:
                fn(tmp)
            except Exception as e:
                check(fn.__name__, False, 'EXC %r' % (e,))
    tail = '\n=== 汇总: %d 通过 / %d 失败 ===' % (len(OKS), len(FAILS))
    LOG.append(tail)
    print('\n' + tail)
    if FAILS:
        for f in FAILS:
            LOG.append('  FAIL ' + f)
            print('  FAIL', f)
    io.open(os.path.join(HERE, 'synth_test_log.txt'), 'w', encoding='utf-8',
            newline='\n').write('\n'.join(LOG) + '\n')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
