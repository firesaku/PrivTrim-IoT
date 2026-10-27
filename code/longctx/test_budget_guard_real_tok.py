"""预算守卫回归 + 性质测试（**真实 Qwen tokenizer**）。

为什么需要这个文件：`test_longctx_synthetic.py` 的 T3 用
`nt = len(re.findall(r'[A-Za-z0-9]+', s))` 当 tokenizer —— 这个函数对空格拼接是
**可加的**（nt(" ".join(units)) == sum(nt(u))），所以它**在原理上不可能**暴露
"逐单元 token 之和不是拼接文本上界"这个缺陷。T10 的 dry-run 用 est_tok（同样是
可加的），也一样瞎。于是合成套件全绿、真实跑却在 longctx-toolmem-needle-0003 上
被 check_budget 拦下（got=5378 B=5377，超 1 个 token）。

本文件用**真实 tokenizer**（唯一会做 check_budget 断言的 nt）重放该案例并做全量
预检，作为启动 4 小时 GPU 作业前的硬闸门。

R1 案例复现：在真实失败案例上，逐臂逐预算比对"修复前算法"与修复后实现。
R2 全量预检：112 案例 x 全部预算臂 x 3 预算，断言 nt(发射文本) <= B。
R3 边界穷举性质测试：对随机单元集，**穷举求和式丢弃循环会停下来的每一个 B**，
   这是唯一能必然命中"和 <= B 但拼接 > B"的构造；同时统计旧算法违规数。
R4 单调性：budget_cluster 下 gold 存活随预算**单调不减**（预算越大留得越多；
   注意 test_longctx_synthetic.py 的 T10 标签写作"单调不增"，其代码实为
   g[0.25] <= g[0.5] <= g[0.75]，以代码为准）。

用法（集群上，RD 为工作目录）：
    PYTHONIOENCODING=utf-8 "$PY" test_budget_guard_real_tok.py
"""
import hashlib
import importlib.util
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_harness():
    """按路径加载 harness —— 集群上它叫 e2e_longctx_r2.py（不覆盖旧版 29858B）。"""
    cands = []
    if os.environ.get('HARNESS'):
        cands.append(os.environ['HARNESS'])
    cands += [os.path.join(HERE, 'e2e_longctx_r2.py'),
              os.path.join(HERE, 'e2e_longctx.py'),
              os.path.join(HERE, os.pardir, 'hpc', 'e2e_longctx.py')]
    for p in cands:
        p = os.path.abspath(p)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location('e2e_longctx_under_test', p)
            m = importlib.util.module_from_spec(spec)
            sys.modules['e2e_longctx_under_test'] = m
            spec.loader.exec_module(m)
            print('harness=%s' % p, flush=True)
            return m
    raise SystemExit('找不到 harness；试过: %s' % cands)


E = load_harness()
from transformers import AutoTokenizer                           # noqa: E402

B_ROOT = ('/path/to/100065732/privtrim_iot_hpc_20261005/'
          'PrivTrim_IoT_HPC_20261005')
LM = os.environ.get('LM') or os.path.join(
    B_ROOT, 'models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/'
            'a09a35458c702b33eeacc393d103063234e8bc28')
CASES = os.environ.get('CASES') or os.path.join(HERE, 'longctx_cases.jsonl')
SEED = 20261015
BUDGETS = [0.25, 0.5, 0.75]
MINTOK = 16
TARGET_KEY = 'longctx-toolmem-needle-0003'

FAILS = []


def check(name, cond, detail=''):
    print('[%s] %s%s' % ('ok  ' if cond else 'FAIL', name,
                         ('  <<< ' + str(detail)[:300]) if not cond else ''),
          flush=True)
    if not cond:
        FAILS.append(name)


def old_drop_to_budget(units, budget, nt, order):
    """修复前的实现（逐单元 token 数之和当记账量）—— 只用于证明缺陷真实存在。"""
    utok = [nt(u) for u in units]
    total = sum(utok)
    keep = [True] * len(units)
    for i in order:
        if total <= budget:
            break
        keep[i] = False
        total -= utok[i]
    return ' '.join(u for k, u in enumerate(units) if keep[k])


def rand_order(key, units):
    rng = random.Random(int(hashlib.sha256(
        ('%d|%s|rand' % (SEED, key)).encode()).hexdigest()[:16], 16))
    o = list(range(len(units)))
    rng.shuffle(o)
    return o


def main():
    print('LM=%s' % LM, flush=True)
    tok = AutoTokenizer.from_pretrained(LM, local_files_only=True)
    nt = lambda s: len(tok.encode(s, add_special_tokens=False))   # noqa: E731
    print('tokenizer vocab=%d' % tok.vocab_size, flush=True)

    cases = E.load_cases(CASES)
    check('R0 载入真实案例', len(cases) == 112, 'n=%d' % len(cases))
    by_key = {c['key']: c for c in cases}
    overall_old_viol = 0

    # ---------------- R1 真实失败案例：逐臂逐预算比对 ----------------------
    if TARGET_KEY not in by_key:
        check('R1 找到失败案例 %s' % TARGET_KEY, False, '不在案例集里')
    else:
        c = by_key[TARGET_KEY]
        ru = E.split_units(c['context'])
        F = nt(c['context'])
        arms_o = [('budget_cluster', E.lex_drop_order(
                       ru, [any(E.privcheck.norm(g) and E.privcheck.norm(g)
                                in E.privcheck.norm(u)
                                for g in E.gold_texts(c)) for u in ru])),
                  ('budget_random', rand_order(c['key'], ru))]
        for ratio in BUDGETS:
            B = E.budget_target(F, ratio, MINTOK)
            for arm, order in arms_o:
                o_got = nt(old_drop_to_budget(ru, B, nt, order))
                n_got = nt(E.drop_to_budget(ru, B, nt, order))
                if o_got > B:
                    overall_old_viol += 1
                print('R1 %-15s ratio=%.2f F=%d B=%d  old=%d(%+d)  new=%d(%+d)'
                      % (arm, ratio, F, B, o_got, o_got - B, n_got, n_got - B),
                      flush=True)
                check('R1 %s@%.2f 修复后不超标' % (arm, ratio), n_got <= B,
                      'got=%d B=%d' % (n_got, B))

    # ---------------- R2 全量预检：真实案例 x 全部预算臂 x 3 预算 -----------
    bad = []
    nrows = 0
    for ci, c in enumerate(cases):
        texts = E.build_arm_texts(c, nt, BUDGETS, E.ARMS_ALL,
                                  E.lex_drop_order, SEED)
        F = nt(c['context'])
        for (arm, ratio) in texts:
            nrows += 1
            if ratio is None:
                continue
            B = E.budget_target(F, ratio, MINTOK)
            got = nt(texts[(arm, ratio)])
            if got > B:
                bad.append((c['key'], arm, ratio, got, B))
        if (ci + 1) % 20 == 0:
            print('R2 progress %d/%d rows=%d bad=%d'
                  % (ci + 1, len(cases), nrows, len(bad)), flush=True)
    print('R2 预检行数=%d  违规=%d' % (nrows, len(bad)), flush=True)
    check('R2 全量预检无预算超标', not bad, bad[:5])
    check('R2 行数=112x18', nrows == 112 * 18, 'n=%d' % nrows)

    # ---------------- R3 边界穷举性质测试 ---------------------------------
    sents = []
    for c in cases[:12]:
        us = E.split_units(c['context'])
        sents.extend([u for u in us if len(u) > 20][:30])
        # 记账间隙来自"单元首 token 与前导空格合并方式不同"，**短单元**才是重灾区
        sents.extend([u for u in us if 2 <= len(u) <= 20][:12])
    rng = random.Random(20261015)
    old_viol = new_viol = trials = 0
    max_gap = 0
    for _t in range(120):
        k = rng.randint(2, 40)
        units = [rng.choice(sents) for _ in range(k)]
        order = list(range(k))
        rng.shuffle(order)
        utok = [nt(u) for u in units]
        keep = [True] * k
        total = sum(utok)
        for step in range(k + 1):
            sel = ' '.join(u for i, u in enumerate(units) if keep[i])
            j = nt(sel)
            max_gap = max(max_gap, j - total)
            # B = total 正好让"和式循环"在这一步停下 —— 缺陷的唯一边界构造
            B = total
            if j > B:
                old_viol += 1
            trials += 1
            if nt(E.drop_to_budget(units, B, nt, order)) > B:
                new_viol += 1
            if step < k:
                keep[order[step]] = False
                total -= utok[order[step]]
            if total <= 0:
                break
    print('R3 boundary_trials=%d  max_accounting_gap=%d  '
          'old_violations=%d  new_violations=%d'
          % (trials, max_gap, old_viol, new_viol), flush=True)
    check('R3 新实现零违规（边界穷举）', new_viol == 0, 'new_viol=%d' % new_viol)
    check('R3 旧实现在边界上确有违规（缺陷可复现）', old_viol > 0,
          'old_viol=%d max_gap=%d' % (old_viol, max_gap))
    overall_old_viol += old_viol

    # ---------------- R4 单调性：gold 存活随预算单调不减 -------------------
    mono_bad = []
    for c in cases:
        golds = [g for g in E.gold_texts(c) if E.privcheck.norm(g)]
        prev = None
        for ratio in sorted(BUDGETS):
            texts = E.build_arm_texts(c, nt, [ratio], ['budget_cluster'],
                                      E.lex_drop_order, SEED)
            nz = E.privcheck.norm
            alive = sum(1 for g in golds if nz(g) in nz(texts[('budget_cluster', ratio)]))
            if prev is not None and alive < prev:
                mono_bad.append((c['key'], ratio, prev, alive))
            prev = alive
    check('R4 gold 存活随预算单调不减', not mono_bad, mono_bad[:5])

    print('', flush=True)
    check('R1+R3 汇总：修复前算法至少有一处超标（回归确实钉住了缺陷）',
          overall_old_viol > 0, 'overall_old_viol=%d' % overall_old_viol)
    print('BUDGET_GUARD_RESULT  total_checks_failed=%d' % len(FAILS), flush=True)
    if FAILS:
        for f in FAILS:
            print('  FAILED: %s' % f, flush=True)
        return 1
    print('BUDGET_GUARD_ALL_PASS', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
