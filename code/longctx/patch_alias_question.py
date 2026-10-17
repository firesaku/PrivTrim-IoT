"""给 e2e_longctx_r4.py 加 `--alias-question`：用同一份别名映射改写问题里的实体。

问题（由作业 10602422 的产物定位并验证）
----------------------------------------
`alias_budget_link` 在 112 例上值泄漏 0.00%、实体存活 tent=1.000、collision=0，
但 F1 只有 0.5397，而同样开启锚点保护的原始串臂是 0.7182。差别在问题一侧：

    问题：        ... for the account ACCT-5ee052b431dc?
    上下文（别名）：... [ACCOUNT-F] ...

模型无法把问题里的原始串对应到上下文里的别名，于是答 UNKNOWN。
证据：两臂 tent 都是 112/112，但 UNKNOWN 次数是 17（原始串）vs 43（别名），差 26 例。

因此：**稳定别名是"问题 + 上下文"的联合变换。** 只改上下文等于把被查询实体从模型的
可指称范围里删掉 —— 隐私保住了，任务同时被毁，而"串还在不在"类指标（tent/gold_in_ctx）
**看不出来**。

为什么直接用新 Aliaser 也安全
-----------------------------
`v4_router.apply_replacements` 对 `canaries` 列表**逐个无条件**调用 `aliaser.token(c)`
（两个循环都遍历全部 canaries），字母按**列表顺序**分配，与文本内容无关，且
`Aliaser.token` 按 value 缓存。所以只要传入同一份 canaries 列表，任何新 Aliaser 都会
产生完全相同的字母分配 —— 问题与上下文天然一致。

**但这个"天然一致"必须验，不能假设**：一旦哪天实现改成"按文本首见分配"，问题与上下文
的字母就会错位，`[ACCOUNT-F]` 两边指向不同实体，结果会比不改更糟。所以补丁同时加入
跑前断言，见 `check_question_alias`。
"""
import io
import re
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'e2e_longctx_r4.py'
src = io.open(TARGET, encoding='utf-8').read()
orig = src

# ---------------------------------------------------------------- A. 臂集合常量
A_ANCHOR = "ARMS_LENGTH = ['full', 'delete', 'alias']\n"
assert src.count(A_ANCHOR) == 1, 'anchor A count=%d' % src.count(A_ANCHOR)
src = src.replace(A_ANCHOR, A_ANCHOR + (
    "# 会改写实体表面串的臂：只有这些臂才谈得上「问题与上下文必须一致地被别名化」\n"
    "ARMS_ALIAS = ['alias', 'alias_budget', 'alias_budget_link']\n"), 1)

# ---------------------------------------------------------------- B. argparse
B_ANCHOR = ("    ap.add_argument('--ext-ratio', type=float, default=0.5,\n"
            "                    help='外部臂按此 ratio 记账。已发布 compress.py 的预算是 '\n"
            "                         'floor(0.5*n)，故默认 0.5')\n")
assert src.count(B_ANCHOR) == 1, 'anchor B count=%d' % src.count(B_ANCHOR)
src = src.replace(B_ANCHOR, B_ANCHOR + (
    "    ap.add_argument('--alias-question', action='store_true',\n"
    "                    help='对别名语义臂，用**同一份**别名映射同步改写问题里的实体提及。'\n"
    "                         '不开则问题保留原始串，此时别名臂的链接断在问题一侧。')\n"), 1)

# ---------------------------------------------------------------- C. 函数
C_ANCHOR = "def make_prompt(q, ctx):\n"
assert src.count(C_ANCHOR) == 1, 'anchor C count=%d' % src.count(C_ANCHOR)
C_NEW = '''def question_for_arm(case, arm, alias_question):
    """返回该臂该案例实际要送进提示词的问题字符串。

    只在 --alias-question 开启、且该臂本身改写了实体表面串时才改写；其它情况原样返回，
    保证与历史结果逐字可比（这也是回归验证的前提）。

    安全性论证见模块补丁说明：apply_replacements 按 canaries 列表顺序无条件分配字母，
    故同一份 canaries 下新旧 Aliaser 的映射完全相同。
    """
    q = case.get('question') or ''
    if not alias_question or arm not in ARMS_ALIAS:
        return q
    return v4_router.apply_replacements(q, case.get('canaries') or [],
                                        v4_router.Aliaser('alias'))


def check_question_alias(cases, keys, arm_txt, alias_question):
    """跑前断言：问题里的别名必须与上下文里的别名指向同一实体。

    分两层，**混在一起会把合法结果误判成 bug**：

    (1) 映射正确性（必须恒成立，失败即 raise）
        - 问题里不再有目标实体的原始串；
        - 问题里出现该实体的别名 token；
        - 该 token 必须出现在**未压缩的别名化全文** `alias_text(context)` 里。
          这一条只查映射是否一致，与压缩无关。
    (2) 投递率（实验结果是变量，只统计不报错）
        - 该 token 是否在该臂**压缩后的**文本里。预算臂本来就会把实体所在的单元丢掉
          （alias_budget@0.5 的 tent 只有 0.705），所以这里不达标是**合法的实验观察**，
          不是映射错误。第一版把它写进断言，280 处里误报 101 处 —— 记下来。

    返回 (n_checked, n_skipped, bad_mapping, delivered, undelivered)。
    """
    if not alias_question:
        return 0, 0, [], 0, 0
    n_checked = n_skipped = 0
    bad = []
    delivered = undelivered = 0
    ali_cache = {}
    for i, (ci, arm, ratio) in enumerate(keys):
        if arm not in ARMS_ALIAS:
            continue
        c = cases[ci]
        raw = c.get('target_entity') or ''
        q0 = c.get('question') or ''
        if not raw or raw not in q0:
            n_skipped += 1
            continue
        n_checked += 1
        cans = c.get('canaries') or []
        q1 = question_for_arm(c, arm, True)
        al = v4_router.Aliaser('alias')
        for _c in cans:
            al.token(_c)
        cmap = {x['value']: x for x in cans if x['type'] == 'account'}
        tok = al.token(cmap[raw]) if raw in cmap else None
        if tok is None:
            bad.append((c['key'], arm, 'no canary for target_entity'))
            continue
        if raw in q1:
            bad.append((c['key'], arm, 'raw entity still in question'))
        if tok not in q1:
            bad.append((c['key'], arm, 'alias token missing from question'))
        key = c['key']
        if key not in ali_cache:
            ali_cache[key] = alias_text(c['context'], cans)
        if tok not in ali_cache[key]:
            bad.append((c['key'], arm, 'MAPPING MISMATCH: alias not in uncompressed aliased passage'))
        if tok in arm_txt[i]:
            delivered += 1
        else:
            undelivered += 1
    return n_checked, n_skipped, bad, delivered, undelivered


'''
src = src.replace(C_ANCHOR, C_NEW + C_ANCHOR, 1)

# ---------------------------------------------------------------- D. 提示词构造
# 前导 \n 是必需的：ext 注入块里也有同名的 16 空格缩进调用，
# 若不带 \n，12 空格前缀会成为 16 空格行的子串，导致 count==2（实测踩过）。
D_ANCHOR = "\n            prompts.append(make_prompt(c['question'], txt))\n"
assert src.count(D_ANCHOR) == 1, 'anchor D count=%d' % src.count(D_ANCHOR)
src = src.replace(D_ANCHOR,
                  "\n            prompts.append(make_prompt(\n"
                  "                question_for_arm(c, arm, a.alias_question), txt))\n", 1)

# ---------------------------------------------------------------- E. 跑前断言
E_ANCHOR = "    # ---- 预测 ----\n"
assert src.count(E_ANCHOR) == 1, 'anchor E count=%d' % src.count(E_ANCHOR)
E_NEW = '''    # ---- 跑前断言：问题别名与上下文别名必须指向同一实体 ----
    n_qalias, n_qskip, qbad, n_deliv, n_undeliv = check_question_alias(
        cases, keys, arm_txt, a.alias_question)
    if a.alias_question:
        log('问题同步别名：映射检查 %d 处（跳过 %d 处问题未提及目标实体）；映射不一致 %d 处'
            % (n_qalias, n_qskip, len(qbad)))
        log('问题同步别名：别名 token 在压缩后文本中的投递率 %d/%d = %.1f%%'
            '（未投递 = 该臂把实体所在单元丢了，属实验观察，不是映射错误）'
            % (n_deliv, n_deliv + n_undeliv,
               100.0 * n_deliv / max(1, n_deliv + n_undeliv)))
        if qbad:
            raise RuntimeError('问题别名与上下文别名映射不一致（会导致两边指向不同实体）: %s'
                               % qbad[:5])

    # ---- 预测 ----
'''
src = src.replace(E_ANCHOR, E_NEW, 1)

assert src != orig, 'nothing changed'
io.open(TARGET, 'w', encoding='utf-8', newline='\n').write(src)
print('PATCHED %s: %d -> %d bytes (%+d)'
      % (TARGET, len(orig.encode('utf-8')), len(src.encode('utf-8')),
         len(src.encode('utf-8')) - len(orig.encode('utf-8'))))
