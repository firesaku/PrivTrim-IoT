"""从 e2e_longctx_r5full.py 生成 e2e_quad.py（两个实验共用）。

新增能力：
  --no-oracle   关闭金标答案 oracle（ans_of 恒为 False）→ 实验 A
  quad4         真正的四象限路由：按 (敏感, 相关) 分别施加 别名/删除/保留/按预算丢
                relevance 只用「问题点名的实体所在 entry」，不用金标 → 实验 B
  quad4_del     同上，但 (敏感,相关) 也用删除而非别名（检验别名是否必要）
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'longctx' / 'e2e_longctx_r5full.py'
DST = SRC.parent / 'e2e_quad.py'

src = SRC.read_text(encoding='utf-8')
orig = src

# ---------------------------------------------------------------- 1) ARMS
old_arms = """ARMS_ALL = ['full', 'delete', 'alias',
            'budget_cluster', 'budget_cluster_link', 'budget_random',
            'delete_budget', 'alias_budget', 'alias_budget_link']
ARMS_BUDGET = ['budget_cluster', 'budget_cluster_link', 'budget_random',
               'delete_budget', 'alias_budget', 'alias_budget_link']"""
new_arms = """ARMS_ALL = ['full', 'delete', 'alias',
            'budget_cluster', 'budget_cluster_link', 'budget_random',
            'delete_budget', 'alias_budget', 'alias_budget_link',
            'quad4', 'quad4_del']
ARMS_BUDGET = ['budget_cluster', 'budget_cluster_link', 'budget_random',
               'delete_budget', 'alias_budget', 'alias_budget_link',
               'quad4', 'quad4_del']

# --- 实验开关 ---
NO_ORACLE = False      # True: ans_of 恒 False（关闭金标答案 oracle）"""
assert old_arms in src, 'ARMS block not found'
src = src.replace(old_arms, new_arms, 1)

old_alias = "ARMS_ALIAS = ['alias', 'alias_budget', 'alias_budget_link']"
new_alias = "ARMS_ALIAS = ['alias', 'alias_budget', 'alias_budget_link', 'quad4']"
assert old_alias in src, 'ARMS_ALIAS not found'
src = src.replace(old_alias, new_alias, 1)

# ---------------------------------------------------------------- 2b) arm_style
# quad4 的锚点是别名形式、quad4_del 是删除形式；不登记会导致
# entity_diag 用原始串去匹配别名文本，ent 列恒为 0（测量错误）。
old_style = """def arm_style(arm):
    if arm in ('delete', 'delete_budget'):
        return 'delete'
    if arm in ('alias', 'alias_budget', 'alias_budget_link'):
        return 'alias'
    return 'none'"""
new_style = """def arm_style(arm):
    if arm in ('delete', 'delete_budget', 'quad4_del'):
        return 'delete'
    if arm in ('alias', 'alias_budget', 'alias_budget_link', 'quad4'):
        return 'alias'
    return 'none'"""
assert old_style in src, 'arm_style not found'
src = src.replace(old_style, new_style, 1)

# ---------------------------------------------------------------- 2) ans_of
old_ans = """    def ans_of(units):
        return [any(nz(g) and nz(g) in nz(u) for g in golds) for u in units]"""
new_ans = """    def ans_of(units):
        # 实验 A：--no-oracle 时完全不看金标答案
        if NO_ORACLE:
            return [False] * len(units)
        return [any(nz(g) and nz(g) in nz(u) for g in golds) for u in units]"""
assert old_ans in src, 'ans_of not found'
src = src.replace(old_ans, new_ans, 1)

# ---------------------------------------------------------------- 3) build_quad4
anchor = "    F = nt(full)\n    texts = {}"
assert anchor in src, 'loop anchor not found'

quad_code = '''    # ---------------- 四象限路由 ----------------
    QUAD_WINDOW = 2       # 无记录块结构时，锚点前后各取几句

    def _entry_index(units):
        """每个单元所属的 entry 序号；----- entry N ----- 作为分界。
        若全文没有标记，则全部为 -1。"""
        idx, cur = [], -1
        for u in units:
            if re.match(r'^-{3,}\\s*entry\\s+\\d+', u.strip(), re.I):
                cur += 1
                idx.append(cur)
            else:
                idx.append(cur)
        return idx

    def _is_sens(u):
        nu = nz(u)
        return any(nz(c['value']) in nu or
                   (c.get('marker') and nz(c['marker']) in nu) for c in cans)

    def build_quad4(redact_rel, B, order_f):
        """按象限施加不同操作。

        相关区域两种确定模式：
          (1) 文本有 ----- entry N ----- 结构 -> 锚点所在 entry 的全部单元
          (2) 无结构（如 LongBench 段落）      -> 锚点前后各 QUAD_WINDOW 句

        (sens, rel)   -> 别名（redact_rel=True 时改为删除）
        (sens, irrel) -> 删除
        (plain, rel)  -> 保留
        (plain, irrel)-> 按冗余度填充到预算
        """
        us = split_units(full)
        ents = _entry_index(us)
        akeys = protect_keys()
        tgt_units = [i for i, u in enumerate(us) if any(k and k in u for k in akeys)]

        has_marker = any(re.match(r'^-{3,}\\s*entry\\s+\\d+', u.strip(), re.I) for u in us)
        if has_marker:
            tgt_ents = set(ents[i] for i in tgt_units)
            rel_set = set(i for i in range(len(us)) if ents[i] in tgt_ents)
        else:
            rel_set = set()
            for i in tgt_units:
                for j in range(max(0, i - QUAD_WINDOW),
                               min(len(us), i + QUAD_WINDOW + 1)):
                    rel_set.add(j)
        if not rel_set:
            rel_set = set(tgt_units)      # 兜底：至少保住锚点本身

        al = v4_router.Aliaser('alias')
        al.preseed(cans)
        for _c in cans:
            al.token(_c)

        out, forced, cand = [], [], []
        for i, u in enumerate(us):
            s = _is_sens(u)
            r = i in rel_set
            if s and r:
                out.append(privcheck.redact(u) if redact_rel
                           else v4_router.apply_replacements(u, cans, al))
                forced.append(i)
            elif s and not r:
                out.append('')            # 敏感且不相关 -> 删除
            elif r:
                out.append(u)
                forced.append(i)
            else:
                out.append(u)
                cand.append(i)

        # 强制集合若本身超预算，按与锚点距离裁剪（锚点优先），
        # 被裁掉的降级为可丢弃 —— 保证 check_budget 不会因超支抛错。
        if tgt_units:
            prio = sorted(forced, key=lambda i: (min(abs(i - j) for j in tgt_units), i))
        else:
            prio = sorted(forced)
        kept, acc = [], 0
        for i in prio:
            c = nt(out[i])
            if acc + c <= B:
                kept.append(i)
                acc += c
            elif out[i]:
                cand.append(i)
        forced = kept

        flags = [False] * len(out)
        for i in forced:
            flags[i] = True
        drop_order = order_f([x if x else '.' for x in out], flags)

        total = sum(nt(out[i]) for i in forced)
        chosen = list(forced)
        # 实测对比：此处按 drop_order 正序加入 = 优先保留**最冗余**的单元。
        # 与 drop_to_budget 的丢弃方向相反，但实测更优（见 docs/27 §27.12）：
        # 因为答案承载单元往往是最冗余的（答案值在大量干扰记录中重复），
        # 保留最冗余恰好保住答案。反向填充在 Type C 上从 0.6733 崩到 0.0167。
        for i in drop_order:
            if i in cand:
                c = nt(out[i])
                if total + c <= B:
                    chosen.append(i)
                    total += c
        chosen.sort()

        # 精确修剪：' '.join 会插入分隔符，真实 token 数可能大于逐单元之和
        forced_set = set(forced)
        while True:
            txt = ' '.join(out[i] for i in chosen if out[i])
            if nt(txt) <= B:
                break
            dropped = False
            for i in reversed(chosen):
                if i not in forced_set:
                    chosen.remove(i)
                    dropped = True
                    break
            if not dropped:
                break
        return ' '.join(out[i] for i in chosen if out[i])

'''
src = src.replace(anchor, quad_code + anchor, 1)

# ---------------------------------------------------------------- 4) arms 分支
old_branch = """            elif arm == 'alias_budget_link':
                t = drop_to_budget(ali_units, B, nt, order_al)
            else:
                continue"""
new_branch = """            elif arm == 'alias_budget_link':
                t = drop_to_budget(ali_units, B, nt, order_al)
            elif arm == 'quad4':
                t = build_quad4(False, B, order_fn)
            elif arm == 'quad4_del':
                t = build_quad4(True, B, order_fn)
            else:
                continue"""
assert old_branch in src, 'arm branch not found'
src = src.replace(old_branch, new_branch, 1)

# ---------------------------------------------------------------- 5) CLI
m2 = re.search(r"([ \t]*)a = ap\.parse_args\(\)\n", src)
assert m2, 'parse_args not found'
ind = m2.group(1)
before = m2.group(0)
cli_new = (ind + "ap.add_argument('--no-oracle', action='store_true',\n"
           + ind + "                help='disable the gold-answer oracle (exp A)')\n"
           + before
           + ind + "NO_ORACLE = bool(getattr(a, 'no_oracle', False))\n")
src = src.replace(before, cli_new, 1)

# 需要 global 声明
src = src.replace("def main():", "def main():\n    global NO_ORACLE", 1)

DST.write_text(src, encoding='utf-8', newline='\n')
print('wrote %s' % DST)
print('  size: %d -> %d bytes' % (len(orig), len(src)))
print('  added: --no-oracle, quad4, quad4_del')

# 语法自检
try:
    compile(src, str(DST), 'exec')
    print('  py_compile: OK')
except SyntaxError as e:
    print('  py_compile: FAIL -> %s' % e)
