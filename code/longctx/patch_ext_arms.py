"""给 e2e_longctx_r4.py 加 `--ext-arms`：把已发布压缩器的输出注入同一条打分路径。

为什么需要
----------
目前所有数字都是我们自己内部臂互比。审稿人会问："你说冗余度压缩在多实体案例上
系统性失效，这是领域问题还是你自己实现的产物？" 唯一能回答的办法是把**已发布的
压缩方法**放到同一个基准、同一个模型、同一条生成+打分+泄漏诊断路径上跑。

为什么是"注入文本"而不是"接入它们的评估代码"
--------------------------------------------
只用它们的输出文本，中间量一概不用。这样唯一的变量是"这段上下文是怎么来的"，
生成、评分、泄漏诊断、实体诊断全部逐字相同。口径一致是这类对比成立的前提。

patch 的五处改动
----------------
A. 新增 `--ext-arms` / `--ext-ratio` 两个参数
B. 建臂循环之后、预测之前，注入 `ext_<method>` 臂（记账口径与内部预算臂完全一致）
C. 两处作用域过滤：`ext_*` 是**等预算**臂，必须进"同预算块"，不能掉进"长度保持块"
D. `[1]` 表的 READ 说明补一行，交代 ext 臂的预算口径差异
E. `ext_methods` / `ext_arm_names` 在无 `--ext-arms` 时也必须存在（报告段会引用）
"""
import io
import re
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else 'e2e_longctx_r4.py'
src = io.open(TARGET, encoding='utf-8').read()
orig = src

# ---------------------------------------------------------------- A. argparse
A_ANCHOR = "    ap.add_argument('--dump-arms', default='', help='把建好的臂文本写出来供人工核对')\n"
assert src.count(A_ANCHOR) == 1, 'anchor A count=%d' % src.count(A_ANCHOR)
A_NEW = A_ANCHOR + (
    "    ap.add_argument('--ext-arms', default='',\n"
    "                    help='外部基线臂 JSONL: {key, method, text}；把已发布压缩器的输出'\n"
    "                         '原样注入同一条生成+打分路径')\n"
    "    ap.add_argument('--ext-ratio', type=float, default=0.5,\n"
    "                    help='外部臂按此 ratio 记账。已发布 compress.py 的预算是 '\n"
    "                         'floor(0.5*n)，故默认 0.5')\n"
)
src = src.replace(A_ANCHOR, A_NEW, 1)

# ---------------------------------------------------------------- B. 注入块
B_ANCHOR = "    # ---- 预测 ----\n"
assert src.count(B_ANCHOR) == 1, 'anchor B count=%d' % src.count(B_ANCHOR)
B_NEW = '''    # ---- 外部基线臂：把已发布压缩器的输出原样注入本路径 ----
    # 动机：此前所有数字都是内部臂互比。审稿人会问"冗余度压缩在多实体案例上系统性
    # 失效"是领域问题还是我们自己实现的产物。这里把 batch1 的已发布方法用**原码**
    # 跑出来的文本，送进与内部臂逐字相同的生成 + 打分 + 泄漏 + 实体诊断路径。
    # 只用它们的输出文本，不用任何中间量 —— 唯一变量是"这段上下文怎么来的"。
    ext_methods = []
    ext_arm_names = set()
    if a.ext_arms:
        ext = {}
        for line in io.open(a.ext_arms, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get('text'):
                ext[(o['key'], o['method'])] = o['text']
        ext_methods = sorted({m for (_, m) in ext})
        ext_arm_names = {'ext_' + m for m in ext_methods}
        n_ext = 0
        for ci, c in enumerate(cases):
            F = nt(c['context'])
            for m in ext_methods:
                txt = ext.get((c['key'], m))
                if txt is None:
                    continue
                arm = 'ext_' + m
                prompts.append(make_prompt(c['question'], txt))
                keys.append((ci, arm, a.ext_ratio))
                toks_used.append(nt(txt))
                arm_txt.append(txt)
                B = min(budget_target(F, a.ext_ratio, a.min_budget_tokens), a.maxlen - 64)
                s = acc[(arm, a.ext_ratio)]
                s['n'] += 1
                s['tgt'] += B
                s['got'] += toks_used[-1]
                dev = toks_used[-1] - B
                s['dev'] += dev
                s['devmax'] = max(s['devmax'], abs(dev))
                s['devmax_signed'] = max(s['devmax_signed'], dev)
                # 已发布 compress_case 内部已 assert matched.tokens <= floor(0.5n)，
                # 这里再独立记一次超标量，避免把"它自己违规"和"我们记账错"混为一谈。
                if toks_used[-1] > B:
                    log('EXT_OVER_BUDGET %s %s got=%d B=%d' % (c['key'], arm, toks_used[-1], B))
                if a.dump_arms:
                    arm_dump.append(dict(key=c['key'], arm=arm, ratio=a.ext_ratio,
                                         ctx_tokens=toks_used[-1], target=B, text=txt))
                n_ext += 1
        log('外部基线臂注入 %d 条（%d 方法 × %d 案例），ratio 记账=%.2f'
            % (n_ext, len(ext_methods), len(cases), a.ext_ratio))
        if a.dump_arms:
            io.open(a.dump_arms, 'w', encoding='utf-8', newline='\\n').write(
                '\\n'.join(json.dumps(d, ensure_ascii=False) for d in arm_dump) + '\\n')

    # ---- 预测 ----
'''
src = src.replace(B_ANCHOR, B_NEW, 1)

# ---------------------------------------------------------------- C. 两处作用域过滤
C1 = "            if (scope == 'budget') != (k[1] in ARMS_BUDGET):\n"
assert src.count(C1) == 1, 'anchor C1 count=%d' % src.count(C1)
src = src.replace(C1, "            if (scope == 'budget') != (k[1] in ARMS_BUDGET or k[1] in ext_arm_names):\n", 1)
C2 = "            if (scope == 'budget') != (k[0] in ARMS_BUDGET):\n"
assert src.count(C2) == 1, 'anchor C2 count=%d' % src.count(C2)
src = src.replace(C2, "            if (scope == 'budget') != (k[0] in ARMS_BUDGET or k[0] in ext_arm_names):\n", 1)

# ---------------------------------------------------------------- D. READ 说明
D_ANCHOR = "    w('      它们的 token 数逐格打印在上表，绝不含混。')\n"
assert src.count(D_ANCHOR) == 1, 'anchor D count=%d' % src.count(D_ANCHOR)
D_NEW = D_ANCHOR + (
    "    if ext_methods:\n"
    "        w('NOTE: ext_* 是**已发布方法**（batch1 原码）输出的上下文，按 ratio=%.2f 记账；'\n"
    "          % a.ext_ratio)\n"
    "        w('      其原生预算是 floor(0.5*n)，本项目是 round(0.5*n)，最多差 1 token。')\n"
    "        w('      它们是等预算对照，不含删除/别名语义（arm_style=none），故 collision 恒 0；')\n"
    "        w('      ctx_leak 反映的是这些方法原生能不能把机密压掉，不是我们加的语义。')\n"
)
src = src.replace(D_ANCHOR, D_NEW, 1)

# ---------------------------------------------------------------- 收尾
assert src != orig, 'nothing changed'
# 幂等保护：不许出现两次
for marker in ('--ext-arms', 'ext_arm_names'):
    n = len(re.findall(re.escape(marker), src))
    if n > 6 and marker == 'ext_arm_names':
        raise SystemExit('ext_arm_names 出现 %d 次，疑似重复打补丁' % n)

io.open(TARGET, 'w', encoding='utf-8', newline='\n').write(src)
print('PATCHED %s: %d -> %d bytes (%+d)' % (TARGET, len(orig.encode('utf-8')),
                                            len(src.encode('utf-8')),
                                            len(src.encode('utf-8')) - len(orig.encode('utf-8'))))
