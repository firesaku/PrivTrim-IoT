"""Patch e2e_longctx_r2.py -> r3: fix the four truncation/config defects.

1. --maxlen default 8192 -> 24576 (real max context is 18,590 tokens; the old
   default came from a `words x 1.33` estimate that was 2.8x too small).
2. Explicit --maxlen-policy {raise,truncate}, default RAISE. Silent truncation
   is what hid all of this; the harness must refuse rather than quietly cut.
   Also set tok.truncation_side explicitly so it is never implicit.
3. Cap the per-case budget so it can never exceed what the model can see
   (0.75 x an 18,590-token context = 13,942, which is fine at 24576 but was
   over the old limit for 25 cases).
4. Record prompt_tokens per row so any future truncation is visible, and make
   ans_in_ctx demonstrably refer to the text the model actually receives.
"""
import io
import re

SRC = r'D:\CodexData\tmp\privtrim-review-20261015\hpc\e2e_longctx_r3.py'
src = io.open(SRC, encoding='utf-8').read()
orig = src
report = []

# ---- 1 + 2: argparse ------------------------------------------------------
m = re.search(r"ap\.add_argument\('--maxlen',\s*type=int,\s*default=\d+,", src)
assert m, 'maxlen arg not found'
src = src[:m.start()] + "ap.add_argument('--maxlen', type=int, default=24576," + src[m.end():]
report.append('maxlen default -> 24576')

m = re.search(r"help='tokenizer 截断长度；[^']*'\)", src)
assert m, 'maxlen help not found'
src = src[:m.start()] + ("help='提示词长度上限；默认 24576（实测最长 18,590 token 上下文 + 60 开销）。'\n"
                         "                         '旧默认 8192 来自 words*1.33 估算，低估 2.8 倍，导致 42%% 案例被静默截断')"
                         ) + src[m.end():]
report.append('maxlen help updated')

m = re.search(r"ap\.add_argument\('--min-budget-tokens',\s*type=int,\s*default=\d+\)", src)
assert m, 'min-budget arg not found'
src = src[:m.end()] + ("\n    ap.add_argument('--maxlen-policy', default='raise', choices=['raise', 'truncate'],\n"
                       "                    help='raise（默认）= 提示词超过 --maxlen 直接报错。静默截断会让模型看不到'\n"
                       "                         '部分上下文，而 ans_in_ctx 之类的指标对此不可见 —— 这正是上一轮'\n"
                       "                         '42%% 案例被截断却没被发现的原因。truncate 仅供诊断，会记录 prompt_tokens')"
                       ) + src[m.end():]
report.append('--maxlen-policy added (default raise)')

# ---- 2b: explicit truncation_side ----------------------------------------
m = re.search(r"^(\s*)tok\.padding_side = 'left'", src, re.M)
assert m, 'padding_side line not found'
src = src[:m.start()] + (m.group(0) + "\n" + m.group(1) +
                         "# 显式声明，不要依赖库默认（默认 'right' = 保留开头、砍掉结尾，且完全静默）\n" +
                         m.group(1) + "tok.truncation_side = 'right'") + src[m.end():]
report.append('truncation_side set explicitly')

# ---- 3 + 4: the generation loop ------------------------------------------
old_enc = """            enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
                      max_length=a.maxlen).to('cuda')"""
assert old_enc in src, 'encode block not found'
new_enc = """            # 提示词长度必须显式检查：静默截断会砍掉上下文尾部，而答案区域可能就在那里
            plens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            if a.maxlen_policy == 'raise':
                bad = [(keys[i + j], plens[j]) for j in range(len(plens)) if plens[j] > a.maxlen]
                if bad:
                    raise RuntimeError(
                        '提示词超过 --maxlen=%d，拒绝静默截断（会砍掉上下文尾部）: %s'
                        % (a.maxlen, bad[:3]))
            prompt_tokens.extend(plens)
            enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
                      max_length=a.maxlen).to('cuda')"""
src = src.replace(old_enc, new_enc, 1)
report.append('pre-encode length check (raise) + prompt_tokens recorded')

m = re.search(r"^(\s*)prompts, keys, toks_used(?:, arm_txt)? = \[\], \[\], \[\](?:, \[\])?", src, re.M)
assert m, 'prompts init not found'
src = src[:m.end()] + "\n" + m.group(1) + "prompt_tokens = []" + src[m.end():]
report.append('prompt_tokens accumulator added')

# ---- 3b: cap the per-case budget by what the model can actually see -------
m = re.search(r"B = F if ratio is None else budget_target\(F, ratio, a\.min_budget_tokens\)", src)
assert m, 'B assignment not found'
src = src[:m.start()] + ("B = F if ratio is None else budget_target(F, ratio, a.min_budget_tokens)\n"
                         "            # 预算不能超过模型能看到的长度（否则该臂也会被截断）\n"
                         "            B = min(B, a.maxlen - 64)") + src[m.end():]
report.append('budget capped by maxlen')

# ---- 4b: ans_in_ctx anchored to the model input, prompt_tokens in percase --
old_ans = "        ans_in = 1 if any(g in ctx_text for g in golds) else 0"
assert old_ans in src, 'ans_in line not found'
src = src.replace(old_ans,
    "        # ctx_text 就是进入提示词的文本；因为没有截断，它等同模型实际收到的上下文。\n"
    "        # prompt_tokens 逐例记录，任何未来的意外截断都会在产物里可见。\n"
    "        ans_in = 1 if any(g in ctx_text for g in golds) else 0", 1)
report.append('ans_in_ctx annotated')

m = re.search(r"ans_in_ctx=ans_in, ctx_leak_value=cv, ctx_leak_frag=cf,", src)
assert m, 'percase ans_in_ctx not found'
src = src[:m.start()] + "ans_in_ctx=ans_in, prompt_tokens=prompt_tokens[i], ctx_leak_value=cv, ctx_leak_frag=cf," + src[m.end():]
report.append('prompt_tokens in percase')

assert src != orig
io.open(SRC, 'w', encoding='utf-8', newline='\n').write(src)

import py_compile
py_compile.compile(SRC, doraise=True)
print('OK patched e2e_longctx_r3.py')
for r in report:
    print('  -', r)
for pat in ('default=24576', '--maxlen-policy', "truncation_side = 'right'", 'prompt_tokens.extend',
            'prompt_tokens=prompt_tokens[i]', 'B = min(B, a.maxlen - 64)'):
    print('  check %-40s %s' % (pat, pat in src))
