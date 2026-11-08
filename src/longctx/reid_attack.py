#!/usr/bin/env python3
"""Re-identification attack: can a model recover a protected value from each arm's prompt?

The paper's leakage number is a pattern match against the context text. That is the right measure
for "is the value recoverable", but it is a static check, and a reviewer is entitled to ask whether
the value is recoverable by a model that never saw the original. This script runs that attack.

Threat model, stated to match the paper's: the adversary is the model provider. It receives the
prompt, may retain it, and may run any model over it. It does NOT hold the alias map and does not
observe the raw log. So the attacker sees exactly the passage an arm would send, with no question,
and is asked to enumerate identifiers.

Scoring uses the same predicates as the paper (`privcheck.exposure_basic`), so an attack success and
a "leak" mean the same thing: the full value, or a distinguishing fragment of it, appears in the
attacker's output.

One design note. The attacker is deliberately the SAME model the paper uses as reader. That keeps
the attack reproducible on the hardware we have and holds the attacker fixed across arms, which is
what makes the comparison between arms meaningful. If a stronger attacker is warranted later, rerun
with --lm pointing at it; the arm construction does not change.

Usage
    python src/longctx/reid_attack.py --cases data/longctx_max.jsonl \
        --lm <snapshot> --out out/reid/report.txt --percase out/reid/percase.jsonl
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e2e_quad as E                                      # noqa: E402  单一实现来源
import privcheck                                          # noqa: E402

ATTACK_SYSTEM = (
    'You are auditing a machine log for privacy exposure. You do not have the original document. '
    'Extract any account identifiers, e-mail addresses or telephone numbers that appear in the '
    'passage, verbatim. One per line, no explanation. If the passage contains none, output NONE.')


def attack_prompt(context):
    return [{'role': 'system', 'content': ATTACK_SYSTEM},
            {'role': 'user', 'content': 'Passage:\n' + context + '\n\nIdentifiers:'}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--percase', required=True)
    ap.add_argument('--lm', required=True)
    ap.add_argument('--arms', default='full,quad4,budget_cluster,alias_budget')
    ap.add_argument('--budgets', default='0.5')
    ap.add_argument('--dtype', default='fp16', choices=['bf16', 'fp16'])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=96)
    ap.add_argument('--maxlen', type=int, default=24576)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--alias-question', action='store_true')
    ap.add_argument('--dump-only', action='store_true',
                    help='不加载模型，只把攻击提示词的长度与臂统计写出来（本地检查用）')
    a = ap.parse_args(argv)

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        raise SystemExit('no cases')

    arms = [x for x in a.arms.split(',') if x]
    bad = [x for x in arms if x not in E.ARMS_ALL]
    if bad:
        raise SystemExit('unknown arms %s; available: %s' % (bad, E.ARMS_ALL))
    budgets = [float(x) for x in a.budgets.split(',') if x]

    if a.dump_only:
        # local sanity pass: no model, just prove the arms build and the prompts fit
        nt = lambda s: max(1, len(s) // 4)
        n = 0
        for c in cases[:20]:
            texts = E.build_arm_texts(c, nt, budgets, arms, E.lex_drop_order)
            for (arm, ratio) in sorted(texts, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
                n += 1
        print('dump-only: %d cases x %d arms -> %d attack prompts, longest budget %s'
              % (len(cases[:20]), len(arms), n, budgets))
        return 0

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print('gpu', torch.cuda.get_device_name(0), flush=True)
    tok = AutoTokenizer.from_pretrained(a.lm, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    tok.truncation_side = 'right'
    TD = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    lm = AutoModelForCausalLM.from_pretrained(a.lm, local_files_only=True,
                                              torch_dtype=TD, device_map='cuda')
    lm.eval()
    nt = lambda s: len(tok.encode(s, add_special_tokens=False))

    # ---- build every arm text and its attack prompt ----
    prompts, keys, n_cans = [], [], []
    t0 = time.time()
    for ci, c in enumerate(cases):
        texts = E.build_arm_texts(c, nt, budgets, arms, E.lex_drop_order)
        for (arm, ratio) in sorted(texts, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
            txt = texts[(arm, ratio)]
            if nt(txt) > a.maxlen - a.max_new - 64:
                continue                      # would be truncated; a truncated prompt is not an arm
            prompts.append(attack_prompt(txt))
            keys.append((ci, arm, ratio))
            n_cans.append(len(E.eval_canaries(c)))
    print('built %d attack prompts in %.1fs' % (len(prompts), time.time() - t0), flush=True)

    outs = []
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i:i + a.batch]
        enc = tok.apply_chat_template(chunk, return_tensors='pt', padding=True,
                                      add_generation_prompt=True,
                                      return_dict=True).to('cuda')
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        new = gen[:, enc['input_ids'].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        if (i // a.batch) % 20 == 0:
            print('  %d/%d' % (min(i + a.batch, len(prompts)), len(prompts)), flush=True)

    # ---- score with the paper's own predicates ----
    agg = collections.defaultdict(lambda: dict(n=0, succ=0, val=0, can=0))
    percase = []
    for (ci, arm, ratio), out, cn in zip(keys, outs, n_cans):
        c = cases[ci]
        cans = E.eval_canaries(c)
        full_hits = frag_hits = 0
        for x in cans:
            full, frag = privcheck.exposure_basic(x, out)
            full_hits += 1 if full else 0
            frag_hits += 1 if frag else 0
        s = agg[(arm, ratio)]
        s['n'] += 1
        s['succ'] += 1 if full_hits else 0
        s['val'] += full_hits
        s['can'] += len(cans)
        percase.append(dict(case_key=c['key'], source=c.get('source'),
                            case_kind=c.get('case_kind'), arm=arm, ratio=ratio,
                            n_canaries=len(cans), reid_full=full_hits, reid_frag=frag_hits,
                            attacker_out=out.strip()[:300]))

    os.makedirs(os.path.dirname(a.percase) or '.', exist_ok=True)
    with io.open(a.percase, 'w', encoding='utf-8', newline='\n') as fh:
        for r in percase:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')

    lines = []
    w = lines.append
    w('re-identification attack')
    w('=' * 60)
    w('attacker = the paper reader, fixed across arms; it sees the passage only, no question.')
    w('%-20s %6s %12s %12s' % ('arm', 'n', 'cases hit', 'values hit'))
    for (arm, ratio), s in sorted(agg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        tag = arm if ratio is None else '%s@%.2f' % (arm, ratio)
        w('  %-20s %6d %11.1f%% %11.2f%%'
          % (tag, s['n'], 100.0 * s['succ'] / max(1, s['n']),
             100.0 * s['val'] / max(1, s['can'])))
    text = '\n'.join(lines) + '\n'
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
