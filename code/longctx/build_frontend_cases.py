#!/usr/bin/env python3
"""Turn DARK's own prompt into a case file, so a published compressor can be applied on top of it.

Experiment 1 asks a question a reviewer will ask first: if you alias the identifier anyway, can the
strongest compressor match DARK? Answering it needs the compressor to run on an ALIASED passage, not
on the raw one, which means the case file itself has to change: `context` becomes what DARK would
have sent, `question` becomes the question DARK would have sent (the same alias map, or the two
halves point at different entities), and the answers stay as they are.

Two properties of the output matter and are checked rather than assumed:

  * The composed arm cannot leak. DARK's passage already carries aliases, so no protected value is
    in it. Any leakage reported for a composed arm would mean the aliasing step failed, not that
    compression reintroduced the value.

  * The mapping is the same one DARK uses. `question_for_arm(case, 'quad4', True)` is the function
    the harness itself calls, so the question text here is byte-identical to what DARK would send,
    rather than a second implementation that could drift.

The DARK context comes from `e2e_quad.py --dump-arms` (arm `quad4`), so this script does no prompt
construction of its own.

Usage
    python src/longctx/build_frontend_cases.py --cases data/longctx_max.jsonl \
        --dump out/quad4_dump.jsonl --out data/frontend_cases.jsonl
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e2e_quad as E                                       # noqa: E402  单一实现来源


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--dump', default='', help='JSONL from e2e_quad.py --dump-arms')
    ap.add_argument('--lm', default='',
                    help='tokenizer snapshot; when --dump is absent the passage is rebuilt here')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', default='quad4')
    ap.add_argument('--budget', type=float, default=0.5)
    a = ap.parse_args(argv)

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    by_key = {c['key']: c for c in cases}

    dark = {}
    if a.dump:
        for line in io.open(a.dump, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('arm') != a.arm or r.get('ratio') is not None:
                continue
            dark[r['key']] = r['text']
        if not dark:
            raise SystemExit('no %s rows with ratio=None in %s' % (a.arm, a.dump))
    else:
        # Rebuild the passage with the REAL tokenizer. A dry-run dump would use the char/4
        # approximation, and arm construction depends on nt, so the passage would differ from what
        # DARK actually sends. Only the tokenizer is loaded here -- no weights.
        if not a.lm:
            raise SystemExit('need --dump or --lm')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.lm, local_files_only=True)
        nt = lambda s: len(tok.encode(s, add_special_tokens=False))
        for c in cases:
            texts = E.build_arm_texts(c, nt, [a.budget], [a.arm], E.lex_drop_order)
            key = (a.arm, a.budget)
            if key in texts:
                dark[c['key']] = texts[key]

    out, n_absent, n_qchange = [], 0, 0
    for key, ctx in sorted(dark.items()):
        c = by_key.get(key)
        if c is None:
            n_absent += 1
            continue
        q0 = c.get('question') or ''
        q1 = E.question_for_arm(c, a.arm, True)
        if q1 != q0:
            n_qchange += 1
        new = dict(c)
        new['context'] = ctx
        new['question'] = q1
        # The composed passage must not be re-redacted: the values are already gone, and re-running
        # redaction on an aliased passage would corrupt the aliases.
        new['canaries_router'] = []
        out.append(new)

    with io.open(a.out, 'w', encoding='utf-8', newline='\n') as fh:
        for c in out:
            fh.write(json.dumps(c, ensure_ascii=False) + '\n')

    # A composed arm cannot leak: if a protected value survives into the aliased passage, the
    # aliasing step itself is broken and every downstream number would be meaningless.
    import privcheck
    leaks = 0
    for c in out[:400]:
        for can in (c.get('canaries') or []):
            full, frag = privcheck.exposure_basic(can, c['context'])
            if full or frag:
                leaks += 1
                break
    checked = min(400, len(out))

    print('frontend cases: %d  (arm=%s, PT_ALIAS_KEY=%s)'
          % (len(out), a.arm, os.environ.get('PT_ALIAS_KEY', '<unset>')))
    print('  keys without a DARK passage: %d' % n_absent)
    print('  questions rewritten by the alias map: %d / %d' % (n_qchange, len(out)))
    print('  composed passages with a protected value still present: %d / %d' % (leaks, checked))
    if leaks:
        raise SystemExit('FATAL: aliased passage still carries a protected value')
    return 0


if __name__ == '__main__':
    sys.exit(main())
