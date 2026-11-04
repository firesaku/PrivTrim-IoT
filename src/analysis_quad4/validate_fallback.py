"""Model-free check of the quad4 routing logic (anchor sentence-window fallback
+ forced-set trimming).

What it asserts, on four datasets:
  1) mode selection: excerpts with record markers use "entry" mode, the rest
     use "window" mode;
  2) forced-retention / budget ratio (must stay <= 1, otherwise the budget
     guard in the harness raises);
  3) whether the reference answer is still present in the emitted prompt -- a
     model-independent upper bound on F1;
  4) that leakage is zero.

No GPU and no model: it exercises the construction path only.
"""
import gzip
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'longctx'))
import v4_router  # noqa: E402
import privcheck  # noqa: E402

QUAD_WINDOW = 2


def split_units(t):
    parts = re.split(r'(?<=[.!?])\s+|\n+', t)
    return [p.strip() for p in parts if len(p.strip()) >= 2]


def norm(s):
    return re.sub(r'\s+', ' ', s.lower()).strip()


def est_tok(s):
    return int(len(re.findall(r'\S+', s)) * 1.33)


def entry_index(units):
    idx, cur = [], -1
    for u in units:
        if re.match(r'^-{3,}\s*entry\s+\d+', u.strip(), re.I):
            cur += 1
        idx.append(cur)
    return idx


def is_sens(u, cans):
    nu = u.lower()
    return any(c['value'].lower() in nu or
               (c.get('marker') and c['marker'].lower() in nu) for c in cans)


def pkeys(case):
    ks = []
    if case.get('target_entity'):
        ks.append(case['target_entity'])
    if case.get('anchor'):
        ks.append('----- entry %s -----' % case['anchor']
                  if case.get('anchor_kind') == 'entry' else case['anchor'])
    return ks


def lex_order(units, flags):
    toks = [set(re.findall(r'[a-z0-9]+', u.lower())) for u in units]
    n = len(units)
    red = []
    for i in range(n):
        best = 0.0
        for j in range(n):
            if i == j:
                continue
            a, b = toks[i], toks[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if inter:
                best = max(best, inter / float(len(a) + len(b) - inter))
        red.append(best)
    rank = sorted(range(n), key=lambda i: (-red[i], i))
    return [i for i in rank if not flags[i]] + [i for i in rank if flags[i]]


def build(case, ratio=0.5):
    full = case['context']
    cans = case.get('canaries') or []
    us = split_units(full)
    ents = entry_index(us)
    ak = pkeys(case)
    tgt_units = [i for i, u in enumerate(us) if any(k and k in u for k in ak)]
    has_marker = any(re.match(r'^-{3,}\s*entry\s+\d+', u.strip(), re.I) for u in us)
    if has_marker:
        tgt_ents = set(ents[i] for i in tgt_units)
        rel_set = set(i for i in range(len(us)) if ents[i] in tgt_ents)
        mode = 'entry'
    else:
        rel_set = set()
        for i in tgt_units:
            for j in range(max(0, i - QUAD_WINDOW), min(len(us), i + QUAD_WINDOW + 1)):
                rel_set.add(j)
        mode = 'window'
    if not rel_set:
        rel_set = set(tgt_units)

    B = int(round(ratio * sum(est_tok(u) for u in us)))
    al = v4_router.Aliaser('alias')
    al.preseed(cans)
    for c in cans:
        al.token(c)

    out, forced, cand = [], [], []
    for i, u in enumerate(us):
        s = is_sens(u, cans)
        r = i in rel_set
        if s and r:
            out.append(v4_router.apply_replacements(u, cans, al)); forced.append(i)
        elif s and not r:
            out.append('')
        elif r:
            out.append(u); forced.append(i)
        else:
            out.append(u); cand.append(i)

    prio = sorted(forced, key=lambda i: (min(abs(i - j) for j in tgt_units), i)) if tgt_units else sorted(forced)
    kept, acc = [], 0
    for i in prio:
        c = est_tok(out[i])
        if acc + c <= B:
            kept.append(i); acc += c
        elif out[i]:
            cand.append(i)
    forced = kept

    flags = [False] * len(out)
    for i in forced:
        flags[i] = True
    order = lex_order([x if x else '.' for x in out], flags)
    total = sum(est_tok(out[i]) for i in forced)
    chosen = list(forced)
    for i in order:
        if i in cand and total + est_tok(out[i]) <= B:
            chosen.append(i); total += est_tok(out[i])
    chosen.sort()
    fs = set(forced)
    while True:
        txt = ' '.join(out[i] for i in chosen if out[i])
        if est_tok(txt) <= B:
            break
        d = False
        for i in reversed(chosen):
            if i not in fs:
                chosen.remove(i); d = True; break
        if not d:
            break
    return ' '.join(out[i] for i in chosen if out[i]), B, mode, sum(est_tok(out[i]) for i in forced)


def run(name, cases, n=150):
    print('=== %s ===' % name)
    rows = []
    for c in cases[:n]:
        txt, B, mode, forced = build(c)
        golds = [str(x.get('text') if isinstance(x, dict) else x)
                 for x in (c.get('answers') or [])]
        golds = [g for g in golds if g]
        ans_ok = any(norm(g) and norm(g) in norm(txt) for g in golds)
        lk = 0
        for can in (c.get('canaries') or []):
            full_hit, _ = privcheck.exposure_basic(can, txt)
            lk += 1 if full_hit else 0
        rows.append((mode, forced / B if B else 0, ans_ok, lk, est_tok(txt) <= B))
    modes = {}
    for m, *_ in rows:
        modes[m] = modes.get(m, 0) + 1
    over = sum(1 for r in rows if r[1] > 1.0)
    idx_over = sum(1 for r in rows if not r[4])
    print('  mode: %s' % modes)
    print('  forced/budget: mean=%.3f max=%.3f  cases over budget=%d/%d'
          % (statistics.mean([r[1] for r in rows]), max(r[1] for r in rows), over, len(rows)))
    print('  emitted prompt over budget: %d / %d' % (idx_over, len(rows)))
    print('  gold answer present in prompt: %d / %d (%.1f%%)'
          % (sum(1 for r in rows if r[2]), len(rows), 100.0 * sum(1 for r in rows if r[2]) / len(rows)))
    print('  cases with a leaked canary: %d / %d' % (sum(1 for r in rows if r[3]), len(rows)))
    print()


mx = []
with gzip.open(REPO / 'data' / 'longctx_max.jsonl.gz', 'rt', encoding='utf-8') as fh:
    for line in fh:
        if line.strip():
            mx.append(json.loads(line))
run('longctx_max (2,728 cases, record-block structure)', mx, 150)

for label, fn in (('public_lb Type A (topical, no entity named)', 'public_lb.jsonl.gz'),
                  ('public_lb Type B (entity named, single hop)', 'public_lb_typeb.jsonl.gz'),
                  ('public_lb Type C (entity named, two hop)', 'public_lb_typec.jsonl.gz')):
    cs = []
    with gzip.open(REPO / 'data' / fn, 'rt', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                cs.append(json.loads(line))
    run(label, cs, 150)
