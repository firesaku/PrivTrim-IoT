"""Attack A2: rank inversion (zero GPU).

v4_router.Aliaser with no PT_ALIAS_KEY assigns alias letters in ascending
(type, value) order.  An attacker who knows the *set* of true values (e.g.
from a leaked database) can therefore invert every alias by sorting.

Attacker strategy is fixed: sort the known values ascending, then pair them
with the aliases in letter order (A = smallest, B = next, ...).  We compare
that single strategy against two defender settings:

  defender without key -> hit rate should be 100%
  defender with key    -> hit rate should collapse to ~1/k!
"""
import json, io, sys, math, collections

RD = '/path/to/project'
sys.path.insert(0, RD + '/code/longctx')
import v4_router as V

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
N = 300
sample = cases[:N]


def alias_pairs(case, defender_key=None):
    al = V.Aliaser('alias', order_key=defender_key)
    cant = case.get('canaries') or []
    al.preseed(cant)
    by = collections.defaultdict(list)
    for c in cant:
        by[c['type']].append((c['value'], al.token(c)))
    return by


def _letter(tok):
    return tok.rstrip(']').split('-')[-1]


def attacker_sorted_guess(by):
    """Attacker does NOT know the mapping: pairs aliases in letter order
    with the ascending-sorted known values."""
    hit = tot = 0
    for t, pairs in by.items():
        vals = sorted(set(v for v, _ in pairs))
        true_map = {}
        for v, tok in pairs:
            true_map[tok] = v
        toks = sorted(true_map, key=_letter)
        for i, tok in enumerate(toks):
            if i >= len(vals):
                break
            tot += 1
            if true_map[tok] == vals[i]:
                hit += 1
    return hit, tot


def perm_space(by):
    p = 1.0
    for t, pairs in by.items():
        p *= math.factorial(len(set(k for _, k in pairs)))
    return p


h0 = t0 = h1 = t1 = 0
exp_random = 0.0
kdist = collections.Counter()

for c in sample:
    by0 = alias_pairs(c, None)
    a, b = attacker_sorted_guess(by0)
    h0 += a; t0 += b
    for t, pairs in by0.items():
        kdist[len(set(k for _, k in pairs))] += 1

    by1 = alias_pairs(c, 'demo-key-2026')
    a, b = attacker_sorted_guess(by1)
    h1 += a; t1 += b
    exp_random += 1.0 / perm_space(by1)

print('=== sample: %d cases; entities per (case,type) distribution ===' % len(sample))
for k in sorted(kdist):
    print('   k=%-2d  %d times   (permutation space %d! = %d)' % (k, kdist[k], k, math.factorial(k)))

print('\n=== same attack (guess by ascending order) vs two defender settings ===')
print('   defender WITHOUT key : %d/%d = %.1f%%' % (h0, t0, 100.0 * h0 / t0))
print('   defender WITH key    : %d/%d = %.1f%%' % (h1, t1, 100.0 * h1 / t1))
print('   theoretical random expectation under key = %.4f%%' % (100.0 * exp_random / len(sample)))
