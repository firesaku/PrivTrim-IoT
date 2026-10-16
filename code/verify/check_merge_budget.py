"""Arm-budget audit for the V2 similarity-greedy-merge feasibility experiment.

WHY THIS EXISTS
---------------
`v2c_feasibility.py` compares three arms at a nominal merge ratio:

    similarity      -- greedy: take the highest-cosine pairs first
    random_adjacent -- shuffle-and-pair control
    random_far      -- same, skipping index-neighbours (min_gap>=2)

The reported metrics (`hit_answer`, `mix_sens`, `fuse_2sec`) are RATES over the
resulting clusters. A rate comparison is only meaningful if the three arms are
matched on how much they actually merged. The nominal `ratio` is a *request*,
not a guarantee, because `make_clusters()` drops any pair whose endpoints were
already consumed by an earlier pair -- and it never back-fills.

`pairs_similarity()` proposes the top-`target/2` pairs from the global cosine
ranking *without tracking unit usage*. High-similarity pairs cluster around
"hub" units, so many of them collide and get discarded. `pairs_random()`
pairs `idx[k]` with `idx[k+1]`, which are distinct by construction and never
collide. The similarity arm therefore under-merges, and the gap grows with the
requested ratio.

This script measures that gap directly. It imports the functions from
`v2c_feasibility.py` instead of reimplementing them on purpose: the project's
`v1b_diag.py` duplicated `v1_checker.py`'s predicates line-for-line and the two
copies silently drifted apart, so a re-derived copy here would prove nothing
about the code that actually produced the published numbers.

USAGE
-----
    python check_merge_budget.py \
        --samples ../../crossdata/frozen_samples.jsonl \
        --v2c ../../hpc/v2c_feasibility.py \
        --ratios 0.125,0.25,0.5

Exit code is 0 when every arm lands within 2% of its requested merge count,
and 1 when any arm is materially under budget (so this can gate a rerun).
"""
import argparse
import importlib.util
import io
import json
import random
import sys
from collections import defaultdict


def load_v2c(path):
    """Import v2c_feasibility.py as a module so we exercise the real functions."""
    spec = importlib.util.spec_from_file_location('v2c_feasibility', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--v2c', required=True, help='path to v2c_feasibility.py')
    ap.add_argument('--ratios', default='0.125,0.25,0.5')
    ap.add_argument('--cap', type=int, default=150)
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tolerance', type=float, default=0.02,
                    help='allowed relative shortfall before an arm is flagged')
    a = ap.parse_args()

    m = load_v2c(a.v2c)
    ratios = [float(x) for x in a.ratios.split(',')]
    rng = random.Random(a.seed)

    recs = [json.loads(l) for l in io.open(a.samples, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]

    # requested merges (pairs) vs achieved disjoint clusters, per (ratio, arm)
    want = defaultdict(int)
    got = defaultdict(int)
    scored = 0

    for r in recs:
        units = m.split_units(r['context'], a.cap)
        if len(units) < 8:
            continue                      # same skip rule as v2c
        scored += 1
        S = m.build_matrix(units)
        import numpy as np
        S = S @ S.T
        np.fill_diagonal(S, -1.0)
        n = len(units)

        for ratio in ratios:
            target = max(int(ratio * n), 2)
            plans = {
                'similarity': m.pairs_similarity(S, target),
                'random_adjacent': m.pairs_random(n, target, rng, min_gap=0),
                'random_far': m.pairs_random(n, target, rng, min_gap=2),
            }
            for arm, pairs in plans.items():
                # what the arm asked for, in merges (each merge consumes 2 units)
                want[(ratio, arm)] += len(pairs)
                # what it actually got after the used-unit filter
                got[(ratio, arm)] += len(m.make_clusters(n, pairs))

    L = []
    w = L.append
    w("ARM-BUDGET AUDIT of v2c_feasibility.py")
    w("samples=%s  records_scored=%d  cap=%d  seed=%d  ratios=%s"
      % (a.samples.split('\\')[-1].split('/')[-1], scored, a.cap, a.seed, ratios))
    w("")
    w("%-8s %-16s %10s %10s %9s" % ("ratio", "arm", "requested", "achieved", "ach/req"))
    worst = 1.0
    flagged = []
    for ratio in ratios:
        sim_ach = got[(ratio, 'similarity')]
        for arm in ('similarity', 'random_adjacent', 'random_far'):
            req, ach = want[(ratio, arm)], got[(ratio, arm)]
            frac = ach / req if req else 0.0
            if arm == 'similarity':
                worst = min(worst, frac)
                if frac < 1.0 - a.tolerance:
                    flagged.append((ratio, frac))
            w("%-8s %-16s %10d %10d %8.1f%%" % (ratio, arm, req, ach, 100 * frac))
        ra = got[(ratio, 'random_adjacent')]
        rf = got[(ratio, 'random_far')]
        if sim_ach:
            w("%-8s %-16s %10s %10s %8.2fx / %.2fx"
              % (ratio, 'CONTROL/SIM', '', '', ra / sim_ach, rf / sim_ach))
        w("")
    if flagged:
        w("VERDICT: FAIL - the similarity arm is materially under its requested budget:")
        for ratio, frac in flagged:
            w("  ratio=%.3f  achieved only %.1f%% of requested merges" % (ratio, 100 * frac))
        w("  The three arms are NOT matched on achieved compression, so the per-cluster")
        w("  rates in the published table compare unequal denominators.")
        w("  Fix: keep scanning the ranking until target/2 DISJOINT pairs are found")
        w("  (standard greedy matching) instead of stopping after target/2 proposals.")
    else:
        w("VERDICT: PASS - every arm within %.0f%% of its requested merge count."
          % (100 * (1 - a.tolerance)))

    out = "\n".join(L)
    print(out)
    return 1 if flagged else 0


if __name__ == '__main__':
    sys.exit(main())
