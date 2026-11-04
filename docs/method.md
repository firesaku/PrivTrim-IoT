# Method: DARK

Given an excerpt *C*, a question *q*, a budget ratio *r*, and the protected value set *V*, DARK
produces a rewritten question *q′* and a compressed context *P*, such that *(q′, P)* fits the token
budget and contains no value of *V*.

```
      C, q, r, V
          |
   (1)  split C into units U                             lines of the excerpt
   (2)  label each unit: sensitive? relevant?
   (3)  route each unit to its operation                 D / A / R / K
   (4)  fit the kept set into B = floor(r * |C|)         hard budget
   (5)  rewrite q through the same alias map             -> q'
          |
   q', P  with no value of V in either half
```

Everything is deterministic and model-free. No training, no gradient, no forward pass. The only
deployment-specific input is *V*, which an operator already holds for redaction.

## Step 1 — units

A unit is one line of the excerpt. Lines are already the natural record granularity in these logs, and
they stay stable under the alias operation.

## Step 2 — two labels

**Sensitivity** is a property of the unit given *V*: a unit is sensitive when it contains a protected
value. Three value types are recognised by `src/longctx/privcheck.py`:

| Type | Pattern | Example |
| --- | --- | --- |
| `email` | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b` | `contact.850f270b15a6@example.invalid` |
| `phone` | `(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)` | `+1-202-555-0027` |
| `account` | `\bACCT-[A-Za-z0-9-]+\b` | `ACCT-7a96d27de743` |

**Relevance** uses a single signal: the region containing a mention of the entity the question names.
The signal takes two forms depending on the layout of the excerpt:

- **Record blocks** (the common case). When the excerpt is segmented into blocks headed by a record
  marker, the whole block containing the anchor mention is relevant. This holds for 100% of
  multi-entity cases and 99.7% of needle cases.
- **Sentence windows** (fallback). LongBench-style passages carry no block markers; there the relevant
  region is the anchor sentence plus one sentence on each side — five units, about 2.3% of the budget
  in our measurements.

**No answer key.** A tempting implementation of relevance is an oracle that marks the unit containing
the reference answer. That implementation is not deployable, and it changes what the method is. DARK
never reads it; [Results](results.md#answer-key) reports how much an oracle would be worth, so the
comparison is explicit rather than implicit.

## Step 3 — four operations

| Cell | Operation | Why |
| --- | --- | --- |
| sensitive & relevant | **alias (A)** | keep the unit, rename the values: concealment *and* referent |
| sensitive & irrelevant | **delete (D)** | contributes nothing; the only cell where a value leaves the prompt entirely |
| relevant & insensitive | **keep (K)** | bears on the question, hides nothing |
| irrelevant & insensitive | **refill (R)** | competes for leftover budget |

`alias` is the departure from prior practice. Redaction gives the same concealment but removes the
referent; aliasing keeps the entity–attribute association while satisfying the constraint on *V*.
Replacing the alias operation by deletion costs **0.1795** F1 overall and **0.3273** on multi-entity
cases — and it costs **0.0013** (not significant) on the topical variant, where the question names no
entity. The rule acts where it is designed to act and is inert elsewhere.

## Step 4 — question-side aliasing and the keyed map

**Question-side aliasing.** Aliasing only the context yields a prompt whose two halves use different
names for the same entity: the context says `[ACCOUNT-A]` while the question still says
`ACCT-4cd02ed8fb3a`. The reader cannot connect them, and a string-presence metric reports the entity as
preserved while the task is destroyed. The question is therefore rewritten through the same map, and
the object of study is the pair *(q′, P)*.

Measured effect on the 112-case set, pipeline otherwise unchanged:

| Arm | question untouched | question aliased | Δ |
| --- | ---: | ---: | ---: |
| `alias` | 0.4933 | 0.6719 | +0.1786 |
| `alias_budget_link` | 0.5048 | 0.7450 | +0.2402 |

It costs nothing in privacy: both halves stay free of the raw value.

**Why the map is keyed.** Alias letters assigned in ascending value order are invertible from rank
alone. An attacker who knows the *set* of true values sorts it and pairs the sorted values with the
aliases in letter order. `src/rank_attack.py` reproduces exactly that attack on 300 cases (4,512
aliases):

| Defender setting | Attack hit rate |
| --- | ---: |
| `PT_ALIAS_KEY` unset | **100.0%** (4512/4512) |
| `PT_ALIAS_KEY` set | **19.1%** (864/4512) |

With k = 5 entities the per-alias random hit rate is 1/k = 20%, so with the key set the attack is no
better than guessing (19.1% vs. 20%). `src/rank_attack.py` reproduces both rows directly; run it with
`data/longctx_max.jsonl` decompressed.
The fix is free: on the same 300 cases, with the environment variable as the only changed input,
`alias_budget_link@0.5` scores 0.9333 → 0.9333 and `full` 0.7200 → 0.7200. On the full 2,728-case
rerun `alias_budget_link@0.50` moves 0.6372 → 0.6520, and the 16 arm×budget combinations that contain
no alias all change by exactly 0.0000 — evidence that the key touches alias assignment only. All
numbers reported here use the keyed assignment.

## Step 5 — budget accounting

The budget is

```
B = floor(r * n_C)
```

and it is **enforced**, not approximated. The surviving set is assembled greedily and then trimmed
against the *actual concatenated text*, because the separator between units costs tokens: a sum of
per-unit token counts is only a **lower bound** on the length of the joined prompt. When the trimmed
set still exceeds *B*, units are discarded in a fixed priority — refill first, then non-anchor relevant
units, then anchor units — so the pipeline degrades instead of failing.

Every budget arm in the paper has **zero overshoot**, and the harness raises instead of silently
overshooting. This sounds like hygiene; it is not. A one-token accounting gap was found this way, and
the first attempted fix made things worse by trimming from the *tail* of the drop order — which is
exactly where the answer-bearing units are ordered. The details, including why a 2,000-case property
test missed it, are in [`src/longctx/README_VERSIONS.md`](../src/longctx/README_VERSIONS.md).

**Refill order runs increasing in redundancy.** For unit *uᵢ*, the redundancy score is the maximum
Jaccard similarity between its token set and that of any other unit. Candidates are visited in
*increasing* redundancy, which keeps the repetitive records. That inverts the usual reading of the
score, and the choice is stated as an empirical regularity with a mechanism, not as a consequence of
the definition: the answer-bearing record is frequently the most redundant one. The two directions are
compared end to end in [Results](results.md#refill-direction) — they differ by −0.6567 F1 on the
two-hop benchmark and by nothing outside it.

## Reference implementation

| Concern | File |
| --- | --- |
| units, labelling, routing, budget fit, prompt assembly | [`src/longctx/e2e_quad.py`](../src/longctx/e2e_quad.py) |
| quadrant router and alias map (`Aliaser`, `apply_replacements`) | [`src/longctx/v4_router.py`](../src/longctx/v4_router.py) |
| predicates (`norm`, `redact`, `EMAIL/PHONE/ACCOUNT`, `exposure_*`) | [`src/longctx/privcheck.py`](../src/longctx/privcheck.py) |
| model-free validation of the routing decisions | [`src/analysis_quad4/validate_fallback.py`](../src/analysis_quad4/validate_fallback.py) |
| the rank-inversion attack | [`src/rank_attack.py`](../src/rank_attack.py) |

`privcheck.py` is the single source of truth for the predicates: no other file carries a copy, because
this project has twice been burned by copies that silently drifted apart.
