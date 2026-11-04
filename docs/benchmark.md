# Benchmark

## Construction

The benchmark is built from six public IoT and smart-home corpora that appear in the retrieval and
long-context literature: `pec_home`, `privacylens`, `simuhome`, `smartbench_tdcs`, `toolmem`,
`toolmem_raw`.

For each source we assemble excerpts in which records for several entities of the same type are
interleaved, and we **inject synthetic protected values** — account identifiers, contact addresses,
phone numbers — into records that already exist, rather than into separate carrier records. Reference
answers are derived deterministically from the injected structure, so **no answer is authored by hand**.

The injected values are **canaries**, not real personal data. Public corpora cannot carry real PII, and
a canary makes leakage measurable exactly. The consequence is stated openly: the privacy claim concerns
the recoverability of *these* values, not of arbitrary personal information that may co-occur in real
logs.

## Question form

Each case poses an entity-resolving question: it names an account and asks for an attribute recorded
for it (which device is associated, which persona identifier is stored, and so on). Two families:

| Family | Layout | Anchor the question names |
| --- | --- | --- |
| `multi_entity` | the same question template applies to six accounts; the target is distinguished only by its identifier | an account identifier |
| `needle` | the question names a specific record identifier and asks for a field value inside it | a record identifier |

The block-level statistics that motivate the method come from this layout:

| Measure | multi_entity | needle |
| --- | ---: | ---: |
| answer lies in the *same block* as the anchor mention | 100% | 99.7% |
| anchor identifier and answer sit on the *same line* | 0% | 34.6% |

## Anchor forms

The anchor key is resolved differently depending on what the question names. All six forms occurring in
the extension set are handled, and the anchor string is present in the context in **every** case
(2,728/2,728):

| `anchor_kind` | Example question form | n |
| --- | --- | ---: |
| none (entity field) | *for account* `ACCT-…` | 1500 |
| `id` | *for record id* `pec_home_…` | 316 |
| `name` | names an entity directly | 300 |
| `session` | names a session | 300 |
| `entry` | *for entry 3* | 281 |
| `native` | native identifier | 31 |

## Sizes

| Set | Cases | Used for |
| --- | ---: | --- |
| **main** | 112 | main table, budget curve, cross-model, paraphrasing — 20 cases from each of five sources and 12 from the sixth |
| **extension** | 2,728 | confirms the main-set conclusions survive an order-of-magnitude increase in size and a different source mix |

Both sets share the same question templates.

> **Key hygiene.** The two sets reuse the same case identifiers for *different* content. Joining
> results across them by identifier silently mixes records. Every analysis reported here resolves
> results within a single set and verifies the join by comparing the uncompressed-context arm against a
> reference value computed from the same set. `src/analysis_quad4/reproduce.py` performs that check
> explicitly (112/112 on the shipped data).

## Public-benchmark variants

To test the rule outside our own construction we derive three variants from LongBench passages, 300
cases each, with the same injected canaries:

| Variant | Question | Placement of the answer |
| --- | --- | --- |
| **topical** (Type A) | the original LongBench question; names no entity | unchanged |
| **single-hop** (Type B) | rewritten to name an account | in the same record as the account |
| **two-hop** (Type C) | rewritten to name an account | six sentences away from the account mention |

The two-hop variant is the one that forces the reader to follow record structure rather than a local
window, and it is where the largest single effect in the paper appears
([Results](results.md#refill-direction)).

The topical variant is reported as *outside* what the rule addresses, not as a failure of the
mechanism: with no entity to locate, DARK degenerates to global aliasing plus redundancy ordering and
loses 0.0665 against the uncompressed prompt.

## What a case record contains

`data/longctx_max.jsonl.gz` — one JSON object per line:

| Field | Meaning |
| --- | --- |
| `key` | case identifier, `longctx-<source>-<kind>-<idx>` |
| `case_kind` | `multi_entity` or `needle` |
| `source`, `dataset` | originating corpus |
| `context` | the excerpt exactly as the model would see it, with `----- entry N -----` markers |
| `context_sha256` | hash of `context`, for integrity |
| `question` | the entity-resolving question |
| `answers` | list of reference answer strings |
| `target_entity` | the value the question names — *also* a protected value |
| `target_attribute` | the answer value |
| `attribute_key`, `attribute_values` | the attribute under test and all of its values across entities |
| `canaries` | the protected values injected into this excerpt: `{type, value, marker, fragment_length}` |
| `guess_baseline` | F1 obtainable without reading the context (1 / n_entities) |
| `entry_spans`, `target_region_start/end` | block boundaries and the answer region |
| `words` | word count (token counting is done with the real tokenizer, never with this) |

Full schema table, file inventory and integrity hashes: [Data reference](data.md).

## Known caveats

1. **`maxlen` truncation matters.** In one earlier run `maxlen=8192` truncated 47 of 112 full contexts
   (42%), because the builder estimated length from word count. The uncompressed arm then collapses to
   F1 0.0019 on those cases *while `ans_in_ctx` stays 1.000*. Comparisons against `full` must use the
   65-case subset that actually fits. Later runs use `maxlen=24576` and a policy that raises rather
   than truncates.
2. **Single seed per configuration.** Case generation is byte-reproducible; model generation is not.
   Variance is reported through paired bootstrap intervals over cases, not across seeds.
3. **Canaries, not real PII** (see above).
