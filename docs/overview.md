# Overview

## The setting

Operational logs from consumer IoT deployments record what each device did, when, and for which
account. They are long — an excerpt in our benchmark reaches 19,838 tokens — repetitive, and dense
with identifiers: account handles, device serials, contact fields. They are also a natural substrate
for question answering. An operator asks *which device is associated with account `ACCT-…`?* and
expects a short answer.

The property that makes such a question answerable is what makes it unsafe to forward verbatim: the
question names an entity, and that entity is itself a protected value.

## Why the two existing remedies both fail

| Remedy | Protected value | Handle the reader needs | Task |
| --- | --- | --- | --- |
| **Redaction** (delete every value) | gone | gone | destroyed: the model cannot tell 200 similar records apart |
| **Compression** (keep the short version) | forwarded | kept | destroyed differently: the answer-bearing record is dropped |
| **DARK** (alias and route) | gone | kept | **preserved at half the length** |

Compression fails quietly. Under a redundancy-ordered rule the record carrying the answer survives in
only 47.5% of cases, and in 37.2% of needle cases — *below* the budget cut, i.e. worse than an
uninformative rule. The reason is an anti-correlation that the compression literature does not report:
the answer-bearing record is often the *most* redundant unit, because the answer value repeats across
distractor records, so a redundancy score gives it no priority.

Protecting the anchor **line** (the unit that mentions the queried entity) repairs the pointer but not
the content, because the identifier line and the answer line are different units in 100% of
multi-entity cases.

## The observation

Records in these excerpts are grouped into **blocks**: an entity's identifying line together with the
attributes recorded for it. The block that contains the queried entity also contains the answer in
**100%** of multi-entity cases and **99.7%** of needle cases. Finding that block needs no answer key —
the question already names the entity, and the block is whatever contains that mention.

So: keep the block, and replace the protected values inside it with stable aliases. The association
survives; the value does not.

## The rule

Each unit gets one of four operations, chosen from the pair *(is it sensitive?, is it relevant?)*:

| | relevant | not relevant |
| --- | --- | --- |
| **sensitive** | **A** — alias | **D** — delete |
| **not sensitive** | **K** — keep | **R** — refill (compete for leftover budget) |

That gives the method its name, **DARK**. Two further pieces make it work:

- **Question-side aliasing.** Aliasing the context alone produces a prompt whose halves name the same
  entity differently. The question is rewritten through the *same* map, so the object handed to the
  reader is the pair *(q′, P)*.
- **A keyed alias map.** Letters handed out in value order are invertible from rank alone. A keyed
  permutation removes that signal.

Details: [Method](method.md).

## Contributions

1. **A failure mode the compression literature does not report.** When the queried entity is also the
   protected object, redundancy ordering carries almost no signal about which record the question is
   about. Quantified on 2,728 cases; protecting the anchor line alone repairs only half the gap.
2. **A training-free routing rule that closes it.** Alias sensitive-and-relevant units instead of
   deleting them; get relevance from the question by locating the record block that contains the
   queried entity. No reference answer, no training, no model access.
3. **An evaluation that separates privacy from utility at equal budget.** Two benchmark sizes, three
   public-benchmark variants, five budgets, two readers, two paraphrases, twelve baselines re-run on
   the same cases, with paired bootstrap intervals on every difference reported.

## What this repository is

The complete PrivTrim project: the method, every benchmark it was evaluated on, the per-case evidence
behind every number, the manuscript, and the earlier lab-notebook documentation (in Chinese) under
`docs/`. It is organised as an English-language release, so the V1–V3 mechanism stage now lives in
`src/` alongside the long-context stage that produced the paper's results. [Status](status.md) is the
inventory.

## Reading order

1. [Method](method.md) — the rule, the budget accounting, the failure it avoids
2. [Benchmark](benchmark.md) — how the canary-bearing excerpts are built
3. [Results](results.md) — every reported table
4. [Reproduction](reproduction.md) — how to check the numbers yourself, on a CPU
5. [Evaluation](evaluation.md) — how to re-run the reader
