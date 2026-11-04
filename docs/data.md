# Data reference

Everything under `data/` is frozen: the files are byte-identified by
[`data/MANIFEST.sha256`](../data/MANIFEST.sha256), so a checkout can be verified with

```bash
cd data && sha256sum -c MANIFEST.sha256      # GNU coreutils / Git Bash
cd data && shasum -a 256 -c MANIFEST.sha256  # macOS
```

| File | Rows | Size | Role |
| --- | ---: | ---: | --- |
| `longctx_cases.jsonl` | 112 | 3.2 MB | **Main set** — main table, budget sweep, cross-model, paraphrases |
| `longctx_max.jsonl.gz` | 2,728 | 15.1 MB | Extension benchmark; drives every number in [Results §1](results.md#extension) |
| `longctx_paraA.jsonl`, `longctx_paraB.jsonl` | 112 each | 3.2 MB | the two mechanical question paraphrases |
| `public_lb.jsonl.gz` | 300 | 4.0 MB | LongBench-derived, **Type A** — topical question, no entity named |
| `public_lb_typeb.jsonl.gz` | 300 | 4.0 MB | LongBench-derived, **Type B** — single-hop, answer in the anchor's record |
| `public_lb_typec.jsonl.gz` | 300 | 4.0 MB | LongBench-derived, **Type C** — two-hop, answer six sentences from the anchor |
| `pc_qa.jsonl.gz` | 300 | 4.0 MB | Perception-Compressor input form of the Type A cases |
| `pc_one_real.jsonl.gz` | 1 | 2 KB | Single real passage, smoke test for the external baseline |
| `pc_tiny.jsonl.gz` | 1 | 0.4 KB | Minimal synthetic passage, dependency smoke test |
| `pc_query_dict.json` | 298 | 87 KB | Question → query map for the Perception-Compressor runs |
| `frozen_samples.jsonl`, `frozen_samples_scattered.jsonl` | 581 | 3.9 MB | frozen sample sets for the V1 clustering/redaction feasibility stage |
| `conflict_cases.jsonl` | — | 315 KB | conflict-resolution cases (the zero-coverage cell) |
| `inscope_*.jsonl` (4 files) | — | 90–188 KB | in-scope (≤512-token) contrast slices |
| `frozen_samples_MANIFEST.json`, `MANIFEST_longctx.json` | — | — | byte provenance of the upstream builds |

All `.jsonl.gz` files are **gzip-compressed JSON Lines**: one JSON object per line. Decompress in place
with `gunzip -k <file>`, which leaves the original untouched. Several scripts expect the decompressed
`data/longctx_max.jsonl`, so decompress before running them.

## Case schema (`longctx_max.jsonl.gz`, `public_lb*.jsonl.gz`)

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | currently `1` |
| `key` | str | case identifier, `longctx-<source>-<kind>-<idx>` |
| `case_kind` | str | `multi_entity` or `needle` |
| `dataset`, `source` | str | originating corpus (`pec_home`, `privacylens`, `simuhome`, `smartbench_tdcs`, `toolmem`, `toolmem_raw`) |
| `context` | str | the excerpt as the model would see it, with `----- entry N -----` markers |
| `context_sha256` | str | SHA-256 of `context` |
| `question` | str | the entity-resolving question |
| `answers` | list[str] | reference answers (usually one) |
| `entity_key`, `entity_type` | str | the attribute family the entity is identified by |
| `entity_values`, `attribute_values` | list[str] | all entities and all attribute values in the excerpt |
| `target_entity` | str | the value the question names — **also a protected value** |
| `target_attribute` | str | the answer value |
| `attribute_key` | str | human-readable attribute name (`persona_id`, `device`, …) |
| `canaries` | list[obj] | protected values injected into this excerpt |
| `guess_baseline` | float | F1 obtainable without reading the context (1 / `n_entities`) |
| `n_entities`, `n_records` | int | construction bookkeeping |
| `entry_spans` | list | `[start, end, record_id]` per record block |
| `target_region_start/end` | int | character span of the answer-bearing block |
| `answer_char_offset`, `answer_value_offset` | int | exact offsets of the answer |
| `words` | int | word count — **never** use this as a token count |

### The `canaries` entries

```json
{"type": "account", "value": "ACCT-214c5d733f00", "marker": "214c5d733f00", "fragment_length": 6}
```

- `value` — the full protected string;
- `marker` — the random hexadecimal payload inside it;
- `fragment_length` — the shortest fragment treated as a leak (recall requirement of the detector,
  not a security parameter of the method).

`src/longctx/privcheck.py` is the single definition of the three value patterns and of the
leak/no-leak verdict.

## Results reference

| Path | Rows / files | Meaning |
| --- | --- | --- |
| `results/percase/*.jsonl.gz` | 755 files, ~997k rows | one row per (case, arm, ratio, generation) |
| `results/percase/quad4/` | 54 shards, 53,912 rows | the gold-free quad4 / DARK runs |
| `results/cells/*.json` | 189 files | per-cell aggregate counts, one file per shard |
| `results/reports/*` | 21 files | generated run reports (human-readable) |

One per-case row:

```json
{"case_key": "longctx-pec_home-multi-0000", "arm": "alias", "ratio": null,
 "case_kind": "multi_entity", "source": "pec_home",
 "gold": "multi_0212_00", "pred": "multi_0212_00", "em": 1, "f1": 1.0,
 "ctx_tokens": 2053, "prompt_tokens": 2129, "target": 2254,
 "n_canaries": 18, "n_entities": 6,
 "ctx_leak_value": 0, "ctx_leak_frag": 0, "out_leak_value": 0,
 "ans_in_ctx": 1, "tgt_entity_present": 1, "tgt_attr_present": 1, "collision": 0}
```

| Column | Meaning |
| --- | --- |
| `arm` | which pipeline produced the row (`full`, `alias`, `delete`, `budget_cluster`, …, `quad4`) |
| `ratio` | budget ratio, or `null` for the length-preserving arms |
| `gold` / `pred` | reference answer and the reader's output |
| `em` / `f1` | SQuAD v2 exact match and token F1 |
| `ctx_tokens` / `prompt_tokens` / `target` | measured tokens of the emitted context, the whole prompt, and the budget target |
| `ctx_leak_*` / `out_leak_*` | canary recoverable from the *context*, and canary *emitted* by the model |
| `ans_in_ctx` | reference answer still present in the emitted prompt |
| `tgt_entity_present` | the entity the question names still present (the referent) |
| `collision` | two distinct protected values mapped to one replacement token |

`ctx_*` and `out_*` are deliberately separate: an arm can keep the answer reachable while never
emitting the value, and conflating the two hides exactly the failure mode this paper is about.

## Not versioned

| Excluded | Why |
| --- | --- |
| model weights (`Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`, the XLM-R encoder) | 13–29 GB each; fetched from the model hub at run time |
| the project's private result database (≈ 6 GB) | contains the published baselines' raw payloads, which are theirs to distribute |
| the six source corpora in raw form | third-party licences; only the derived, canary-bearing excerpts ship here |

Nothing else is missing: every file that a number in this repository depends on is present, and
`data/MANIFEST.sha256` pins all 20 data files.
