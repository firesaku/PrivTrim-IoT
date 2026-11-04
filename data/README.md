# `data/` — frozen benchmarks

Everything here is byte-identified by [`MANIFEST.sha256`](MANIFEST.sha256) (20 files). To verify a
checkout:

```bash
cd data
sha256sum -c MANIFEST.sha256          # GNU coreutils
shasum -a 256 -c MANIFEST.sha256      # macOS
# Windows (no coreutils): certutil -hashfile <file> SHA256, compare by eye
```

| File | Cases | What it is |
| --- | ---: | --- |
| `longctx_cases.jsonl` | 112 | **main set** — main table, budget sweep, cross-model, paraphrase experiments |
| `longctx_max.jsonl.gz` | 2,728 | extension benchmark — six IoT/smart-home corpora, canary-bearing |
| `longctx_paraA.jsonl`, `longctx_paraB.jsonl` | 112 each | the two mechanical question paraphrases |
| `public_lb.jsonl.gz` | 300 | LongBench-derived, **Type A**: topical question, no entity named |
| `public_lb_typeb.jsonl.gz` | 300 | LongBench-derived, **Type B**: single-hop, answer in the anchor's record |
| `public_lb_typec.jsonl.gz` | 300 | LongBench-derived, **Type C**: two-hop, answer six sentences from the anchor |
| `frozen_samples.jsonl`, `frozen_samples_scattered.jsonl` | 581 | frozen sample sets for the V1 clustering/redaction stage |
| `conflict_cases.jsonl` | — | conflict-resolution cases (the zero-coverage cell) |
| `inscope_ext_arms.jsonl`, `inscope_multi_cases.jsonl`, `inscope_target_first.jsonl`, `inscope_target_last.jsonl` | — | in-scope (≤512-token) contrast slices |
| `pc_qa.jsonl.gz` | 300 | Type A cases in Perception-Compressor input form |
| `pc_one_real.jsonl.gz` | 1 | single real passage, smoke test for that baseline |
| `pc_tiny.jsonl.gz` | 1 | minimal synthetic passage, dependency smoke test |
| `pc_query_dict.json` | 298 | question → query map for the Perception-Compressor runs |
| `frozen_samples_MANIFEST.json`, `MANIFEST_longctx.json` | — | byte provenance of the upstream builds |

All `.jsonl.gz` files are gzip-compressed JSON Lines. `gunzip -k <file>` decompresses in place and
leaves the original; several scripts expect `longctx_max.jsonl` uncompressed.

**All injected sensitive values are canaries** — `…@example.invalid`, `+1-202-555-xxxx`,
`ACCT-<hex>` — not real personal data. Schema per field: [`docs/data.md`](../docs/data.md).

Model weights and the raw source corpora are not versioned here; see
[`docs/data.md`](../docs/data.md#not-versioned).
