# PrivTrim-IoT — long-context benchmark results

Job **10601756** on Almesbar (`gpu-11-2`, Tesla V100-PCIE-32GB): state `COMPLETED`, exit `0:0`, elapsed `01:14:58`. Harness `e2e_longctx_r2.py`, sha256 `b6a584f845b8816df5860fd166cbeb3ac1d411eea991c77d89eeb80b136188cf`. Qwen2.5-7B-Instruct fp16, `--sel lex`, batch 4, maxlen 8192, max-new 64, seed 20261015.

112 cases x 18 (arm, ratio) = 2016 generations; all 2016 per-case rows present.

## Headline

**1. `budget_cluster` vs `budget_cluster_link` is the dominant effect, and `ans_in_ctx` cannot see it.** Multi-entity cases at ratio 0.50, restricted to the 65 cases where the full context fits in `maxlen=8192`:

| arm | n | F1 | target_entity_present | ans_in_ctx | ctx_tokens |
|---|---|---|---|---|---|
| budget_cluster | 25 | 0.0000 | 0.000 | 1.000 | 2097 |
| budget_cluster_link | 25 | 0.6000 | 1.000 | 1.000 | 2099 |

Both arms keep the gold answer string (`ans_in_ctx = 1.000`) at the same budget. Only `budget_cluster_link` keeps the *entity the question asks about* (`0.000` vs `1.000`). The pure-redundancy selector drops the near-identical entity blocks first because they are the most redundant units in the log: the secret survives, the link does not, and the model answers `UNKNOWN`.

**2. The apparent "+0.31 F1 over uncompressed" is a truncation artefact, not a compression win.** 47 of 112 full contexts (42%) exceed `maxlen=8192` and are truncated:

| subset | n | full F1 | full EM | full ans_in_ctx |
|---|---|---|---|---|
| fits in 8192 | 65 | 0.7002 | 0.6615 | 1.000 |
| truncated | 47 | 0.0019 | 0.0000 | 1.000 |

`full` is the strongest arm on the cases it can actually see (F1=0.7002); on truncated cases it collapses to F1=0.0019 while `ans_in_ctx` stays 1.000, because the answer lies past the truncation point. Comparisons against `full` must use the 65-case subset.

## Aggregate per (arm, budget), all 112 cases

`leak` = sum(out_leak_value)/sum(n_canaries) — canaries the model *emitted*.

| arm | budget | n | EM | F1 | ctx_tokens | leak | coll | ans_in_ctx |
|---|---|---|---|---|---|---|---|---|
| full | ref | 112 | 0.3839 | 0.4072 | 7964 | 0.000000 | 0.000 | 1.000 |
| delete | ref | 112 | 0.2411 | 0.2594 | 7845 | 0.000000 | 0.357 | 0.982 |
| alias | ref | 112 | 0.2411 | 0.2547 | 7824 | 0.000000 | 0.000 | 1.000 |
| budget_cluster | 0.25 | 112 | 0.3929 | 0.4107 | 1630 | 0.000000 | 0.000 | 0.911 |
| budget_cluster | 0.50 | 112 | 0.4643 | 0.4688 | 3705 | 0.000000 | 0.000 | 1.000 |
| budget_cluster | 0.75 | 112 | 0.3750 | 0.3795 | 5578 | 0.000000 | 0.000 | 1.000 |
| budget_cluster_link | 0.25 | 112 | 0.6250 | 0.6545 | 1633 | 0.000000 | 0.000 | 0.911 |
| budget_cluster_link | 0.50 | 112 | 0.7054 | 0.7182 | 3706 | 0.000000 | 0.000 | 1.000 |
| budget_cluster_link | 0.75 | 112 | 0.5714 | 0.5772 | 5581 | 0.000000 | 0.000 | 1.000 |
| budget_random | 0.25 | 112 | 0.1250 | 0.1250 | 1588 | 0.000000 | 0.000 | 0.223 |
| budget_random | 0.50 | 112 | 0.2143 | 0.2188 | 3659 | 0.000000 | 0.000 | 0.455 |
| budget_random | 0.75 | 112 | 0.3661 | 0.3709 | 5646 | 0.000000 | 0.000 | 0.723 |
| delete_budget | 0.25 | 112 | 0.3839 | 0.4018 | 1629 | 0.000000 | 0.357 | 0.893 |
| delete_budget | 0.50 | 112 | 0.4643 | 0.4688 | 3696 | 0.000000 | 0.357 | 0.982 |
| delete_budget | 0.75 | 112 | 0.3482 | 0.3534 | 5565 | 0.000000 | 0.357 | 0.982 |
| alias_budget | 0.25 | 112 | 0.3929 | 0.4107 | 1630 | 0.000000 | 0.000 | 0.911 |
| alias_budget | 0.50 | 112 | 0.4821 | 0.4866 | 3708 | 0.000000 | 0.000 | 1.000 |
| alias_budget | 0.75 | 112 | 0.3571 | 0.3622 | 5579 | 0.000000 | 0.000 | 1.000 |

## Fair comparison: the 65 cases where `full` fits in 8192

| arm | budget | n | EM | F1 | ctx_tokens | leak |
|---|---|---|---|---|---|---|
| full | ref | 65 | 0.6615 | 0.7002 | 4726 | 0.000000 |
| delete | ref | 65 | 0.4154 | 0.4470 | 4602 | 0.000000 |
| alias | ref | 65 | 0.4154 | 0.4389 | 4584 | 0.000000 |
| budget_cluster | 0.25 | 65 | 0.4154 | 0.4154 | 960 | 0.000000 |
| budget_cluster | 0.50 | 65 | 0.4462 | 0.4538 | 2223 | 0.000000 |
| budget_cluster | 0.75 | 65 | 0.4769 | 0.4846 | 3326 | 0.000000 |
| budget_cluster_link | 0.25 | 65 | 0.6462 | 0.6662 | 962 | 0.000000 |
| budget_cluster_link | 0.50 | 65 | 0.7077 | 0.7299 | 2224 | 0.000000 |
| budget_cluster_link | 0.75 | 65 | 0.7077 | 0.7159 | 3328 | 0.000000 |
| budget_random | 0.25 | 65 | 0.0923 | 0.0923 | 906 | 0.000000 |
| budget_random | 0.50 | 65 | 0.1846 | 0.1923 | 2174 | 0.000000 |
| budget_random | 0.75 | 65 | 0.4462 | 0.4544 | 3378 | 0.000000 |
| delete_budget | 0.25 | 65 | 0.4000 | 0.4000 | 957 | 0.000000 |
| delete_budget | 0.50 | 65 | 0.4462 | 0.4538 | 2219 | 0.000000 |
| delete_budget | 0.75 | 65 | 0.4308 | 0.4385 | 3320 | 0.000000 |
| alias_budget | 0.25 | 65 | 0.4154 | 0.4154 | 959 | 0.000000 |
| alias_budget | 0.50 | 65 | 0.4615 | 0.4692 | 2228 | 0.000000 |
| alias_budget | 0.75 | 65 | 0.4462 | 0.4538 | 3335 | 0.000000 |

Best budget arm on the fair subset: **budget_cluster_link @ 0.50, F1=0.7299** vs `full` F1=0.7002 (+0.0297). At the same ratio, `budget_cluster` is F1=0.4538 and `budget_random` is F1=0.1923.

## By case_kind on the fair subset

| case_kind | arm | budget | n | F1 | target_entity_present | ans_in_ctx |
|---|---|---|---|---|---|---|
| multi_entity | full | ref | 25 | 0.6533 | 1.000 | 1.000 |
| multi_entity | delete | ref | 25 | 0.0000 | 1.000 | 1.000 |
| multi_entity | alias | ref | 25 | 0.0000 | 1.000 | 1.000 |
| multi_entity | budget_cluster | 0.25 | 25 | 0.0000 | 0.000 | 0.800 |
| multi_entity | budget_cluster | 0.50 | 25 | 0.0000 | 0.000 | 1.000 |
| multi_entity | budget_cluster | 0.75 | 25 | 0.0800 | 0.080 | 1.000 |
| multi_entity | budget_cluster_link | 0.25 | 25 | 0.5600 | 0.800 | 0.800 |
| multi_entity | budget_cluster_link | 0.50 | 25 | 0.6000 | 1.000 | 1.000 |
| multi_entity | budget_cluster_link | 0.75 | 25 | 0.6400 | 1.000 | 1.000 |
| multi_entity | budget_random | 0.25 | 25 | 0.0000 | 0.080 | 0.240 |
| multi_entity | budget_random | 0.50 | 25 | 0.0800 | 0.400 | 0.440 |
| multi_entity | budget_random | 0.75 | 25 | 0.3200 | 0.760 | 0.680 |
| multi_entity | delete_budget | 0.25 | 25 | 0.0000 | 0.280 | 0.800 |
| multi_entity | delete_budget | 0.50 | 25 | 0.0000 | 0.360 | 1.000 |
| multi_entity | delete_budget | 0.75 | 25 | 0.0000 | 0.360 | 1.000 |
| multi_entity | alias_budget | 0.25 | 25 | 0.0000 | 0.000 | 0.800 |
| multi_entity | alias_budget | 0.50 | 25 | 0.0400 | 0.280 | 1.000 |
| multi_entity | alias_budget | 0.75 | 25 | 0.0000 | 0.560 | 1.000 |
| needle | full | ref | 40 | 0.7295 | 1.000 | 1.000 |
| needle | delete | ref | 40 | 0.7265 | 1.000 | 0.950 |
| needle | alias | ref | 40 | 0.7131 | 1.000 | 1.000 |
| needle | budget_cluster | 0.25 | 40 | 0.6750 | 1.000 | 0.950 |
| needle | budget_cluster | 0.50 | 40 | 0.7375 | 1.000 | 1.000 |
| needle | budget_cluster | 0.75 | 40 | 0.7375 | 1.000 | 1.000 |
| needle | budget_cluster_link | 0.25 | 40 | 0.7325 | 1.000 | 0.950 |
| needle | budget_cluster_link | 0.50 | 40 | 0.8111 | 1.000 | 1.000 |
| needle | budget_cluster_link | 0.75 | 40 | 0.7634 | 1.000 | 1.000 |
| needle | budget_random | 0.25 | 40 | 0.1500 | 1.000 | 0.150 |
| needle | budget_random | 0.50 | 40 | 0.2625 | 1.000 | 0.400 |
| needle | budget_random | 0.75 | 40 | 0.5385 | 1.000 | 0.750 |
| needle | delete_budget | 0.25 | 40 | 0.6500 | 1.000 | 0.900 |
| needle | delete_budget | 0.50 | 40 | 0.7375 | 1.000 | 0.950 |
| needle | delete_budget | 0.75 | 40 | 0.7125 | 1.000 | 0.950 |
| needle | alias_budget | 0.25 | 40 | 0.6750 | 1.000 | 0.950 |
| needle | alias_budget | 0.50 | 40 | 0.7375 | 1.000 | 1.000 |
| needle | alias_budget | 0.75 | 40 | 0.7375 | 1.000 | 1.000 |

## Per source: cluster vs link vs full (all 112)

| source | budget | n | cluster F1 | cluster tent | link F1 | link tent | full F1 |
|---|---|---|---|---|---|---|---|
| pec_home | 0.25 | 20 | 0.5000 | 0.600 | 0.9750 | 1.000 | 0.8750 |
| pec_home | 0.50 | 20 | 0.5750 | 0.600 | 0.9250 | 1.000 | 0.8750 |
| pec_home | 0.75 | 20 | 0.6250 | 0.650 | 0.9750 | 1.000 | 0.8750 |
| privacylens | 0.25 | 20 | 0.4500 | 0.600 | 0.5900 | 1.000 | 0.6067 |
| privacylens | 0.50 | 20 | 0.4500 | 0.600 | 0.6900 | 1.000 | 0.6067 |
| privacylens | 0.75 | 20 | 0.4500 | 0.600 | 0.6000 | 1.000 | 0.6067 |
| simuhome | 0.25 | 20 | 0.1000 | 0.600 | 0.3000 | 0.750 | 0.4500 |
| simuhome | 0.50 | 20 | 0.4500 | 0.600 | 0.7000 | 1.000 | 0.4500 |
| simuhome | 0.75 | 20 | 0.5000 | 0.600 | 0.7500 | 1.000 | 0.4500 |
| smartbench_tdcs | 0.25 | 20 | 0.7500 | 0.600 | 0.8500 | 1.000 | 0.0000 |
| smartbench_tdcs | 0.50 | 20 | 0.6000 | 0.600 | 0.7000 | 1.000 | 0.0000 |
| smartbench_tdcs | 0.75 | 20 | 0.0500 | 0.650 | 0.0500 | 1.000 | 0.0000 |
| toolmem | 0.25 | 12 | 0.5833 | 1.000 | 0.5000 | 1.000 | 0.4901 |
| toolmem | 0.50 | 12 | 0.4167 | 1.000 | 0.4286 | 1.000 | 0.4901 |
| toolmem | 0.75 | 12 | 0.5000 | 1.000 | 0.4196 | 1.000 | 0.4901 |
| toolmem_raw | 0.25 | 20 | 0.1500 | 0.600 | 0.6500 | 1.000 | 0.0543 |
| toolmem_raw | 0.50 | 20 | 0.3000 | 0.600 | 0.7500 | 1.000 | 0.0543 |
| toolmem_raw | 0.75 | 20 | 0.2000 | 0.700 | 0.6056 | 1.000 | 0.0543 |

## Verification

- **Budget accounting.** Per-row overshoot = **0** of 1680 budget rows; aggregate `budget_accounting` entries over target = **0**. The harness raises on overshoot, so zero here means the guard ran and never fired.
- **Achieved vs target tokens** (means from `budget_accounting`):
  - ratio 0.25: budget_cluster 1630/1991, budget_cluster_link 1633/1991, budget_random 1588/1991, delete_budget 1629/1991, alias_budget 1630/1991
  - ratio 0.50: budget_cluster 3705/3982, budget_cluster_link 3706/3982, budget_random 3659/3982, delete_budget 3696/3982, alias_budget 3708/3982
  - ratio 0.75: budget_cluster 5578/5973, budget_cluster_link 5581/5973, budget_random 5646/5973, delete_budget 5565/5973, alias_budget 5579/5973
- **Output leak.** `out_leak_value` is **0.000000 for every arm and every budget** — no arm ever emitted a canary.
  - `full`: ctx_leak_value=1.000000, ctx_leak_frag=1.000000, out_leak_value=0.000000, ans_in_ctx=1.0000
  - `delete`: ctx_leak_value=0.000000, ctx_leak_frag=0.000000, out_leak_value=0.000000, ans_in_ctx=0.9821
  - `alias`: ctx_leak_value=0.000000, ctx_leak_frag=0.000000, out_leak_value=0.000000, ans_in_ctx=1.0000
- **`delete` destroys real answers.** `ans_in_ctx` for `delete` = 0.9821 vs 1.0000 for `full` and `alias`: it removes the gold string in **2 of 112** cases, because the gold is a real address/identifier present in the source data and matched by `privcheck.EMAIL`. `alias` removes none. This reproduces the dry-run finding on the real model.
- **`full` is highest only within its own block.** Length-preserving arms: `full` F1=0.4072 > `delete` F1=0.2594 > `alias` F1=0.2547. Budget arms are a separate block, printed separately by the harness report.
- **Below `guess_baseline`.** 491 of 1680 budget-arm rows (29.2%). **79.2% of those return `UNKNOWN`** and 0.0% return an empty string — the model abstains rather than confabulating, i.e. the "information looks deleted" pattern already recorded in this project. This is a finding, not a harness bug.

## Row-level evidence

Representative case `longctx-pec_home-multi-0000` at ratio 0.50:

- Question: In the log above, which persona id is recorded for the account ACCT-5ee052b431dc?
- gold = `multi_0212_00`, target_entity = `ACCT-5ee052b431dc`
- `budget_cluster`     : ctx=1119, target_entity_present=0, ans_in_ctx=1, pred=`UNKNOWN`, F1=0.000
- `budget_cluster_link`: ctx=1123, target_entity_present=1, ans_in_ctx=1, pred=`multi_0212_00`, F1=1.000
- `full`               : ctx=2254, pred=`multi_0212_00`, F1=1.000

The two compressed contexts are within 4 tokens of each other (1119 vs 1123). The gold string is in *both*. The only difference is whether the entity named by the question survived — and that alone moves F1 from 0.000 to 1.000.

## Could not verify / caveats

1. **`maxlen=8192` truncates 47 of 112 full contexts (42%).** The builder estimated the longest case at ~6.5k tokens from `words x 1.33`; the real Qwen tokenizer gives up to 16344 tokens for `smartbench_tdcs` — about 2.5x the estimate. `--dry-run` cannot catch this because `est_tok` is a word-count proxy. A future run should raise `maxlen` (Qwen2.5-7B-Instruct supports 32768) or cap case length.
2. **The `full` baseline is therefore not a clean uncompressed baseline.** Use the 65-case fair subset for honest comparisons; the all-112 numbers are also reported.
3. **Not verified: whether raising `maxlen` changes the `budget_cluster` vs `budget_cluster_link` contrast.** That contrast is a unit-selection effect and should be independent of truncation, but this run cannot prove it.
4. **Single seed, single model, single card** (`--seed 20261015`); no variance estimate. Case generation is byte-reproducible, generation is not.
5. A second job ran on the same node on a different GPU during part of this run; no memory contention was observed (GPU 0: 25.4 GiB, GPU 1: 19.1 GiB of 32 GiB).

