# 长上下文臂 model-free 预览（dry-run / est tokens；**不是模型结果**）

这四张表只用 e2e_longctx.py 的 `--dry-run` 就能算出来，不需要 GPU/模型。
它们给出每条臂在"答案还在不在上下文里 / 实体链接还在不在"上的 model-free 上界，
也是 GPU 主实验的预期方向；真实 EM/F1 必须在 HPC 上用 7B 重跑。
g = gold_in_ctx，te = target_entity_present，coll = collision，leak = canary 整值泄漏

**gold_in_ctx（答案串是否还在臂上下文里）**

| kind | budget | full | delete | alias | b_cluster | b_cluster_link | b_random | del_budget | ali_budget |
|---|---|---|---|---|---|---|---|---|---|
| needle | 0.25 | — | — | — | 95.8 | 95.8 | 19.4 | 93.1 | 95.8 |
| needle | 0.50 | — | — | — | 100.0 | 100.0 | 37.5 | 97.2 | 100.0 |
| needle | 0.75 | — | — | — | 100.0 | 100.0 | 66.7 | 97.2 | 100.0 |
| needle | ref | 100.0 | 97.2 | 100.0 | — | — | — | — | — |
| multi_entity | 0.25 | — | — | — | 87.5 | 87.5 | 30.0 | 87.5 | 87.5 |
| multi_entity | 0.50 | — | — | — | 100.0 | 100.0 | 60.0 | 100.0 | 100.0 |
| multi_entity | 0.75 | — | — | — | 100.0 | 100.0 | 77.5 | 100.0 | 100.0 |
| multi_entity | ref | 100.0 | 100.0 | 100.0 | — | — | — | — | — |

**target_entity_present（目标实体的表面 token 是否还在）**

| kind | budget | full | delete | alias | b_cluster | b_cluster_link | b_random | del_budget | ali_budget |
|---|---|---|---|---|---|---|---|---|---|
| multi_entity | 0.25 | — | — | — | 0.0 | 87.5 | 17.5 | 25.0 | 0.0 |
| multi_entity | 0.50 | — | — | — | 0.0 | 100.0 | 50.0 | 30.0 | 5.0 |
| multi_entity | 0.75 | — | — | — | 5.0 | 100.0 | 75.0 | 35.0 | 42.5 |
| multi_entity | ref | 100.0 | 100.0 | 100.0 | — | — | — | — | — |

**collision（多个不同实体落到同一表面 token 的比例）**

| kind | budget | full | delete | alias | b_cluster | b_cluster_link | b_random | del_budget | ali_budget |
|---|---|---|---|---|---|---|---|---|---|
| needle | 0.25 | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| needle | 0.50 | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| needle | 0.75 | — | — | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| needle | ref | 0.0 | 0.0 | 0.0 | — | — | — | — | — |
| multi_entity | 0.25 | — | — | — | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| multi_entity | 0.50 | — | — | — | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| multi_entity | 0.75 | — | — | — | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| multi_entity | ref | 0.0 | 100.0 | 0.0 | — | — | — | — | — |

**ctx_leak_value（canary 整值残留比例；分母 = canary 数）**

| kind | budget | full | delete | alias | b_cluster | b_cluster_link | b_random | del_budget | ali_budget |
|---|---|---|---|---|---|---|---|---|---|
| needle | 0.25 | — | — | — | 0.0 | 0.0 | 24.1 | 0.0 | 0.0 |
| needle | 0.50 | — | — | — | 1.5 | 1.5 | 48.2 | 0.0 | 0.0 |
| needle | 0.75 | — | — | — | 8.3 | 8.3 | 71.8 | 0.0 | 0.0 |
| needle | ref | 100.0 | 0.0 | 0.0 | — | — | — | — | — |
| multi_entity | 0.25 | — | — | — | 0.0 | 6.3 | 26.1 | 0.0 | 0.0 |
| multi_entity | 0.50 | — | — | — | 0.0 | 7.2 | 51.2 | 0.0 | 0.0 |
| multi_entity | 0.75 | — | — | — | 4.5 | 11.4 | 71.9 | 0.0 | 0.0 |
| multi_entity | ref | 100.0 | 0.0 | 0.0 | — | — | — | — | — |
