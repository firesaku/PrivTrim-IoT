# 外部基线 vs 内部臂（同一批 40 个 multi_entity 案例）

锚点校验：集群 full F1=0.6083，本地重跑 F1=0.6083 —— **一致**，故可合并。

| 臂 | 来源 | n | F1 | EM | ctx_tokens | gold_in_ctx |
|---|---|---:|---:|---:|---:|---:|
| `budget_cluster_link @0.75` | internal | 40 | 0.6500 | 0.6500 | 5414 | 1.000 |
| `budget_cluster_link @0.25` | internal | 40 | 0.6250 | 0.6250 | 1592 | 0.875 |
| `ext_full` | external | 40 | 0.6083 | 0.6000 | 8154 | 1.000 |
| `full(ref)` | internal | 40 | 0.6083 | 0.6000 | 8154 | 1.000 |
| `budget_cluster_link @0.50` | internal | 40 | 0.5500 | 0.5500 | 3726 | 1.000 |
| `ext_provence` | external | 40 | 0.4750 | 0.4750 | 4077 | 0.900 |
| `ext_provence_native` | external | 40 | 0.4500 | 0.4500 | 3902 | 0.900 |
| `ext_quito_x` | external | 29 | 0.3793 | 0.3448 | 3409 | 0.724 |
| `budget_random @0.75` | internal | 40 | 0.3750 | 0.3750 | 5684 | 0.775 |
| `ext_cpc_native` | external | 40 | 0.3750 | 0.3750 | 3444 | 0.800 |
| `ext_cpc` | external | 40 | 0.3500 | 0.3500 | 4077 | 0.850 |
| `ext_llmlingua2` | external | 40 | 0.3417 | 0.3250 | 4077 | 0.700 |
| `ext_quito_x_native` | external | 29 | 0.3103 | 0.2759 | 3405 | 0.724 |
| `ext_llmlingua2_native` | external | 40 | 0.2667 | 0.2500 | 4026 | 0.675 |
| `budget_random @0.50` | internal | 40 | 0.1500 | 0.1500 | 3658 | 0.600 |
| `alias_budget @0.25` | internal | 40 | 0.1250 | 0.1250 | 1586 | 0.875 |
| `budget_cluster @0.25` | internal | 40 | 0.1250 | 0.1250 | 1586 | 0.875 |
| `delete_budget @0.25` | internal | 40 | 0.1250 | 0.1250 | 1585 | 0.875 |
| `budget_cluster @0.75` | internal | 40 | 0.0750 | 0.0750 | 5406 | 1.000 |
| `budget_random @0.25` | internal | 40 | 0.0500 | 0.0500 | 1537 | 0.300 |
| `alias_budget @0.50` | internal | 40 | 0.0250 | 0.0250 | 3728 | 1.000 |
| `alias @ref` | internal | 40 | 0.0000 | 0.0000 | 8000 | 1.000 |
| `alias_budget @0.75` | internal | 40 | 0.0000 | 0.0000 | 5429 | 1.000 |
| `budget_cluster @0.50` | internal | 40 | 0.0000 | 0.0000 | 3721 | 1.000 |
| `delete @ref` | internal | 40 | 0.0000 | 0.0000 | 8026 | 1.000 |
| `delete_budget @0.50` | internal | 40 | 0.0000 | 0.0000 | 3713 | 1.000 |
| `delete_budget @0.75` | internal | 40 | 0.0000 | 0.0000 | 5401 | 1.000 |
| `ext_dac` ⚠️ | external | 1 | 0.0000 | 0.0000 | 1127 | 0.000 |
| `ext_dac_native` ⚠️ | external | 1 | 0.0000 | 0.0000 | 1127 | 0.000 |
| `ext_mask_llmlingua2` | external | 40 | 0.0000 | 0.0000 | 4077 | 0.800 |
| `ext_mask_llmlingua2_native` | external | 40 | 0.0000 | 0.0000 | 4026 | 0.800 |

注：⚠️ = 覆盖案例数不足一半，其 F1 建立在更容易的子集上，**不可直接比较**。

关键读法：`budget_cluster @0.50` 与 `ext_mask_llmlingua2` 的 F1 是 0，
但它们的 `gold_in_ctx` 分别是 1.000 / 0.800 —— 即**答案还在上下文里，模型却答不出**。
这说明多实体任务的失效机制不是"把答案丢了"，而是"把实体→属性的链接毁了"。
