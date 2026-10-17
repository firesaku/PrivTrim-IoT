# 优势分层：领先出现在哪一类案例上

预算 @0.5；分层内 F1 = 该层 f1 总和 / 案例数；案例 = 40 个多实体案例中元信息齐全的 40 个。

## A. 按答案位置分层

| 分层 | n | `alias_budget` | `alias_budget_link` | `budget_cluster` | `budget_cluster_link` | `budget_random` | `delete_budget` | `ext_cpc` | `ext_dac` | `ext_llmlingua2` | `ext_mask_llmlingua2` | `ext_mask_quito_x` | `ext_provence` | `ext_quito_x` | 该层第一 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 答案在头部（<1/3） | 15 | 0.0667 | 0.5333 | 0.0000 | 0.4667 | 0.1333 | 0.0000 | 0.4000 | -- | 0.3333 | 0.0000 | 0.0000 | 0.4667 | 0.4615 | **alias_budget_link** |
| 答案在中部（1/3–2/3） | 19 | 0.0526 | 0.5789 | 0.0000 | 0.5263 | 0.2105 | 0.0000 | 0.4211 | -- | 0.3509 | 0.0000 | 0.0000 | 0.4211 | 0.2308 | **alias_budget_link** |
| 答案在尾部（≥2/3） | 6 | 0.3333 | 0.8333 | 0.0000 | 0.8333 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.6667 | 0.6667 | **budget_cluster_link** |

## B. 按实体数分层

| 分层 | n | `alias_budget` | `alias_budget_link` | `budget_cluster` | `budget_cluster_link` | `budget_random` | `delete_budget` | `ext_cpc` | `ext_dac` | `ext_llmlingua2` | `ext_mask_llmlingua2` | `ext_mask_quito_x` | `ext_provence` | `ext_quito_x` | 该层第一 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 实体数 = 3（n=8） | 8 | 0.1250 | 0.5000 | 0.0000 | 0.6250 | 0.2500 | 0.0000 | 0.2500 | -- | 0.1250 | 0.0000 | 0.0000 | 0.1250 | 0.1250 | **budget_cluster_link** |
| 实体数 = 5（n=31） | 31 | 0.0645 | 0.6129 | 0.0000 | 0.5161 | 0.1613 | 0.0000 | 0.3871 | -- | 0.4086 | 0.0000 | 0.0000 | 0.5806 | 0.4500 | **alias_budget_link** |
| 实体数 ≥ 6（n=1） | 1 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | **ext_quito_x** |

## C. 分层第一名的计数

- 位置分层：`alias_budget_link` 第一 2 层
- 位置分层：`budget_cluster_link` 第一 1 层
- 实体数分层：`budget_cluster_link` 第一 1 层
- 实体数分层：`alias_budget_link` 第一 1 层
- 实体数分层：`ext_quito_x` 第一 1 层
