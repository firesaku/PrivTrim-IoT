# Long-context results (job 10601756, Almesbar V100, Qwen2.5-7B-Instruct fp16)

112 cases x 18 (arm,ratio) = 2016 generations. ctx_tokens is the mean **achieved** context tokens; leak = sum(out_leak_value)/sum(n_canaries).

| source | arm | budget | n | EM | F1 | ctx_tokens | leak | coll | ans_in_ctx |
|---|---|---|---|---|---|---|---|---|---|
| pec_home | full | ref | 20 | 0.8500 | 0.8750 | 2110 | 0.000000 | 0.000 | 1.000 |
| pec_home | delete | ref | 20 | 0.5500 | 0.5750 | 1973 | 0.000000 | 0.400 | 1.000 |
| pec_home | alias | ref | 20 | 0.5500 | 0.5750 | 1943 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster | 0.25 | 20 | 0.5000 | 0.5000 | 512 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster | 0.50 | 20 | 0.5500 | 0.5750 | 1004 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster | 0.75 | 20 | 0.6000 | 0.6250 | 1529 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster_link | 0.25 | 20 | 0.9500 | 0.9750 | 514 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster_link | 0.50 | 20 | 0.9000 | 0.9250 | 1008 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_cluster_link | 0.75 | 20 | 0.9500 | 0.9750 | 1533 | 0.000000 | 0.000 | 1.000 |
| pec_home | budget_random | 0.25 | 20 | 0.1000 | 0.1000 | 494 | 0.000000 | 0.000 | 0.250 |
| pec_home | budget_random | 0.50 | 20 | 0.2500 | 0.2750 | 1020 | 0.000000 | 0.000 | 0.550 |
| pec_home | budget_random | 0.75 | 20 | 0.6000 | 0.6250 | 1551 | 0.000000 | 0.000 | 0.800 |
| pec_home | delete_budget | 0.25 | 20 | 0.5000 | 0.5000 | 512 | 0.000000 | 0.400 | 1.000 |
| pec_home | delete_budget | 0.50 | 20 | 0.5500 | 0.5750 | 1004 | 0.000000 | 0.400 | 1.000 |
| pec_home | delete_budget | 0.75 | 20 | 0.5500 | 0.5750 | 1526 | 0.000000 | 0.400 | 1.000 |
| pec_home | alias_budget | 0.25 | 20 | 0.5000 | 0.5000 | 512 | 0.000000 | 0.000 | 1.000 |
| pec_home | alias_budget | 0.50 | 20 | 0.5500 | 0.5750 | 1004 | 0.000000 | 0.000 | 1.000 |
| pec_home | alias_budget | 0.75 | 20 | 0.5500 | 0.5750 | 1535 | 0.000000 | 0.000 | 1.000 |

| privacylens | full | ref | 20 | 0.5500 | 0.6067 | 4881 | 0.000000 | 0.000 | 1.000 |
| privacylens | delete | ref | 20 | 0.3500 | 0.3900 | 4736 | 0.000000 | 0.400 | 0.950 |
| privacylens | alias | ref | 20 | 0.4000 | 0.4400 | 4733 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster | 0.25 | 20 | 0.4500 | 0.4500 | 1194 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster | 0.50 | 20 | 0.4500 | 0.4500 | 2381 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster | 0.75 | 20 | 0.4500 | 0.4500 | 3610 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster_link | 0.25 | 20 | 0.5500 | 0.5900 | 1192 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster_link | 0.50 | 20 | 0.6500 | 0.6900 | 2374 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_cluster_link | 0.75 | 20 | 0.6000 | 0.6000 | 3607 | 0.000000 | 0.000 | 1.000 |
| privacylens | budget_random | 0.25 | 20 | 0.1500 | 0.1500 | 1145 | 0.000000 | 0.000 | 0.300 |
| privacylens | budget_random | 0.50 | 20 | 0.1500 | 0.1500 | 2372 | 0.000000 | 0.000 | 0.400 |
| privacylens | budget_random | 0.75 | 20 | 0.5500 | 0.5500 | 3609 | 0.000000 | 0.000 | 0.850 |
| privacylens | delete_budget | 0.25 | 20 | 0.4500 | 0.4500 | 1185 | 0.000000 | 0.400 | 0.950 |
| privacylens | delete_budget | 0.50 | 20 | 0.4500 | 0.4500 | 2368 | 0.000000 | 0.400 | 0.950 |
| privacylens | delete_budget | 0.75 | 20 | 0.4000 | 0.4000 | 3601 | 0.000000 | 0.400 | 0.950 |
| privacylens | alias_budget | 0.25 | 20 | 0.4500 | 0.4500 | 1194 | 0.000000 | 0.000 | 1.000 |
| privacylens | alias_budget | 0.50 | 20 | 0.4500 | 0.4500 | 2381 | 0.000000 | 0.000 | 1.000 |
| privacylens | alias_budget | 0.75 | 20 | 0.4500 | 0.4500 | 3610 | 0.000000 | 0.000 | 1.000 |

| simuhome | full | ref | 20 | 0.4500 | 0.4500 | 7166 | 0.000000 | 0.000 | 1.000 |
| simuhome | delete | ref | 20 | 0.1500 | 0.1500 | 7075 | 0.000000 | 0.400 | 1.000 |
| simuhome | alias | ref | 20 | 0.1500 | 0.1500 | 7054 | 0.000000 | 0.000 | 1.000 |
| simuhome | budget_cluster | 0.25 | 20 | 0.1000 | 0.1000 | 856 | 0.000000 | 0.000 | 0.500 |
| simuhome | budget_cluster | 0.50 | 20 | 0.4500 | 0.4500 | 3139 | 0.000000 | 0.000 | 1.000 |
| simuhome | budget_cluster | 0.75 | 20 | 0.5000 | 0.5000 | 4649 | 0.000000 | 0.000 | 1.000 |
| simuhome | budget_cluster_link | 0.25 | 20 | 0.3000 | 0.3000 | 860 | 0.000000 | 0.000 | 0.500 |
| simuhome | budget_cluster_link | 0.50 | 20 | 0.7000 | 0.7000 | 3146 | 0.000000 | 0.000 | 1.000 |
| simuhome | budget_cluster_link | 0.75 | 20 | 0.7500 | 0.7500 | 4656 | 0.000000 | 0.000 | 1.000 |
| simuhome | budget_random | 0.25 | 20 | 0.0000 | 0.0000 | 646 | 0.000000 | 0.000 | 0.000 |
| simuhome | budget_random | 0.50 | 20 | 0.2000 | 0.2000 | 2858 | 0.000000 | 0.000 | 0.250 |
| simuhome | budget_random | 0.75 | 20 | 0.4000 | 0.4000 | 4804 | 0.000000 | 0.000 | 0.500 |
| simuhome | delete_budget | 0.25 | 20 | 0.1000 | 0.1000 | 856 | 0.000000 | 0.400 | 0.500 |
| simuhome | delete_budget | 0.50 | 20 | 0.4500 | 0.4500 | 3109 | 0.000000 | 0.400 | 1.000 |
| simuhome | delete_budget | 0.75 | 20 | 0.5000 | 0.5000 | 4591 | 0.000000 | 0.400 | 1.000 |
| simuhome | alias_budget | 0.25 | 20 | 0.1000 | 0.1000 | 856 | 0.000000 | 0.000 | 0.500 |
| simuhome | alias_budget | 0.50 | 20 | 0.5500 | 0.5500 | 3150 | 0.000000 | 0.000 | 1.000 |
| simuhome | alias_budget | 0.75 | 20 | 0.6000 | 0.6000 | 4660 | 0.000000 | 0.000 | 1.000 |

| smartbench_tdcs | full | ref | 20 | 0.0000 | 0.0000 | 16344 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | delete | ref | 20 | 0.0000 | 0.0000 | 16222 | 0.000000 | 0.400 | 1.000 |
| smartbench_tdcs | alias | ref | 20 | 0.0000 | 0.0000 | 16196 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster | 0.25 | 20 | 0.6500 | 0.7500 | 3099 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster | 0.50 | 20 | 0.6000 | 0.6000 | 7239 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster | 0.75 | 20 | 0.0500 | 0.0500 | 10920 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster_link | 0.25 | 20 | 0.7500 | 0.8500 | 3106 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster_link | 0.50 | 20 | 0.7000 | 0.7000 | 7246 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_cluster_link | 0.75 | 20 | 0.0500 | 0.0500 | 10926 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | budget_random | 0.25 | 20 | 0.3500 | 0.3500 | 3118 | 0.000000 | 0.000 | 0.500 |
| smartbench_tdcs | budget_random | 0.50 | 20 | 0.3000 | 0.3000 | 7217 | 0.000000 | 0.000 | 0.650 |
| smartbench_tdcs | budget_random | 0.75 | 20 | 0.0000 | 0.0000 | 11140 | 0.000000 | 0.000 | 0.750 |
| smartbench_tdcs | delete_budget | 0.25 | 20 | 0.6500 | 0.7500 | 3099 | 0.000000 | 0.400 | 1.000 |
| smartbench_tdcs | delete_budget | 0.50 | 20 | 0.6000 | 0.6000 | 7235 | 0.000000 | 0.400 | 1.000 |
| smartbench_tdcs | delete_budget | 0.75 | 20 | 0.1000 | 0.1000 | 10912 | 0.000000 | 0.400 | 1.000 |
| smartbench_tdcs | alias_budget | 0.25 | 20 | 0.6500 | 0.7500 | 3099 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | alias_budget | 0.50 | 20 | 0.6000 | 0.6000 | 7248 | 0.000000 | 0.000 | 1.000 |
| smartbench_tdcs | alias_budget | 0.75 | 20 | 0.0000 | 0.0000 | 10904 | 0.000000 | 0.000 | 1.000 |

| toolmem | full | ref | 12 | 0.4167 | 0.4901 | 7031 | 0.000000 | 0.000 | 1.000 |
| toolmem | delete | ref | 12 | 0.5000 | 0.5632 | 6920 | 0.000000 | 0.000 | 0.917 |
| toolmem | alias | ref | 12 | 0.4167 | 0.4354 | 6898 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster | 0.25 | 12 | 0.5833 | 0.5833 | 1716 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster | 0.50 | 12 | 0.4167 | 0.4167 | 3440 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster | 0.75 | 12 | 0.5000 | 0.5000 | 5242 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster_link | 0.25 | 12 | 0.5000 | 0.5000 | 1721 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster_link | 0.50 | 12 | 0.4167 | 0.4286 | 3439 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_cluster_link | 0.75 | 12 | 0.4167 | 0.4196 | 5245 | 0.000000 | 0.000 | 1.000 |
| toolmem | budget_random | 0.25 | 12 | 0.0000 | 0.0000 | 1732 | 0.000000 | 0.000 | 0.000 |
| toolmem | budget_random | 0.50 | 12 | 0.0833 | 0.0833 | 3494 | 0.000000 | 0.000 | 0.417 |
| toolmem | budget_random | 0.75 | 12 | 0.1667 | 0.1699 | 5213 | 0.000000 | 0.000 | 0.667 |
| toolmem | delete_budget | 0.25 | 12 | 0.5000 | 0.5000 | 1718 | 0.000000 | 0.000 | 0.917 |
| toolmem | delete_budget | 0.50 | 12 | 0.4167 | 0.4167 | 3442 | 0.000000 | 0.000 | 0.917 |
| toolmem | delete_budget | 0.75 | 12 | 0.5000 | 0.5000 | 5244 | 0.000000 | 0.000 | 0.917 |
| toolmem | alias_budget | 0.25 | 12 | 0.5833 | 0.5833 | 1716 | 0.000000 | 0.000 | 1.000 |
| toolmem | alias_budget | 0.50 | 12 | 0.4167 | 0.4167 | 3441 | 0.000000 | 0.000 | 1.000 |
| toolmem | alias_budget | 0.75 | 12 | 0.5000 | 0.5000 | 5242 | 0.000000 | 0.000 | 1.000 |

| toolmem_raw | full | ref | 20 | 0.0500 | 0.0543 | 9879 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | delete | ref | 20 | 0.0000 | 0.0000 | 9775 | 0.000000 | 0.400 | 1.000 |
| toolmem_raw | alias | ref | 20 | 0.0000 | 0.0000 | 9750 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster | 0.25 | 20 | 0.1500 | 0.1500 | 2439 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster | 0.50 | 20 | 0.3000 | 0.3000 | 4922 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster | 0.75 | 20 | 0.2000 | 0.2000 | 7381 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster_link | 0.25 | 20 | 0.6500 | 0.6500 | 2443 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster_link | 0.50 | 20 | 0.7500 | 0.7500 | 4920 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_cluster_link | 0.75 | 20 | 0.6000 | 0.6056 | 7384 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | budget_random | 0.25 | 20 | 0.1000 | 0.1000 | 2449 | 0.000000 | 0.000 | 0.200 |
| toolmem_raw | budget_random | 0.50 | 20 | 0.2500 | 0.2500 | 4927 | 0.000000 | 0.000 | 0.450 |
| toolmem_raw | budget_random | 0.75 | 20 | 0.4000 | 0.4000 | 7384 | 0.000000 | 0.000 | 0.750 |
| toolmem_raw | delete_budget | 0.25 | 20 | 0.1500 | 0.1500 | 2438 | 0.000000 | 0.400 | 1.000 |
| toolmem_raw | delete_budget | 0.50 | 20 | 0.3000 | 0.3000 | 4915 | 0.000000 | 0.400 | 1.000 |
| toolmem_raw | delete_budget | 0.75 | 20 | 0.1000 | 0.1042 | 7385 | 0.000000 | 0.400 | 1.000 |
| toolmem_raw | alias_budget | 0.25 | 20 | 0.1500 | 0.1500 | 2438 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | alias_budget | 0.50 | 20 | 0.3000 | 0.3000 | 4920 | 0.000000 | 0.000 | 1.000 |
| toolmem_raw | alias_budget | 0.75 | 20 | 0.1000 | 0.1034 | 7385 | 0.000000 | 0.000 | 1.000 |

## By case_kind, all arms

| case_kind | arm | budget | n | EM | F1 | ans_in_ctx | tgt_entity_present |
|---|---|---|---|---|---|---|---|
| multi_entity | full | ref | 40 | 0.4000 | 0.4083 | 1.000 | 1.000 |
| multi_entity | delete | ref | 40 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| multi_entity | alias | ref | 40 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| multi_entity | budget_cluster | 0.25 | 40 | 0.1250 | 0.1250 | 0.875 | 0.000 |
| multi_entity | budget_cluster | 0.50 | 40 | 0.0000 | 0.0000 | 1.000 | 0.000 |
| multi_entity | budget_cluster | 0.75 | 40 | 0.0750 | 0.0750 | 1.000 | 0.100 |
| multi_entity | budget_cluster_link | 0.25 | 40 | 0.6250 | 0.6250 | 0.875 | 0.875 |
| multi_entity | budget_cluster_link | 0.50 | 40 | 0.5500 | 0.5500 | 1.000 | 1.000 |
| multi_entity | budget_cluster_link | 0.75 | 40 | 0.5750 | 0.5750 | 1.000 | 1.000 |
| multi_entity | budget_random | 0.25 | 40 | 0.0500 | 0.0500 | 0.300 | 0.175 |
| multi_entity | budget_random | 0.50 | 40 | 0.1250 | 0.1250 | 0.600 | 0.500 |
| multi_entity | budget_random | 0.75 | 40 | 0.3250 | 0.3250 | 0.775 | 0.775 |
| multi_entity | delete_budget | 0.25 | 40 | 0.1250 | 0.1250 | 0.875 | 0.250 |
| multi_entity | delete_budget | 0.50 | 40 | 0.0000 | 0.0000 | 1.000 | 0.325 |
| multi_entity | delete_budget | 0.75 | 40 | 0.0000 | 0.0000 | 1.000 | 0.350 |
| multi_entity | alias_budget | 0.25 | 40 | 0.1250 | 0.1250 | 0.875 | 0.000 |
| multi_entity | alias_budget | 0.50 | 40 | 0.0250 | 0.0250 | 1.000 | 0.175 |
| multi_entity | alias_budget | 0.75 | 40 | 0.0000 | 0.0000 | 1.000 | 0.600 |
| needle | full | ref | 72 | 0.3750 | 0.4065 | 1.000 | 1.000 |
| needle | delete | ref | 72 | 0.3750 | 0.4036 | 0.972 | 1.000 |
| needle | alias | ref | 72 | 0.3750 | 0.3962 | 1.000 | 1.000 |
| needle | budget_cluster | 0.25 | 72 | 0.5417 | 0.5694 | 0.931 | 1.000 |
| needle | budget_cluster | 0.50 | 72 | 0.7222 | 0.7292 | 1.000 | 1.000 |
| needle | budget_cluster | 0.75 | 72 | 0.5417 | 0.5486 | 1.000 | 1.000 |
| needle | budget_cluster_link | 0.25 | 72 | 0.6250 | 0.6708 | 0.931 | 1.000 |
| needle | budget_cluster_link | 0.50 | 72 | 0.7917 | 0.8117 | 1.000 | 1.000 |
| needle | budget_cluster_link | 0.75 | 72 | 0.5694 | 0.5784 | 1.000 | 1.000 |
| needle | budget_random | 0.25 | 72 | 0.1667 | 0.1667 | 0.181 | 1.000 |
| needle | budget_random | 0.50 | 72 | 0.2639 | 0.2708 | 0.375 | 1.000 |
| needle | budget_random | 0.75 | 72 | 0.3889 | 0.3964 | 0.694 | 1.000 |
| needle | delete_budget | 0.25 | 72 | 0.5278 | 0.5556 | 0.903 | 1.000 |
| needle | delete_budget | 0.50 | 72 | 0.7222 | 0.7292 | 0.972 | 1.000 |
| needle | delete_budget | 0.75 | 72 | 0.5417 | 0.5498 | 0.972 | 1.000 |
| needle | alias_budget | 0.25 | 72 | 0.5417 | 0.5694 | 0.931 | 1.000 |
| needle | alias_budget | 0.50 | 72 | 0.7361 | 0.7431 | 1.000 | 1.000 |
| needle | alias_budget | 0.75 | 72 | 0.5556 | 0.5635 | 1.000 | 1.000 |

## Below guess_baseline

rows_below_guess_baseline = 491 of 1680 budget-arm rows (29.2%)
  of those: pred starts with UNKNOWN = 389 (79.2%), pred empty = 0 (0.0%)
  UNKNOWN predictions across ALL budget rows = 652 of 1680 (38.8%)
  by arm|ratio: [(('alias_budget', 0.25), 35), (('alias_budget', 0.5), 39), (('alias_budget', 0.75), 40), (('budget_cluster', 0.25), 35), (('budget_cluster', 0.5), 40), (('budget_cluster', 0.75), 37), (('budget_cluster_link', 0.25), 15), (('budget_cluster_link', 0.5), 18), (('budget_cluster_link', 0.75), 17), (('budget_random', 0.25), 38), (('budget_random', 0.5), 35), (('budget_random', 0.75), 27), (('delete_budget', 0.25), 35), (('delete_budget', 0.5), 40), (('delete_budget', 0.75), 40)]
  by case_kind: [('multi_entity', 491)]

## Row-level spot checks

rows where the gold string is NOT in the emitted context = 227 of 2016
  by arm|ratio: [(('alias_budget', 0.25), 10), (('budget_cluster', 0.25), 10), (('budget_cluster_link', 0.25), 10), (('budget_random', 0.25), 87), (('budget_random', 0.5), 61), (('budget_random', 0.75), 31), (('delete', None), 2), (('delete_budget', 0.25), 12), (('delete_budget', 0.5), 2), (('delete_budget', 0.75), 2)]

multi_entity rows where cluster_link F1 > cluster F1: 62 of 120

- case=longctx-pec_home-multi-0000 ratio=0.25 kind=multi_entity
  - q='In the log above, which persona id is recorded for the account ACCT-5ee052b431dc?'
  - gold='multi_0212_00' target_entity='ACCT-5ee052b431dc' target_attribute='multi_0212_00'
  - cluster:      tent=0 ans_in_ctx=1 ctx=558 pred='UNKNOWN' f1=0.000
  - cluster_link: tent=1 ans_in_ctx=1 ctx=562 pred='multi_0212_00' f1=1.000

- case=longctx-pec_home-multi-0001 ratio=0.25 kind=multi_entity
  - q='In the log above, which persona id is recorded for the account ACCT-5be08d44f1fd?'
  - gold='multi_0104_02' target_entity='ACCT-5be08d44f1fd' target_attribute='multi_0104_02'
  - cluster:      tent=0 ans_in_ctx=1 ctx=491 pred='UNKNOWN' f1=0.000
  - cluster_link: tent=1 ans_in_ctx=1 ctx=494 pred='multi_0104_02' f1=1.000

- case=longctx-pec_home-multi-0002 ratio=0.25 kind=multi_entity
  - q='In the log above, which persona id is recorded for the account ACCT-7358fe654211?'
  - gold='dynam_0053_01' target_entity='ACCT-7358fe654211' target_attribute='dynam_0053_01'
  - cluster:      tent=0 ans_in_ctx=1 ctx=486 pred='UNKNOWN' f1=0.000
  - cluster_link: tent=1 ans_in_ctx=1 ctx=495 pred='dynam_0053_01' f1=1.000

- case=longctx-pec_home-multi-0003 ratio=0.25 kind=multi_entity
  - q='In the log above, which persona id is recorded for the account ACCT-573c8510555b?'
  - gold='multi_0095_01' target_entity='ACCT-573c8510555b' target_attribute='multi_0095_01'
  - cluster:      tent=0 ans_in_ctx=1 ctx=498 pred='UNKNOWN' f1=0.000
  - cluster_link: tent=1 ans_in_ctx=1 ctx=474 pred='multi_0095_01' f1=1.000

- case=longctx-pec_home-multi-0004 ratio=0.25 kind=multi_entity
  - q='In the log above, which persona id is recorded for the account ACCT-c9a6ba2852a7?'
  - gold='dynam_0063_01' target_entity='ACCT-c9a6ba2852a7' target_attribute='dynam_0063_01'
  - cluster:      tent=0 ans_in_ctx=1 ctx=507 pred='UNKNOWN' f1=0.000
  - cluster_link: tent=1 ans_in_ctx=1 ctx=514 pred='dynam_0063_01' f1=1.000

### Cases whose gold sits late in the context, cluster vs link @0.50
- longctx-pec_home-multi-0000 gold_at=83% of context | cluster tent=0 pred='UNKNOWN' f1=0.000 || link tent=1 pred='multi_0212_00' f1=1.000
- longctx-pec_home-multi-0006 gold_at=80% of context | cluster tent=0 pred='UNKNOWN' f1=0.000 || link tent=1 pred='dynam_0326_00' f1=1.000
- longctx-smartbench_tdcs-multi-0007 gold_at=80% of context | cluster tent=0 pred='UNKNOWN' f1=0.000 || link tent=1 pred='family_on_vacation' f1=1.000
- longctx-smartbench_tdcs-multi-0006 gold_at=80% of context | cluster tent=0 pred='UNKNOWN' f1=0.000 || link tent=1 pred='UNKNOWN' f1=0.000
