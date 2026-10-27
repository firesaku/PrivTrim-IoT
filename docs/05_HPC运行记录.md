# HPC 运行记录

> 本文档记录本次工作使用过的集群、作业、资源与踩过的坑。

---

## 1. 两个集群的对比（2026-10-15/16 实测）

| | **<CLUSTER>** | **Almesbar** |
|---|---|---|
| 登录主机 | `login.<CLUSTER>.<ADMIN_DOMAIN>` | `login.almesbar.<ADMIN_DOMAIN>` |
| 登录名 | `100065732_ku` | `100065732@kunet.ae` |
| 认证方式 | **SSH 私钥**（`100065732_ku-ssh-key.pem` + 口令文件） | **密码**（来自 `KU_ZSJ` 指向的文件） |
| Slurm account | `<SLURM_ACCOUNT>` | `<SLURM_ACCOUNT>`（**相同**） |
| 共享目录 | `/path/to` | `/path/to`（**同一个盘**） |
| `gpu` 分区 | 1–9 节点，`gpu:nvidia_h200:8` | 3 节点，`gpu:v100:2` |
| GPU 排队 | **严重**：本项目 fairshare 低，预计等 **约 6 小时** | **无排队**：提交即运行 |
| `prod` 分区 | 充裕（有整节点 idle，22,464 空闲 CPU） | 充裕（多个 idle 分区） |
| `interactive` 分区 | 存在但 **`AllowAccounts=rcadmin`** → 本项目**不可用** | 未见此限制 |

**结论**：**Almesbar 是本项目当前唯一能立即用上 GPU 的集群**，且因为共享 `/dpc`，模型/数据/脚本**零搬运**。

---

## 2. 作业清单

| 作业号 | 集群 | 分区 | 资源 | 名称 | 时长 | 结果 |
|---|---|---|---|---|---|---|
| 1443690 | <CLUSTER> | prod | 8 CPU / 16G | pt-v2c | **3 s** | ✅ COMPLETED |
| 1444010 | <CLUSTER> | gpu | **1×H200** / 8 CPU / 64G | pt-v2d | **65 s** | ✅ COMPLETED |
| 1444036 | <CLUSTER> | gpu | **数组 4×1 H200**（`%2` 限流） | pt-v3 | ~65 s/任务 | ✅ COMPLETED（3/4；gran 失败） |
| 1444037 | <CLUSTER> | prod | 4 CPU / 16G | pt-v1 | **42 s** | ✅ COMPLETED（含 bug） |
| 1444040 | <CLUSTER> | prod | 4 CPU / 16G | pt-v1b | ~40 s | ✅ COMPLETED（诊断） |
| 1444041 | <CLUSTER> | prod | 4 CPU / 16G | pt-v1fix | **42 s** | ✅ COMPLETED（修正） |
| 1444045 | <CLUSTER> | gpu | **1×H200** | pt-v3gran | **33 s** | ✅ COMPLETED |
| 1444034/1443516/1449162/… | <CLUSTER> | prod/gpu | — | （本项目其他作业，非本次） | — | — |
| 1443650 / 1451115 / 1451097 / 1451098 | <CLUSTER> | gpu | 1×H200 | tc-* / sift-* | — | 排队/pending（非本次） |
| **1443690, 1444010, 1444036, 1444037, 1444040, 1444041, 1444045** | | | | **本次 <CLUSTER> 全部** | | **全部 COMPLETED** |
| 1451077 | <CLUSTER> | gpu | 1×H200 | pt-e2e | 6:18 | ❌ **FAILED**（评分函数名错误） |
| 1451107 / 1451134 | <CLUSTER> | gpu | 1×H200 | pt-e2e2 | 未启动 | ⛔ **已取消**（预计排队至 23:08） |
| 1451131 | <CLUSTER> | interactive | — | pt-e2ei | — | ⛔ **被拒**（`AllowAccounts=rcadmin`） |
| 10598862 | **Almesbar** | gpu | **1×V100** / 8 CPU / 64G | pt-e2ea | 25:40 | ⚠️ COMPLETED 但**数字作废**（金标准 bug） |
| **10599046** | **Almesbar** | gpu | **1×V100** | pt-e2eb | **25:40** | ✅ **有效结果** |
| **10599443** | **Almesbar** | gpu | **1×V100** | pt-e2e2b | **31:03** | ✅ **有效结果** |
| **10600642** | **Almesbar** | gpu | **1×V100** | pt-e2e3 | **18:40** | ✅ **有效结果** |

**合计**：**12 个作业**（<CLUSTER> 10 + Almesbar 4，含取消/失败），全部单卡或小 CPU，**从未占用整节点**。

---

## 3. 工具链（本次自建）

| 工具 | 用途 | 说明 |
|---|---|---|
| `code/almesbar_run.py` | Almesbar 远程执行 | **密码只读进进程内存**，不写入脚本、不拼进命令行、不打印；来源为环境变量 `KU_ZSJ` 指向的文件 |
| `rput.py`（复用旧会话） | SFTP 上传 + sha256 校验 | 位于 `D:\CodexData\tmp\privtrim-<CLUSTER>-20261013\rput.py` |
| `rget.py`（本次自建） | SFTP 下载 | 用于回传报告与逐例结果 |
| `<CLUSTER>_run.py`（技能自带） | <CLUSTER> 远程执行 | 私钥认证 |
| 数据打包 | `tar` 远端打包 + `rget` 单文件下载 | 比逐文件 SFTP 快得多（1.84 MB 一次拉回） |

---

## 4. 踩过的坑（按重要性）

### 4.1 远程路径与执行
| 坑 | 现象 | 解决 |
|---|---|---|
| 多行命令被拆参数 | `--command` 传多行字符串时 PowerShell 拆成多个参数 | **改用 `--script`**（管道给 `bash -s`） |
| PowerShell 吃掉内层双引号 | `sinfo -o "%P..."` 引号丢失 | 同上：写成 `.sh` 文件 |
| 写了 BOM 的 .sh | `RD=...` 变量赋值失败（`\ufeff` 前缀） | 用 `write` 工具写文件，不用 `Out-File -Encoding utf8` |
| Windows 控制台编码 | `UnicodeEncodeError: 'charmap' codec` | 设 `PYTHONIOENCODING=utf-8` + `[Console]::OutputEncoding` |
| SFTP 目标目录不存在 | `FileNotFoundError: [Errno 2] No such file` | 先 `mkdir -p` 再上传 |

### 4.2 集群与调度
| 坑 | 现象 | 解决 |
|---|---|---|
| <CLUSTER> GPU 排队 6 小时 | `squeue --start` 显示 23:08 | 换 **Almesbar** |
| `interactive` 被拒 | `reason=(PartitionConfig)` | 查明 `AllowAccounts=rcadmin`；**放弃该分区** |
| V100 无 BF16 | — | 加 `--dtype fp16` 开关 |

### 4.3 代码与数据
| 坑 | 现象 | 影响 |
|---|---|---|
| 评分函数名错 | `module 'squad_scorer' has no attribute 'metric_max_over_ground_truths'` | 作业 FAILED；改用 `compute_exact` / `compute_f1` |
| **金标准提取错** | EM 全 0、F1 假高 | **5000 次生成作废重跑** |
| 样本建库时归一化了空白 | 按行切分得到 0 个单元 → `records=0` | 改用 200 字符窗，粒度对比才跑通 |
| 数字段检查用错类型 | `redact_miss` 假报 12.35% | 独立诊断后修正为 1.14% |

---

## 5. 资源效率

| 项 | 数值 |
|---|---|
| 本次 <CLUSTER> 作业总时长 | ~4 分钟（CPU）+ ~3.5 分钟（GPU 单卡/数组） |
| 本次 Almesbar 作业总时长 | 25:40 + 25:40 + 31:03 + 18:40 ≈ **1 小时 41 分** |
| 单次端到端实验成本 | **1×V100 × ~25–31 分钟** |
| 数据回传 | 1.84 MB 打包一次拉回 |
| 峰值占用 | 全部 `--gres=gpu:*:1`，**未占整节点** |

---

## 6. 可复现命令

```bash
# Almesbar（本次主力）
python code/almesbar_run.py --script <本地脚本.sh>

# <CLUSTER>
python "$DSH_HOME/skills/<CLUSTER>-hpc/scripts/<CLUSTER>_run.py" --script <本地脚本.sh>

# 上传（SFTP + sha256 校验）
python rput.py <本地文件> /path/to/100065732/privtrim_v2c_20261015/<远端名>

# 下载
python rget.py /path/to/.../<远端文件> <本地路径>
```

**远端工作目录**：`/path/to/100065732/privtrim_v2c_20261015/`
**结果打包**：`/path/to/100065732/privtrim_results_20261016.tgz`（1,841,245 B）
