# 10 · alias_budget_link 的新增与 GLM 复查记录

> 2026-10-17 ｜ 目的：补上"高效用"与"零泄漏"两个半边的交点
> 流程：数据驱动动机 → 写补丁 → 自查（发现两处残留）→ GLM 复查（发现一处更深的 bug）→ 修复 → 投 GPU

---

## 1. 为什么需要这一臂（数据驱动的动机）

r3 主结果的隐私列暴露出一个洞：

| 臂 @0.5 | F1 | 值泄漏 | 身份摧毁 |
|---|---:|---:|---:|
| `budget_cluster` | 0.4688 | 3.19% | 0 |
| **`budget_cluster_link`** | **0.7182** | **6.03%** | 0 |
| `alias_budget` | 0.4866 | 0.00% | 0 |
| `delete_budget` | 0.4688 | 0.00% | 40 |

**效用最高的两个臂恰好是唯一泄漏的两个。** 机制上讲得通：保护锚点 = 保住含敏感值的那
个单元 = 原值跟着留下。这正是第四象限的核心矛盾 —— 敏感且相关时，**只"保留"不行，
必须"换别名"**。

`alias_budget_link` = 别名脱敏 + 锚点保护 + 压到预算 B，是唯一能把两半拼起来的组合。

---

## 2. 自查找出的两处残留（都是我的补丁引入的）

写补丁时我按"精确文本匹配"删除旧函数体，因注释文本不同而**匹配失败**，留下化石代码。
两处都藏在 `protect_of` 里：

```python
# 残留 (a)
if case.get('anchor'):
    keys.append('----- entry %s -----' % case['anchor'] ...)
# -> keys 此时是隐式未定义，带 anchor 的案例直接 AttributeError

# 残留 (b)
keys = []
if case.get('target_entity'):
    keys.append(case['target_entity'])
# -> 把 keys 参数覆盖成 list，导致下一行
#    ks = protect_keys() if keys is None else keys
#    永远忽略传入的 ali_keys，别名保护**静默失效**
```

**残留 (b) 比 (a) 危险得多**：(a) 会崩溃，(b) 会让新臂退化成 `alias_budget` 的孪生，
而报告里完全看不出来。**崩溃是好事，静默才是灾难** —— 这与本项目一路的教训一致。

修法：按行号区间整体删除（不依赖单行锚点），并加断言确认函数体内不再出现
`keys = []` / `keys.append`。

---

## 3. GLM 复查发现的更深问题（不是本补丁引入，但被本补丁接入）

**`entity_diag` 的 alias 分支字母顺序错配。**

- 臂文本里的别名字母，由 `v4_router.apply_replacements` **逐 canary 调 token()** 按
  canary 列表顺序首见分配；
- 而 `entity_diag` 新建一个 Aliaser，按 **`entity_values` 顺序**分配。

真实数据里这两个顺序**不一致**：`build_longctx_cases.py` 的 `dedup_canaries` 把
canaries 按 `(type, value)` 排序，而 `entity_values` 跟随洗牌后的 `recs` 顺序。

**后果**：`tgt_entity_present` 对所有别名语义臂系统性失真 ——
既会假阴性（目标别名在文本里、诊断报 0），也会假阳性（目标单元被丢、诊断报 1）。
**而这一列正是"链接是否保住"的核心证据。**

GLM 给了可复现构造：`ents=[V2,V1,V3]`、target=V2 时，`alias_budget_link` 的文本含
`[ACCOUNT-B]`（目标真实别名），`entity_diag` 却在找 `[ACCOUNT-A]`，报 0；
而 `alias_budget` 明明丢了目标单元，却报 1。

**修法**：同一 Aliaser 先按 canary 列表顺序**预播**一遍映射，再按 `ents` 查。

### 修复的影响范围（关键）

| 臂 | 受影响 | 验证 |
|---|---|---|
| `budget_cluster` / `budget_cluster_link` | **否**（走 raw 分支，用原始 ACCT 串匹配） | 修复后 tent=0.6429，与 r3 的 0.643 一致 ✓ |
| `alias` / `alias_budget` / `alias_budget_link` | **是** | `alias_budget` tent 由 0.705 变 0.6607（之前被虚假抬高） |

**所以 r3 的主结论（cluster tent 0.643 vs link tent 1.000、增益 +0.249）不受影响。**
变的是别名语义臂的诊断列。

---

## 4. GLM 的回归检查（最重要的一项）

GLM 用 spy `order_fn` 捕获真实代码路径的 `(units, flags)` 输入，把改后文件与 `.bak`
并行加载，覆盖 4 种合成案例形态 × 3 个预算 × 8 条既有臂：

- **全部既有臂的 `(arm, ratio)` 文本与 `.bak` 字节级完全一致**
- `order_c / order_cl / order_d / order_a` 四组输入逐元素相同
- `budget_random` 的 `sha256(seed|key)` 随机序不变
- `alias_budget_link` 的 `order_al` 标志 == `token∈单元 ∨ gold∈单元`

这是我最担心的回归点（`protect_of` 被重构，`budget_cluster_link` 依赖它），已通过。

---

## 5. 我自己做的两项行为验证（在 GLM 之前）

**保护是否真的生效**（若失效，两臂文本会 100% 相同）：

| 案例类型 | n | 两臂文本不同 |
|---|---:|---:|
| 多实体 | 40 | **38（95%）** |
| needle | 72 | 24（33%） |

**方向是否正确**（link 臂应保留更多锚点）：

- 38 个多实体案例上 `alias_budget_link` 保留**更多** `[ACCOUNT-*]` token
- 2 个相等，**0 个更少**

---

## 6. 教训

1. **按精确文本匹配删旧代码是不安全的**，尤其带注释块时。按行号区间或结构锚点删，
   删完必须断言"不该再出现的模式确实不在了"。
2. **静默失效比崩溃危险得多**：残留 (b) 让新臂退化成孪生臂，输出看起来完全正常。
3. **GLM 复查的价值在"它去看我没看的地方"**。我自己验了保护生效与方向，但没去查
   诊断函数里的字母分配顺序 —— 而那正是核心证据列。
4. **修复前先划定影响范围**（哪些臂受影响），否则会把一个局部 bug 误判成全局失效。
   这次若不分清，很容易误以为主结论也崩了。
