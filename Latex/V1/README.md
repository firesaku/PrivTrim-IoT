# DARK — LaTeX V1

**Keeping the Referent in the DARK: Anchor-Preserving Token Compression for Privacy-Aware Inference on IoT Logs**

Yukun Wen, Shijie Zeng — Department of Computer Science, Khalifa University

---

## 编译

需要 TeX Live 或 MiKTeX（含 `natbib`、`algorithm2e`/`algorithmicx`、`booktabs`、`graphicx`）。

```bash
cd Latex/V1
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

产物为 `main.pdf`，当前 18 页、7 图、7 表。

首次编译若报缺包，按 MiKTeX 提示安装，或 `tlmgr install <pkg>`。

---

## 目录结构

```
Latex/V1/
├── main.tex              # 入口：导言区 + \input 各章节
├── refs.bib              # 参考文献 51 条（2023--2026，49 条 CCF-A / JCR-Q1）
├── main.pdf              # 编译产物
├── sections/             # 12 个章节文件
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_problem.tex
│   ├── 04_method.tex
│   ├── 05_benchmark.tex
│   ├── 06_setup.tex
│   ├── 07_results.tex
│   ├── 08_analysis.tex
│   ├── 09_discussion.tex
│   ├── 10_conclusion.tex
│   └── 11_appendix.tex
├── tables/               # 表格与案例图
│   ├── tab_main.tex
│   ├── tab_external.tex
│   ├── tab_budget.tex
│   ├── tab_ablation_alias.tex
│   └── fig6_cases.tex
└── figures/              # 6 张矢量图（PDF）
    ├── fig0_teaser.pdf
    ├── fig_method.pdf
    ├── fig_persource.pdf
    ├── fig1_main.pdf
    ├── fig2_diagnostic.pdf
    └── fig3_alias.pdf
```

---

## 图表一览

| 编号 | 文件 | 位置 | 内容 |
|---|---|---|---|
| Fig. 1 | `fig0_teaser.pdf` | §1 | 三条判据 × 四条路线：传统管线 / 本文管线 / 策略落点 / 各自代价 |
| Fig. 2 | `fig_method.pdf` | §4.1 | DARK 流水线 + 键控别名 + line-vs-block |
| Fig. 3 | `fig_persource.pdf` | §7.3 | 六个来源 × 三个指标（F1 / 泄漏 / 存活） |
| Fig. 4 | `fig1_main.pdf` | §7.4 | 预算曲线 + 泄漏 + 结果构成 + 跨模型/改述 |
| Fig. 5 | `fig2_diagnostic.pdf` | §8.1 | 答案保留率 + 解离 + 分来源 |
| Fig. 6 | `fig3_alias.pdf` | §8.4 | 别名 vs 删除 |
| Fig. 7 | `fig6_cases.tex` | §8.4 | 两个真实案例（LaTeX 直排） |

| 表 | 文件 | 内容 |
|---|---|---|
| Table 1 | `tab_main.tex` | 112 例主比较 |
| Table 2 | `tab_external.tex` | 2,728 例外部分法对比 |
| Table 3 | `tab_budget.tex` | 预算曲线 |
| Table 4 | `tab_ablation_alias.tex` | 别名 vs 删除，七个设置 |
| Table 5–7 | 内联于 `11_appendix.tex` | 分来源、锚点类型 |

---

## 方法要点

DARK 用四个操作处理 context unit：

| 字母 | 操作 | 象限 |
|---|---|---|
| **D** | delete | sensitive · irrelevant |
| **A** | alias | sensitive · relevant ← 唯一保留指代方式的格子 |
| **R** | refill | plain · irrelevant |
| **K** | keep | plain · relevant |

关键在于 **A 格不删除而是改名**：查询实体本身就是受保护值，删掉它就丢失指代，压缩掉又常丢掉答案记录；给同一个实体分配稳定别名、并对问题施加同一映射，既满足零泄漏又保住可回答性。

---

## 数据说明

所有数值来自 `results/percase/` 下的逐例评测记录（正向填充、无金标、keyed alias）：

- 主集 112 例，扩展集 2,728 例，LongBench 衍生三变体各 300 例
- 对照 12 个已发表压缩器/脱敏器，同批同预算重跑
- 两套基准复用 case key，跨集合按 key 连接会混记录，分析时以「不压缩臂复现参考值」作为归属校验

---

## 修改指引

- 改数字：只改 `sections/*.tex` 与 `tables/*.tex` 中的字面值；图表数据由外部脚本生成后导出到 `figures/`
- 改配色/字号：`figures/*.pdf` 由 Python (matplotlib) 生成，图内文字统一 ≥10 pt、字体 Times New Roman，设计宽度等于 `\textwidth`（7.1 in）；配色只用红黑与灰阶（红黑 = 本文方法，红 = 泄漏/失败）
- 增删章节：在 `main.tex` 的 `\input` 序列中调整
