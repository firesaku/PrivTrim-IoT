# 论文（IEEE S&P / IEEEtran conference 格式）

- `main.tex`：正文（完整稿）。`pdflatex main.tex` 连跑两遍即可；参考文献用 `thebibliography`，无需 bibtex。
- `tabs.tex`：**由脚本生成**的表格（主结果 / 已发布方法 / 预算曲线 / 稳健性 / 逐来源），
  在 `main.tex` 里用 `\input{tabs}` 引入。**不要手改 `tabs.tex`**，改 `tables/make_tables.py`。
- `figs/make_figs.py`：图 2（F1–token 前沿）、图 3（机制：gic vs F1）、图 4（预算曲线 + 答案位置分层），
  只读 `results/` 与 `fixres/` 下的真实产物。图 1（方法机制图）为 TikZ，直接写在 `main.tex`。
- `main.pdf`：编译产物。

重建全部：

```bash
python paper/tables/make_tables.py --repo <repo> --fix <fixres> --base <review tmp> --out paper/tabs.tex
python paper/figs/make_figs.py --repo <repo> --fix <fixres> --ext <extbl_multi> --outdir paper/figs
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

数字纪律：正文数字与 `docs/17_证据汇总.md` 同源，由 `code/longctx/audit_docs17.py` 从原始产物
重算校验（184 项）。**改数顺序：产物 → docs/17 → tabs/figs → 论文**，每步都跑审计。
