# 论文（IEEE S&P / IEEEtran conference 格式）

- `main.tex`：正文。`pdflatex main.tex` 跑两遍即可（BibTeX 用 `thebibliography`，无需 bibtex）。
- `main.pdf`：编译产物（5 页）。
- `figs/make_figs.py`：图 2/3/4 的生成脚本，**只读 `results/` 与 `fixres/` 里的真实产物**，不手填数字。
- 图 1（方法机制图）用 TikZ 直接写在 `main.tex` 里。

数字来源：正文所有数字都写在 `docs/17_证据汇总.md`，并由 `code/longctx/audit_docs17.py`
从原始产物重算校验（184 项，全部一致）。改数时**先改产物、再改文档、再改论文**，三步都要跑审计。

重建：

```bash
python paper/figs/make_figs.py --repo <repo> --fix <fixres> --ext <extbl_multi> --outdir paper/figs
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```
