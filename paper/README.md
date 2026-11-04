# `paper/` — manuscript

**"Keeping the Referent in the DARK: Anchor-Preserving Token Compression for Privacy-Aware Inference
on IoT Logs"** — Yukun Wen, Shijie Zeng (Khalifa University).

| Path | Content |
| --- | --- |
| `main.pdf` | the compiled manuscript (18 pages) — read this |
| `main.tex` | entry point: preamble, title block, `\input`s, bibliography |
| `sections/` | one file per section, `00_abstract` … `11_appendix` |
| `tables/` | `tab_main`, `tab_external`, `tab_budget`, `tab_ablation_alias`, `fig6_cases` |
| `figures/` | the vector figures (`fig0_teaser`, `fig_method`, `fig1_main`, `fig2_diagnostic`, `fig3_alias`, `fig_persource`) |
| `refs.bib` | 51 references |

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The tree is complete and self-contained: it builds cleanly from a fresh clone with 0 undefined
references, 0 undefined citations and 0 overfull hboxes, and produces 18 pages.

> `main.pdf` in this repository is the artifact as originally shipped, not the output of the command
> above; a local rebuild differs from it by a few hundred bytes (PDF metadata), not in content. If you
> edit the sources, rebuild rather than trusting the committed PDF.

## Style notes

- Two-column Letter, text block pinned to 7.10 in so figures designed at that width place 1:1.
- Numbers in tables and body text use ↑ / ↓ to state the direction of "better" for each metric.
- Arms are typeset with `\arm{...}`, metric names with `\msr{...}`, and the method as small caps
  `\textsc{DARK}`.
- Every paired difference is reported with a 95% percentile bootstrap interval over cases
  (10,000 resamples); a difference is called significant only when the interval excludes zero.
- No significance claim is ever attached to a comparison between the 112-case and 2,728-case sets,
  because they draw different cases.

## Where each number lives

| Table | Section | Benchmarks |
| --- | --- | --- |
| `tab_main` | Results §2 | main set, 112 cases |
| `tab_external` | Results §1 | extension, 2,728 cases, twelve baselines |
| `tab_budget` | Results §4 | main set, five ratios |
| `tab_ablation_alias` | Analysis | alias-vs-delete on five settings |
| `fig6_cases` | Analysis | two failure modes at the level of one case |

`docs/results.md` restates these tables with the same values; if the two ever disagree, the `.tex`
files are authoritative and `docs/results.md` is the bug.
