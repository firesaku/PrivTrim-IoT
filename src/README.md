# `src/` — layout

| Directory | What lives there |
| --- | --- |
| `longctx/` | the method: harness (`e2e_quad.py`), predicate library (`privcheck.py`), quadrant router (`v4_router.py`), harness revisions r1–r5 |
| `analysis_quad4/` | reproducibility: `reproduce.py`, `validate_fallback.py`, `make_e2e_quad.py` |
| `external/` | vendored official SQuAD v2 scorer |
| `pc_stubs/` | import-only stubs for the Perception-Compressor checkout |
| `data_build/` | benchmark construction and structural probes over the source corpora |
| `paraphrase_probe/` | the motivation study behind the paraphrase-coverage claim |
| `mechanism/` | mechanism-level probes (V2 era) |
| `tools/` | export helpers over archived artifacts |
| `verify/` | independent audit drivers |
| `*.py` | the V1–V3 harnesses (`e2e*.py`, `v2c_*`, `v2d_*`, `v3_batch.py`), benchmark builders, analysis passes and export utilities |

Each script's purpose and its dependency status are tabulated in
[`docs/reference.md`](../docs/reference.md).

## Conventions

**No duplicated predicates.** `longctx/privcheck.py` defines the three value patterns and the leak
verdict. `privcheck.py` next to this file is a byte-identical copy kept so the V1–V3 scripts can load
their sibling module; nothing else carries a version of the rules. Earlier independent copies drifted
apart and produced a false "12.35% miss" figure that took a full audit to retract.

**Self-loading modules.** Scripts load their sibling modules by absolute path
(`importlib.util.spec_from_file_location` or `sys.path.insert` on `__file__`), so they run from any
working directory and need no install step. This also means two versions of a module can sit in
different directories without shadowing each other — which is why `v4_router.py` here and
`longctx/v4_router.py` can differ.

**Docstrings are the contract.** Where a docstring states an invariant, the tests or the harness
enforce it. `longctx/README_VERSIONS.md` records what happens when they do not.

**Some inline comments are still in Chinese.** The module docstrings and the printed report labels of
the load-bearing files were translated for this English release; older analysis scripts keep their
original inline comments. Nothing in the code path depends on them.
