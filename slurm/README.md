# Slurm scripts

These are **templates**, not runnable files. Every path and the account name are placeholders:

| Placeholder | Replace with |
| --- | --- |
| `<ACCOUNT>` | your Slurm account, from `sacctmgr show assoc user="$USER"` |
| `/path/to/project` | the project root on the cluster, e.g. `$HOME/privtrim` |

The repository is public, so the real paths are not committed. Substitute them into a scratch copy
and submit that:

```bash
sed -e 's|<ACCOUNT>|myacct|g' \
    -e 's|/path/to/project|$HOME/privtrim|g' \
    slurm/frontend.slurm > /tmp/frontend.slurm
sbatch /tmp/frontend.slurm
```

## Why this note exists

Submitting a template directly fails with a message that does not mention the placeholder -- Slurm
reports `Invalid account or account/partition combination specified`, which reads like a permissions
problem rather than a missing substitution. Worse, if a working copy already sits at the destination,
copying the template over it destroys the working copy silently. Both happened once; this file is the
fix.

## Layout

| Script | Partition | What it does |
| --- | --- | --- |
| `preflight.slurm` | `prod` | checks the environment before anything expensive runs |
| `budget_dark.slurm` | `gpu` | DARK and its delete variant at r = 0.05/0.10/0.15 |
| `budget_ctrl.slurm` | `gpu` | the two redundancy controls at the same ratios, `--no-oracle` |
| `budget_ext.slurm` | `gpu` | published compressors at the tight ratios |
| `frontend.slurm` | `gpu` | DARK as a front end: alias the passage, then compress it |
| `recall.slurm` / `gen_recall.slurm` | `gpu` / `prod` | the detection-recall sweep and its case generation |
| `reid.slurm` | `gpu` | the extraction attack |
| `xmodel.slurm`, `granite.slurm`, `window.slurm`, `mprobe.slurm`, `eval_quad4.slurm`, `eval_lexcomp.slurm` | `gpu` | earlier families, kept for reproduction |

Every script is idempotent: each shard checks for its own `report.txt` and exits if it is already
there, so a timed-out array can be resubmitted and will only redo the missing shards.

`job ids are recorded in docs/reproduction.md` alongside the result family each one produced.
