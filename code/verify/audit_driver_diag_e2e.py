# Audit driver: execute diag_e2e.py verbatim, with only the hardcoded HPC JSON path
# swapped to the local mirror copy, to verify real runtime behavior.
import io, json, sys

HPC = r"/dpc/kuex0095/100065732/privtrim_v2c_20261015/e2eout/e2e_alm_cells.json"
LOCAL = r"D:\CodexData\tmp\privtrim-review-20261015\repo_staging\extracted\e2eout\e2e_alm_cells.json"

src = io.open(r"D:\CodexData\tmp\privtrim-review-20261015\hpc\diag_e2e.py", encoding="utf-8").read()
assert HPC in src, "hardcoded path not found in diag_e2e.py"
src = src.replace(HPC, LOCAL)
# Emulate the HPC login node's UTF-8 stdout (Windows console here is cp1252).
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.stderr.write("[driver] executing diag_e2e.py verbatim against local JSON mirror\n")
exec(compile(src, "diag_e2e.py", "exec"))

# Cross-check: recompute EM/F1 straight from the JSON to validate diag arithmetic.
d = json.loads(io.open(LOCAL, encoding="utf-8").read())
sys.stderr.write("\n[driver] cross-check of diag arithmetic vs raw JSON:\n")
for k, v in sorted(d["agg"].items()):
    em = v[1] / max(v[0], 1)
    f1 = v[2] / max(v[0], 1)
    sys.stderr.write("  %-24s n=%-4d em=%.4f f1=%.4f pins(full,frag)=(%d,%d)\n" % (k, v[0], em, f1, v[3], v[4]))
sys.stderr.write("[driver] arms=%s cases=%s sample_len=%d sample_none=%d\n"
                 % (d["arms"], d["cases"], len(d["answers_sample"]),
                    sum(1 for a in d["answers_sample"] if a is None)))
