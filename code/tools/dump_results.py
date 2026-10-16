import csv, json, io, os, re, sys

FIN = r"D:\CodexData\artifacts\privtrim-madar-20261013\final"
README = os.path.join(FIN, "README.md")
TABLES = os.path.join(FIN, "tables.csv")
GLOBAL = os.path.join(FIN, "comparisons_global.json")

print("=" * 100)
print("A. HEADLINE TABLE (from final/README.md)")
print("=" * 100)
rows = []
with io.open(README, encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 9 or cells[0] in ("模型", "---"):
            continue
        if set(cells[0]) <= set("-: "):
            continue
        rows.append(cells)

print("parsed rows:", len(rows))
bykey = {}
for model, data, method, rate, scored, fails, f1, frag, joint in rows:
    bykey.setdefault((model, data), []).append((method, rate, scored, fails, f1, frag, joint))

for (model, data), rs in bykey.items():
    print()
    print("--- %s / %s ---" % (model, data))
    print("%-28s %5s %8s %8s %8s %8s  %s" % ("method", "rate", "F1", "fragRem", "joint", "scored", "fails"))
    for method, rate, scored, fails, f1, frag, joint in sorted(rs, key=lambda r: (r[0], int(r[1]))):
        print("%-28s %5s %8s %8s %8s %8s  %s" % (method, rate, f1, frag, joint, scored, fails))

print()
print("=" * 100)
print("B. tables.csv schema + matched/native privacy metrics")
print("=" * 100)
with io.open(TABLES, encoding="utf-8-sig") as fh:
    rd = csv.DictReader(fh)
    cols = rd.fieldnames
    data_rows = list(rd)
print("columns:", cols)
print("rows:", len(data_rows))

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

key_cols = ["model", "dataset", "method", "rate", "track"]
extra = [c for c in ["scored", "executed_ok", "failures", "em", "f1", "pin_full",
                     "pin_fragment", "joint_fragment", "pout_fragment", "retention",
                     "budget_not_exact", "truncated", "privacy_unknown",
                     "privacy_lower_all", "privacy_upper_all"] if c in cols]

for (model, data) in sorted(set((r["model"], r["dataset"]) for r in data_rows)):
    for track in sorted(set(r["track"] for r in data_rows)):
        sel = [r for r in data_rows if r["model"] == model and r["dataset"] == data and r["track"] == track]
        if not sel:
            continue
        print()
        print("--- %s / %s / track=%s ---" % (model, data, track))
        print("%-28s %5s %7s %7s %8s %8s %8s %8s %8s" %
              ("method", "rate", "F1", "EM", "pinFull", "pinFrag", "jointF", "poutF", "retain"))
        for r in sorted(sel, key=lambda r: (r["method"], f(r["rate"]))):
            print("%-28s %5s %7.2f %7.2f %8.4f %8.4f %8.4f %8.4f %8.4f" % (
                r["method"], r["rate"], f(r["f1"]) * 100, f(r["em"]) * 100,
                f(r["pin_full"]), f(r["pin_fragment"]), f(r["joint_fragment"]),
                f(r["pout_fragment"]), f(r["retention"])))

print()
print("=" * 100)
print("C. comparisons_global.json")
print("=" * 100)
with io.open(GLOBAL, encoding="utf-8") as fh:
    g = json.load(fh)
print("top-level type:", type(g).__name__)
if isinstance(g, dict):
    print("top-level keys:", list(g.keys())[:40])
    for k, v in g.items():
        if isinstance(v, (int, float, str)):
            print("  %s = %s" % (k, v))
        elif isinstance(v, list):
            print("  %s: list len=%d" % (k, len(v)))
        elif isinstance(v, dict):
            print("  %s: dict keys=%s" % (k, list(v.keys())[:14]))
elif isinstance(g, list):
    print("list len:", len(g))
    print("first item keys:", list(g[0].keys()) if g and isinstance(g[0], dict) else g[:1])
