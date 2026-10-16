import csv, io, json, os

FIN = r"D:\CodexData\artifacts\privtrim-madar-20261013\final"
OUT = r"D:\CodexData\tmp\privtrim-review-20261015\results_dump.txt"

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

L = []
def w(s=""):
    L.append(s)

# ---------- 1. tables.csv ----------
with io.open(os.path.join(FIN, "tables.csv"), encoding="utf-8-sig") as fh:
    rows = list(csv.DictReader(fh))

w("TABLE 1: tables.csv rows=%d" % len(rows))
w("models  = %s" % sorted(set(r["model"] for r in rows)))
w("datasets= %s" % sorted(set(r["dataset"] for r in rows)))
w("tracks  = %s" % sorted(set(r["track"] for r in rows)))
w("methods = %s" % sorted(set(r["method"] for r in rows)))
w("rates   = %s" % sorted(set(r["rate"] for r in rows)))
w("")

# ---------- 2. matched track, ranked by F1, one block per model/dataset ----------
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        sel = [r for r in rows if r["model"] == model and r["dataset"] == data and r["track"] == "matched"]
        if not sel:
            continue
        w("=== %s / %s / matched / n=%d ===" % (model, data, len(sel)))
        w("%-30s %5s %8s %8s %8s %8s %8s %8s" % ("method", "rate", "F1", "EM", "pinFull", "pinFrag", "jointF", "poutF"))
        for r in sorted(sel, key=lambda r: -f(r["f1"])):
            w("%-30s %5s %8.2f %8.2f %8.4f %8.4f %8.4f %8.4f" % (
                r["method"], r["rate"], f(r["f1"]) * 100, f(r["em"]) * 100,
                f(r["pin_full"]), f(r["pin_fragment"]), f(r["joint_fragment"]), f(r["pout_fragment"])))
        w("")

# ---------- 3. matched vs native for the same method/rate ----------
w("=== TABLE 3: matched vs native (F1) ===")
w("%-12s %-10s %-30s %5s %10s %10s %8s" % ("model", "dataset", "method", "rate", "F1_match", "F1_native", "delta"))
idx = {}
for r in rows:
    idx[(r["model"], r["dataset"], r["method"], r["rate"], r["track"])] = r
for k in sorted(idx):
    model, data, method, rate, track = k
    if track != "matched":
        continue
    nat = idx.get((model, data, method, rate, "native"))
    a = f(idx[k]["f1"]) * 100
    b = f(nat["f1"]) * 100 if nat else float("nan")
    w("%-12s %-10s %-30s %5s %10.2f %10.2f %8.2f" % (model, data, method, rate, a, b, a - b))
w("")

# ---------- 4. mask vs unmasked at rate 0.5 ----------
w("=== TABLE 4: mask_* vs same compressor at rate 50 (matched) ===")
w("%-12s %-10s %-26s %10s %10s %10s %10s" % ("model", "dataset", "pair", "F1_plain", "F1_mask", "jF_plain", "jF_mask"))
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        for base in ["cpc", "llmlingua2", "quito_x", "quito_x_paper_sum"]:
            p = idx.get((model, data, base, "0.5", "matched"))
            m = idx.get((model, data, "mask_" + base, "0.5", "matched"))
            if not p or not m:
                continue
            w("%-12s %-10s %-26s %10.2f %10.2f %10.4f %10.4f" % (
                model, data, base + " vs mask_" + base,
                f(p["f1"]) * 100, f(m["f1"]) * 100,
                f(p["joint_fragment"]), f(m["joint_fragment"])))
w("")

# ---------- 5. extremes ----------
w("=== TABLE 5: extremes across all 184 rows (matched+native) ===")
top_f1 = sorted(rows, key=lambda r: -f(r["f1"]))[:8]
w("-- best F1 --")
for r in top_f1:
    w("  %-12s %-10s %-28s rate=%-5s %-8s F1=%6.2f EM=%6.2f pinFrag=%.4f jF=%.4f" % (
        r["model"], r["dataset"], r["method"], r["rate"], r["track"],
        f(r["f1"]) * 100, f(r["em"]) * 100, f(r["pin_fragment"]), f(r["joint_fragment"])))
bot_f1 = sorted(rows, key=lambda r: f(r["f1"]))[:8]
w("-- worst F1 --")
for r in bot_f1:
    w("  %-12s %-10s %-28s rate=%-5s %-8s F1=%6.2f pinFrag=%.4f jF=%.4f" % (
        r["model"], r["dataset"], r["method"], r["rate"], r["track"],
        f(r["f1"]) * 100, f(r["pin_fragment"]), f(r["joint_fragment"])))
top_j = sorted(rows, key=lambda r: -f(r["joint_fragment"]))[:8]
w("-- best joint_fragment --")
for r in top_j:
    w("  %-12s %-10s %-28s rate=%-5s %-8s F1=%6.2f pinFrag=%.4f jF=%.4f poutF=%.4f" % (
        r["model"], r["dataset"], r["method"], r["rate"], r["track"],
        f(r["f1"]) * 100, f(r["pin_fragment"]), f(r["joint_fragment"]), f(r["pout_fragment"])))
w("-- worst pin_fragment (privacy proxy failure) --")
for r in sorted(rows, key=lambda r: f(r["pin_fragment"]))[:8]:
    w("  %-12s %-10s %-28s rate=%-5s %-8s pinFrag=%.4f F1=%6.2f" % (
        r["model"], r["dataset"], r["method"], r["rate"], r["track"],
        f(r["pin_fragment"]), f(r["f1"]) * 100))
w("-- pout_fragment != 1.0 (output residue present) --")
odd = [r for r in rows if abs(f(r["pout_fragment"]) - 1.0) > 1e-9]
w("   count=%d of %d" % (len(odd), len(rows)))
for r in odd[:12]:
    w("  %-12s %-10s %-28s rate=%-5s %-8s poutF=%.4f" % (
        r["model"], r["dataset"], r["method"], r["rate"], r["track"], f(r["pout_fragment"])))
w("-- full track rows (no compression) --")
for r in rows:
    if r["method"] == "full":
        w("  %-12s %-10s %-28s rate=%-5s %-8s F1=%6.2f EM=%6.2f pinFrag=%.4f jF=%.4f" % (
            r["model"], r["dataset"], r["method"], r["rate"], r["track"],
            f(r["f1"]) * 100, f(r["em"]) * 100, f(r["pin_fragment"]), f(r["joint_fragment"])))
w("")

# ---------- 6. comparisons_global.json ----------
with io.open(os.path.join(FIN, "comparisons_global.json"), encoding="utf-8") as fh:
    g = json.load(fh)
w("=== TABLE 6: comparisons_global.json ===")
w("type=%s" % type(g).__name__)
if isinstance(g, dict):
    for k, v in g.items():
        if isinstance(v, list):
            w("  %s: list len=%d" % (k, len(v)))
            if v and isinstance(v[0], dict):
                w("    item keys: %s" % list(v[0].keys()))
        elif isinstance(v, dict):
            w("  %s: dict keys=%s" % (k, list(v.keys())[:20]))
        else:
            w("  %s = %r" % (k, v))

def walk_sig(node, path=""):
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            found += walk_sig(v, path + "/" + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found += walk_sig(v, path + "[%d]" % i)
    return found

if isinstance(g, dict):
    items = None
    for k, v in g.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            items = v
            w("")
            w("first list field = %s, len=%d" % (k, len(v)))
            break
    if items:
        w("sample item: %s" % json.dumps(items[0], ensure_ascii=False)[:600])
        sigkey = None
        for cand in ["significant_holm", "significant", "signif", "is_significant"]:
            if cand in items[0]:
                sigkey = cand
                break
        w("significance key = %s" % sigkey)
        if sigkey:
            sig = [it for it in items if it.get(sigkey) is True]
            w("total comparisons=%d ; significant=%d" % (len(items), len(sig)))
            w("")
            w("%-14s %-14s %-16s %9s %28s %10s" % ("a", "b", "metric", "diff_pp", "ci95", "p_adj"))
            for it in sig:
                ci = it.get("ci95_pp") or it.get("ci95") or ""
                if isinstance(ci, list):
                    ci = "[%.2f, %.2f]" % (ci[0], ci[1])
                p = it.get("p_holm", it.get("p_adj", it.get("p_raw", "")))
                w("%-14s %-14s %-16s %9.2f %28s %10s" % (
                    it.get("a", "?"), it.get("b", "?"), it.get("metric", "?"),
                    float(it.get("difference_pp", 0)), str(ci)[:28], str(p)))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT, len(L), "lines")
