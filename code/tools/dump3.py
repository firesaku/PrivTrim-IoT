import csv, io, json, os

FIN = r"D:\CodexData\artifacts\privtrim-madar-20261013\final"
OUT = r"D:\CodexData\tmp\privtrim-review-20261015\compact.txt"

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

with io.open(os.path.join(FIN, "tables.csv"), encoding="utf-8-sig") as fh:
    rows = list(csv.DictReader(fh))

idx = {}
for r in rows:
    idx[(r["model"], r["dataset"], r["method"], r["rate"], r["track"])] = r

L = []
def w(s):
    L.append(s)

n = [0]
def tag():
    n[0] += 1
    return "#%02d" % n[0]

# 1) rankings at each rate, matched track
for rate in ["0.25", "0.5", "0.75"]:
    for model in ["qwen25_7b", "llama32_1b"]:
        for data in ["squad", "hotpotqa"]:
            sel = [r for r in rows if r["model"] == model and r["dataset"] == data
                   and r["track"] == "matched" and r["rate"] == rate]
            if not sel:
                continue
            parts = []
            for r in sorted(sel, key=lambda r: -f(r["f1"])):
                parts.append("%s F1=%.2f joint=%.3f pinFrag=%.3f" % (
                    r["method"], f(r["f1"]) * 100, f(r["joint_fragment"]), f(r["pin_fragment"])))
            w("%s RANK rate=%s | %s | %s :: %s" % (tag(), rate, model, data, "  ||  ".join(parts)))
    w("")

# 2) full baseline
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        r = idx.get((model, data, "full", "1.0", "matched"))
        if r:
            w("%s BASELINE-FULL | %s | %s | F1=%.2f EM=%.2f pinFull=%.4f pinFrag=%.4f joint=%.4f" % (
                tag(), model, data, f(r["f1"]) * 100, f(r["em"]) * 100,
                f(r["pin_full"]), f(r["pin_fragment"]), f(r["joint_fragment"])))
w("")

# 3) mask vs plain at 0.5
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        parts = []
        for base in ["cpc", "llmlingua2", "quito_x", "quito_x_paper_sum"]:
            p = idx.get((model, data, base, "0.5", "matched"))
            m = idx.get((model, data, "mask_" + base, "0.5", "matched"))
            if p and m:
                parts.append("%s: plainF1=%.2f/j=%.3f vs maskF1=%.2f/j=%.3f" % (
                    base, f(p["f1"]) * 100, f(p["joint_fragment"]),
                    f(m["f1"]) * 100, f(m["joint_fragment"])))
        if parts:
            w("%s MASK-VS-PLAIN rate=0.5 | %s | %s :: %s" % (tag(), model, data, "  ||  ".join(parts)))
w("")

# 4) matched vs native at 0.25 and 0.5
for rate in ["0.25", "0.5"]:
    for model in ["qwen25_7b", "llama32_1b"]:
        parts = []
        for data in ["squad", "hotpotqa"]:
            for method in ["cpc", "dac", "llmlingua2", "provence", "quito_x", "quito_x_paper_sum"]:
                a = idx.get((model, data, method, rate, "matched"))
                b = idx.get((model, data, method, rate, "native"))
                if a and b:
                    parts.append("%s/%s M=%.1f N=%.1f" % (data, method, f(a["f1"]) * 100, f(b["f1"]) * 100))
        if parts:
            w("%s NATIVE-VS-MATCHED rate=%s | %s :: %s" % (tag(), rate, model, "  ".join(parts)))
w("")

# 5) retention / truncated / budget_not_exact / privacy_unknown
agg = {}
for r in rows:
    k = (r["model"], r["method"])
    a = agg.setdefault(k, {"n": 0, "ret": 0.0, "trunc": 0, "bne": 0, "pu": 0, "fail": 0, "scored": 0})
    a["n"] += 1
    a["ret"] += f(r["retention"])
    a["trunc"] += int(float(r["truncated"] or 0))
    a["bne"] += int(float(r["budget_not_exact"] or 0))
    a["pu"] += int(float(r["privacy_unknown"] or 0))
    a["fail"] += int(float(r["failures"] or 0))
    a["scored"] += int(float(r["scored"] or 0))
for model in ["qwen25_7b", "llama32_1b"]:
    parts = []
    for (m, method), a in sorted(agg.items()):
        if m != model:
            continue
        parts.append("%s meanRetain=%.3f truncated=%d budgetNotExact=%d privacyUnknown=%d failures=%d" % (
            method, a["ret"] / a["n"], a["trunc"], a["bne"], a["pu"], a["fail"]))
    w("%s COST-AND-QUALITY-FLAGS | %s :: %s" % (tag(), model, "  ||  ".join(parts)))
w("")

# 6) comparisons_global.json
with io.open(os.path.join(FIN, "comparisons_global.json"), encoding="utf-8") as fh:
    g = json.load(fh)

def find_list(obj):
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            got = find_list(v)
            if got:
                return got
    return None

items = find_list(g)
w("%s COMPARISONS top-level-type=%s" % (tag(), type(g).__name__))
if isinstance(g, dict):
    w("%s COMPARISONS top-keys :: %s" % (tag(), " | ".join(
        "%s(%s)" % (k, ("list%d" % len(v)) if isinstance(v, list) else ("dict%d" % len(v)) if isinstance(v, dict) else repr(v))
        for k, v in g.items())))
if items:
    w("%s COMPARISONS n=%d item-keys :: %s" % (tag(), len(items), ",".join(items[0].keys())))
    w("%s COMPARISONS sample :: %s" % (tag(), json.dumps(items[0], ensure_ascii=False)))
    sigkey = None
    for c in ["significant_holm", "significant", "signif"]:
        if c in items[0]:
            sigkey = c
            break
    if sigkey:
        sig = [it for it in items if it.get(sigkey) is True]
        w("%s COMPARISONS sigkey=%s significant=%d of %d" % (tag(), sigkey, len(sig), len(items)))
        from collections import Counter
        w("%s COMPARISONS by-metric :: %s" % (tag(), dict(Counter(it.get("metric") for it in sig))))
        w("%s COMPARISONS by-model-pair :: %s" % (tag(), dict(Counter(
            (it.get("model") or it.get("model_a") or "?") for it in sig))))
        top = sorted(sig, key=lambda it: -abs(float(it.get("difference_pp") or 0)))[:20]
        for it in top:
            ci = it.get("ci95_pp") or it.get("ci95")
            if isinstance(ci, list):
                ci = "[%.2f,%.2f]" % (ci[0], ci[1])
            w("%s SIG %s vs %s | %s | diff=%.2fpp | CI=%s | p_holm=%s | model=%s | dataset=%s | rate=%s" % (
                tag(), it.get("a"), it.get("b"), it.get("metric"),
                float(it.get("difference_pp") or 0), ci,
                it.get("p_holm", it.get("p_adj")), it.get("model"), it.get("dataset"), it.get("rate")))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT, len(L), "lines")
