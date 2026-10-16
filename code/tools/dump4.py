import csv, io, json, os
from collections import Counter

FIN = r"D:\CodexData\artifacts\privtrim-madar-20261013\final"
OUT = r"D:\CodexData\tmp\privtrim-review-20261015\conclusions.txt"

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

with io.open(os.path.join(FIN, "tables.csv"), encoding="utf-8-sig") as fh:
    rows = list(csv.DictReader(fh))
idx = {(r["model"], r["dataset"], r["method"], r["rate"], r["track"]): r for r in rows}
L = []
w = L.append

for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        for rate in ["0.25", "0.5"]:
            sel = [r for r in rows if r["model"] == model and r["dataset"] == data
                   and r["track"] == "matched" and r["rate"] == rate]
            # exclude 'full' from ranking (it is the uncompressed baseline)
            sel = [r for r in sel if r["method"] != "full"]
            if not sel:
                continue
            byf1 = sorted(sel, key=lambda r: -f(r["f1"]))[:4]
            byj = sorted(sel, key=lambda r: -f(r["joint_fragment"]))[:2]
            full = idx.get((model, data, "full", "1.0", "matched"))
            w("CONC %s %s @%s :: FULL_F1=%.2f || TOP_F1 :: %s || TOP_JOINT :: %s" % (
                model, data, rate,
                f(full["f1"]) * 100 if full else float("nan"),
                " > ".join("%s %.2f(j%.3f,p%.3f)" % (r["method"], f(r["f1"]) * 100,
                                                     f(r["joint_fragment"]), f(r["pin_fragment"])) for r in byf1),
                " > ".join("%s %.2f(j%.3f)" % (r["method"], f(r["f1"]) * 100,
                                               f(r["joint_fragment"])) for r in byj)))

# mask effect
dF1, dJ, pairs = [], [], []
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        for base in ["cpc", "llmlingua2", "quito_x", "quito_x_paper_sum"]:
            p = idx.get((model, data, base, "0.5", "matched"))
            m = idx.get((model, data, "mask_" + base, "0.5", "matched"))
            if p and m:
                dF1.append((f(m["f1"]) - f(p["f1"])) * 100)
                dJ.append(f(m["joint_fragment"]) - f(p["joint_fragment"]))
                pairs.append((model, data, base, (f(m["f1"]) - f(p["f1"])) * 100,
                              f(m["joint_fragment"]) - f(p["joint_fragment"]),
                              f(p["pin_fragment"]), f(m["pin_fragment"])))
w("MASK_EFFECT n_pairs=%d mean_dF1=%+.2fpp median_dF1=%+.2fpp mean_dJoint=%+.4f" % (
    len(dF1), sum(dF1) / len(dF1), sorted(dF1)[len(dF1) // 2], sum(dJ) / len(dJ)))
w("MASK_EFFECT per-pair :: " + " ; ".join(
    "%s/%s/%s dF1=%+.2f dJ=%+.3f pinFrag %.3f->%.3f" % p for p in pairs))

# pin_fragment extremes
best_pin = sorted(rows, key=lambda r: -f(r["pin_fragment"]))[:6]
worst_pin = sorted(rows, key=lambda r: f(r["pin_fragment"]))[:6]
w("PRIVACY_MAX_PINFRAG :: " + " ; ".join(
    "%s/%s/%s@%s/%s pinFrag=%.4f F1=%.2f j=%.3f" % (r["model"], r["dataset"], r["method"],
                                                    r["rate"], r["track"], f(r["pin_fragment"]),
                                                    f(r["f1"]) * 100, f(r["joint_fragment"]))
    for r in best_pin))
w("PRIVACY_MIN_PINFRAG :: " + " ; ".join(
    "%s/%s/%s@%s/%s pinFrag=%.4f F1=%.2f" % (r["model"], r["dataset"], r["method"],
                                             r["rate"], r["track"], f(r["pin_fragment"]),
                                             f(r["f1"]) * 100) for r in worst_pin))

# method-level means across all 184 rows
agg = {}
for r in rows:
    a = agg.setdefault(r["method"], {"f1": [], "j": [], "pf": [], "ret": [], "bne": 0, "trunc": 0, "n": 0})
    a["f1"].append(f(r["f1"]) * 100)
    a["j"].append(f(r["joint_fragment"]))
    a["pf"].append(f(r["pin_fragment"]))
    a["ret"].append(f(r["retention"]))
    a["bne"] += int(float(r["budget_not_exact"] or 0))
    a["trunc"] += int(float(r["truncated"] or 0))
    a["n"] += 1
w("METHOD_MEANS rows_per_method=%d :: " % agg["full"]["n"] + " ; ".join(
    "%s F1=%.1f j=%.3f pinFrag=%.3f ret=%.3f bne=%d trunc=%d" % (
        m, sum(a["f1"]) / len(a["f1"]), sum(a["j"]) / len(a["j"]),
        sum(a["pf"]) / len(a["pf"]), sum(a["ret"]) / len(a["ret"]), a["bne"], a["trunc"])
    for m, a in sorted(agg.items(), key=lambda kv: -sum(kv[1]["f1"]) / len(kv[1]["f1"]))))

# model comparison on shared conditions
shers = [k for k in idx if k[4] == "matched" and idx.get(("llama32_1b", k[1], k[2], k[3], "matched"))]
d = [(f(idx[k]["f1"]) - f(idx[("llama32_1b", k[1], k[2], k[3], "matched")]["f1"])) * 100 for k in shers if k[0] == "qwen25_7b"]
w("MODEL_COMPARISON shared_conditions=%d mean_F1_gap_qwen_minus_llama=%+.2fpp median=%+.2fpp" % (
    len(d), sum(d) / len(d), sorted(d)[len(d) // 2]))
w("MODEL_MEAN_F1 :: qwen=%.2f llama=%.2f" % (
    sum(f(r["f1"]) for r in rows if r["model"] == "qwen25_7b") / len([r for r in rows if r["model"] == "qwen25_7b"]) * 100,
    sum(f(r["f1"]) for r in rows if r["model"] == "llama32_1b") / len([r for r in rows if r["model"] == "llama32_1b"]) * 100))

# comparisons
with io.open(os.path.join(FIN, "comparisons_global.json"), encoding="utf-8") as fh:
    g = json.load(fh)
sig = [it for it in g if it.get("significant_holm")]
w("COMPARISONS total=%d significant=%d by_metric=%s by_model=%s" % (
    len(g), len(sig), dict(Counter(it["metric"] for it in sig)),
    dict(Counter(it["model"] for it in sig))))
w("COMPARISONS statuses=%s" % dict(Counter(it.get("status") for it in g)))
w("COMPARISONS p_holm_global_sig=%d" % sum(1 for it in g if it.get("significant_holm_global")))

# winner counts: who wins each significant f1 comparison
win = Counter()
for it in sig:
    if it["metric"] != "f1":
        continue
    a, b, dd = it["a"], it["b"], float(it["difference_pp"])
    win[a if dd > 0 else b] += 1
w("WIN_COUNT_f1 :: " + " ; ".join("%s=%d" % kv for kv in win.most_common()))
winj = Counter()
for it in sig:
    if it["metric"] != "joint_fragment":
        continue
    a, b, dd = it["a"], it["b"], float(it["difference_pp"])
    winj[a if dd > 0 else b] += 1
w("WIN_COUNT_joint :: " + " ; ".join("%s=%d" % kv for kv in winj.most_common()))

# biggest reversals: f1 winner vs joint winner
w("BIGGEST_f1_DIFFS :: " + " ; ".join(
    "%s>%s %s@%s %s +%.1fpp" % (it["a"], it["b"], it["dataset"], it["rate"], it["model"], float(it["difference_pp"]))
    for it in sorted([x for x in sig if x["metric"] == "f1"], key=lambda x: -abs(float(x["difference_pp"])))[:10]))
w("BIGGEST_joint_DIFFS :: " + " ; ".join(
    "%s>%s %s@%s %s %+.1fpp" % (it["a"], it["b"], it["dataset"], it["rate"], it["model"], float(it["difference_pp"]))
    for it in sorted([x for x in sig if x["metric"] == "joint_fragment"], key=lambda x: -abs(float(x["difference_pp"])))[:10]))

io.open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT, len(L), "lines")
