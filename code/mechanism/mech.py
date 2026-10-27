import csv, io, os

FIN = r"D:\CodexData\artifacts\privtrim-<CLUSTER>-20261013\final"
D = r"D:\CodexData\tmp\privtrim-review-20261015\mech"
os.makedirs(D, exist_ok=True)

def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

with io.open(os.path.join(FIN, "tables.csv"), encoding="utf-8-sig") as fh:
    rows = list(csv.DictReader(fh))
idx = {(r["model"], r["dataset"], r["method"], r["rate"], r["track"]): r for r in rows}

def g(model, data, method, rate, track="matched"):
    return idx.get((model, data, method, rate, track))

out = []
n = 0
def emit(tag, text):
    global n
    n += 1
    fn = os.path.join(D, "%02d_%s.txt" % (n, tag))
    io.open(fn, "w", encoding="utf-8").write(text + "\n")
    out.append(fn)

# A. mask 前端：同模型、同数据集、同预算、同压缩器的配对
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        for base in ["cpc", "quito_x", "llmlingua2", "quito_x_paper_sum"]:
            p = g(model, data, base, "0.5")
            m = g(model, data, "mask_" + base, "0.5")
            if not (p and m):
                continue
            emit("MASK_%s_%s_%s" % (model, data, base),
                 "MASK | %s | %s | %s -> mask_%s @0.5 matched | dF1=%+.2fpp (%.2f->%.2f) | dJoint=%+.3f | dEM=%+.2fpp | pinFrag %.3f->%.3f | pinFull %.3f->%.3f | poutF %.3f->%.3f | scored=%s" % (
                     model, data, base, base,
                     (f(m["f1"]) - f(p["f1"])) * 100, f(p["f1"]) * 100, f(m["f1"]) * 100,
                     f(m["joint_fragment"]) - f(p["joint_fragment"]),
                     (f(m["em"]) - f(p["em"])) * 100,
                     f(p["pin_fragment"]), f(m["pin_fragment"]),
                     f(p["pin_full"]), f(m["pin_full"]),
                     f(p["pout_fragment"]), f(m["pout_fragment"]), p["scored"]))

# B. QUITO 聚合方式：max(作者源码) vs sum(论文公式)
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        a = g(model, data, "quito_x", "0.5")
        b = g(model, data, "quito_x_paper_sum", "0.5")
        c = g(model, data, "mask_quito_x", "0.5")
        d = g(model, data, "mask_quito_x_paper_sum", "0.5")
        if not (a and b):
            continue
        emit("AGG_%s_%s" % (model, data),
             "AGG | %s | %s @0.5 matched | max: F1=%.2f joint=%.3f pinFrag=%.3f || sum: F1=%.2f joint=%.3f pinFrag=%.3f || dF1(max-sum)=%+.2fpp || masked max F1=%.2f / masked sum F1=%.2f" % (
                 model, data, f(a["f1"]) * 100, f(a["joint_fragment"]), f(a["pin_fragment"]),
                 f(b["f1"]) * 100, f(b["joint_fragment"]), f(b["pin_fragment"]),
                 (f(a["f1"]) - f(b["f1"])) * 100,
                 f(c["f1"]) * 100 if c else float("nan"),
                 f(d["f1"]) * 100 if d else float("nan")))

# C. 训练型 vs 训练无关，同模型同条件
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        tr = g(model, data, "cpc", "0.5")
        fr = g(model, data, "quito_x", "0.5")
        dc = g(model, data, "dac", "0.5")
        ll = g(model, data, "llmlingua2", "0.5")
        if not (tr and fr):
            continue
        emit("TRAINED_%s_%s" % (model, data),
             "TRAINED-vs-FREE | %s | %s @0.5 matched | cpc(trained) F1=%.2f pinFrag=%.3f | quito_x(free) F1=%.2f pinFrag=%.3f | dac(free) F1=%.2f pinFrag=%.3f | llmlingua2(trained) F1=%.2f pinFrag=%.3f | full F1=%.2f" % (
                 model, data, f(tr["f1"]) * 100, f(tr["pin_fragment"]),
                 f(fr["f1"]) * 100, f(fr["pin_fragment"]),
                 f(dc["f1"]) * 100 if dc else float("nan"), f(dc["pin_fragment"]) if dc else float("nan"),
                 f(ll["f1"]) * 100 if ll else float("nan"), f(ll["pin_fragment"]) if ll else float("nan"),
                 f(g(model, data, "full", "1.0")["f1"]) * 100))

# D. native vs matched 轨道
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        for method in ["cpc", "quito_x", "llmlingua2"]:
            a = g(model, data, method, "0.5", "matched")
            b = g(model, data, method, "0.5", "native")
            if not (a and b):
                continue
            emit("TRACK_%s_%s_%s" % (model, data, method),
                 "TRACK | %s | %s | %s @0.5 | matched F1=%.2f retain=%.3f budgetNotExact=%s || native F1=%.2f retain=%.3f budgetNotExact=%s || dF1(native-matched)=%+.2fpp" % (
                     model, data, method,
                     f(a["f1"]) * 100, f(a["retention"]), a["budget_not_exact"],
                     f(b["f1"]) * 100, f(b["retention"]), b["budget_not_exact"],
                     (f(b["f1"]) - f(a["f1"])) * 100))

# E. 同模型方法总排名 @0.5 matched
for model in ["qwen25_7b", "llama32_1b"]:
    for data in ["squad", "hotpotqa"]:
        sel = [r for r in rows if r["model"] == model and r["dataset"] == data
               and r["track"] == "matched" and r["rate"] == "0.5"]
        if not sel:
            continue
        emit("RANK_%s_%s" % (model, data),
             "RANK | %s | %s @0.5 matched | %s" % (
                 model, data,
                 " > ".join("%s F1=%.2f(p%.3f)" % (r["method"], f(r["f1"]) * 100, f(r["pin_fragment"]))
                            for r in sorted(sel, key=lambda r: -f(r["f1"])))))

print("wrote %d files to %s" % (n, D))
for p in out:
    print(" ", os.path.basename(p))
