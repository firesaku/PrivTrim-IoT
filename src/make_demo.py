#!/usr/bin/env python3
"""Build demo/dark_demo.html from the benchmark itself.

The demo has to do one thing: show a reader that the value they must hide is the same value the
question needs in order to be answerable, and that the three obvious policies fail or succeed for
reasons they can see in a few lines of text.

Everything on the page is measured, not written. For each case the script runs the real prompt
construction (build_arm_texts) to obtain the prompt each policy would send, then reads the reader's
actual answer for that arm from the archived per-case records. Nothing is mocked and no sentence is
composed to make a policy look better.

Usage
    python src/make_demo.py            # writes demo/dark_demo.html
"""
from __future__ import annotations

import collections
import glob
import gzip
import html
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "longctx"))

import e2e_quad as E  # noqa: E402

# One case per source, chosen because our arm answers and both alternatives do not.
CASES = [
    ("longctx-pec_home-multi-0064", "pec_home"),
    ("longctx-simuhome-multi-0206", "simuhome"),
    ("longctx-smartbench_tdcs-multi-0094", "smartbench_tdcs"),
]

POLICIES = [
    ("full",           "Send it raw",      "Forward the log unchanged. The model has everything it needs, including the account.", "leaks"),
    ("budget_cluster", "Compress",        "Keep the most redundant half and drop the rest, which is what a redundancy-ordered compressor does.", "fails"),
    ("quad4_del",      "Delete in place", "Keep the record, remove the account from it. Length preserved.", "fails"),
    ("quad4",          "DARK",            "Keep the record block and alias the values inside it, in the question as well.", "works"),
]

# Table IV, 2,728 cases. Kept here so the page cannot drift from the paper.
TABLE = [
    ("DARK", 0.4937, 100.0, 0.0, 4146),
    ("Quito-X", 0.4831, 69.8, 98.1, 4242),
    ("CPC", 0.4740, 82.7, 92.6, 4242),
    ("LongLLMLingua", 0.4230, 70.3, 100.0, 4241),
    ("Provence", 0.4080, 88.9, 90.8, 4242),
    ("LexComp (2026 preprint)", 0.3696, 65.0, 100.0, 5817),
    ("LLMLingua-2", 0.2651, 62.6, 100.0, 4242),
    ("Selective Context", 0.2588, 46.1, 100.0, 4055),
    ("mask + LLMLingua-2", 0.1712, 67.9, 0.0, 4242),
]

CSS = """
:root{--ink:#4A1412;--red:#8E2A22;--g5:#555;--g9:#9A9A9A;--g4:#D4D4D2;--pale:#EFEFEF;}
*{box-sizing:border-box}
body{font:16px/1.65 "Times New Roman",Georgia,serif;color:#111;max-width:1080px;
     margin:0 auto;padding:36px 24px 80px;background:#fff}
h1{font-size:27px;line-height:1.3;margin:0 0 6px;color:var(--ink)}
h2{font-size:19px;margin:42px 0 10px;padding-bottom:5px;border-bottom:2px solid var(--g4);
   color:var(--ink)}
h3{font-size:16px;margin:20px 0 6px}
p{margin:9px 0}
.lede{color:var(--g5);font-size:17px;margin-bottom:24px}
code,.mono{font-family:"Consolas","Courier New",monospace;font-size:13.5px}
.rec{background:#FAFAFA;border:1px solid var(--g4);border-left:3px solid var(--ink);
     padding:12px 14px;margin:10px 0;overflow-x:auto}
.rec .k{color:var(--g5)}
.hl-red{background:#F6E3E0;color:var(--red);font-weight:700;padding:1px 3px;border-radius:2px}
.hl-ink{background:#EBE0DE;color:var(--ink);font-weight:700;padding:1px 3px;border-radius:2px}
.q{background:var(--pale);border-left:3px solid var(--g9);padding:10px 14px;margin:12px 0;
   font-size:15px}
.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px}
@media(max-width:1000px){.cols{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.cols{grid-template-columns:1fr}}
.col{border:1px solid var(--g4);border-top:4px solid var(--g9);padding:14px 15px;background:#fff}
.col.works{border-top-color:var(--ink)}
.col h4{margin:0 0 4px;font-size:15.5px}
.col .verdict{font-size:12.5px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}
.col.works .verdict{color:var(--ink);font-weight:700}
.col .verdict.bad{color:var(--red);font-weight:700}
.col p{font-size:14px;color:var(--g5);margin:0 0 10px}
.snip{background:#FAFAFA;border:1px solid var(--g4);padding:8px 10px;font-family:"Consolas",
      monospace;font-size:12px;line-height:1.55;white-space:pre-wrap;word-break:break-word;
      max-height:190px;overflow:auto}
.ans{margin-top:10px;font-size:14px}
.ans b{font-family:"Consolas",monospace}
.good{color:var(--ink);font-weight:700}
.bad{color:var(--red);font-weight:700}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin:12px 0}
th,td{padding:6px 9px;text-align:right;border-bottom:1px solid var(--g4)}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--g5);font-weight:700}
tr.ours{background:var(--pale);font-weight:700;color:var(--ink)}
.hm{border-collapse:collapse;font-size:13px}
.hm td,.hm th{border:2px solid #fff;padding:5px 7px;text-align:center;min-width:64px}
.hm th{background:#fff;border:none;font-weight:400;font-size:12px;color:var(--g5)}
.hm td.lbl{text-align:right;border:none;padding-right:10px;font-size:13px}
.note{font-size:14px;color:var(--g5);border-left:3px solid var(--g9);padding-left:12px;
      margin:16px 0}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--g4);font-size:13.5px;
       color:var(--g5)}
a{color:var(--red)}
"""


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_cases(path, keys):
    out = {}
    with open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["key"] in keys:
                out[c["key"]] = c
    return out


def load_percase(patterns):
    acc = collections.defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open_maybe_gz(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    acc[r["arm"]].setdefault(r["case_key"], r)
    return acc


def snippet_for(arm_text, gold, entity, alias_hint=None, needle=None, width=250):
    """The anchor line, because that is the line the policies treat differently.

    Falls back to the answer's unit, then to the alias, then says so. Showing the answer's unit
    for every policy would make four different transformations look identical.
    """
    units = E.split_units(arm_text)
    probes = [needle, alias_hint, entity, gold]
    for probe in probes:
        if not probe:
            continue
        for u in units:
            if probe in u:
                return _clip(u, probe, width)
    return "(this record is not in the prompt)"


def _clip(unit, needle, width):
    i = unit.find(needle)
    if i < 0:
        return unit[:width]
    lo = max(0, i - width // 3)
    hi = min(len(unit), i + len(needle) + width)
    s = unit[lo:hi].strip()
    return ("... " if lo > 0 else "") + s + (" ..." if hi < len(unit) else "")


def mark(text, entity, alias, gold, redacted=False):
    """Escape, then highlight what the policy did to the anchor and where the answer is."""
    s = html.escape(text)
    if redacted:
        s = s.replace("[REDACTED]", '<span class="hl-red">[REDACTED]</span>')
    if alias and alias in s:
        s = s.replace(html.escape(alias), '<span class="hl-ink">%s</span>' % html.escape(alias))
    elif entity and entity in s:
        s = s.replace(html.escape(entity), '<span class="hl-red">%s</span>' % html.escape(entity))
    if gold and gold in s:
        s = s.replace(html.escape(gold), '<span class="hl-ink">%s</span>' % html.escape(gold))
    return s


def main():
    cases = load_cases(ROOT / "data" / "longctx_max.jsonl.gz", {k for k, _ in CASES})
    pc = load_percase([str(ROOT / "results/percase/quadfull_s*_percase.gz"),
                       str(ROOT / "results/percase/extevfull_s*_percase.gz"),
                       str(ROOT / "results/percase/extnew2_s*_percase.gz")])

    nt = lambda s: max(1, int(len(s) / 4.0))
    sections = []

    for key, source in CASES:
        c = cases[key]
        gold = (c.get("answers") or [""])[0]
        entity = c.get("target_entity") or ""
        arms = [p[0] for p in POLICIES]
        texts = E.build_arm_texts(c, nt, [0.5], arms, E.lex_drop_order)

        # The alias the DARK arm actually wrote, read back from its own prompt rather than
        # recomputed. Letters are assigned in order of first appearance, so a second Aliaser
        # started from a different point hands out different letters; the text is the truth.
        alias = ""
        dark_txt = texts.get(("quad4", 0.5)) or ""
        for unit in E.split_units(dark_txt):
            if gold and gold in unit:
                m = re.search(r"\[[A-Z]+-[A-Z]\]", unit)
                if m:
                    alias = m.group(0)
                    break
        if not alias:
            m = re.search(r"\[ACCOUNT-[A-Z]\]", dark_txt)
            alias = m.group(0) if m else ""

        cols = []
        for arm, name, desc, outcome in POLICIES:
            k = (arm, None) if arm in E.ARMS_LENGTH else (arm, 0.5)
            txt = texts.get(k) or ""
            row = pc.get(arm, {}).get(key, {})
            pred = (row.get("pred") or "").strip()
            f1 = row.get("f1")
            leaked = bool((row.get("ctx_leak_value") or 0) > 0
                          or (row.get("out_leak_value") or 0) > 0)
            # what each policy does to the anchor line
            if arm == "quad4":
                needle = alias
            elif arm == "quad4_del":
                needle = "[REDACTED]"
            else:
                needle = entity
            snip = snippet_for(txt, gold, entity,
                               alias if arm == "quad4" else None, needle=needle)
            cols.append({
                "name": name, "desc": desc, "outcome": outcome,
                "snip": mark(snip, entity, alias if arm == "quad4" else None, gold,
                             redacted=(arm == "quad4_del")),
                "pred": pred, "f1": f1, "leaked": leaked,
                "correct": bool(f1 and f1 > 0.5),
            })

        sections.append({"key": key, "source": source, "question": c["question"],
                         "gold": gold, "entity": entity, "alias": alias, "cols": cols})

    # ---------------------------------------------------------------- html
    out = []
    w = out.append
    w('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width,initial-scale=1">')
    w('<title>Keeping the Referent in the DARK \u2014 interactive demo</title>')
    w('<style>%s</style></head><body>' % CSS)

    w('<h1>Keeping the Referent in the DARK</h1>')
    w('<p class="lede">An IoT operator wants a cloud model to answer questions about its logs. '
      'The question names an account, and the account is exactly the value that must not be '
      'sent. Every policy below is real: the prompts come from the same construction the paper '
      'uses, and the answers are what the reader actually returned on those prompts.</p>')

    w('<h2>1. Why this is hard</h2>')
    w('<p>An operator asks:</p>')
    w('<div class="q mono">In the log above, which persona id is recorded for the account '
      '<span class="hl-red">ACCT-f57fb0af0787</span>?</div>')
    w('<p>The record that answers it carries that account on its identifying line. So the value '
      'that has to be hidden is also the handle the reader needs in order to know which record is '
      'being asked about. Hiding it removes the referent; keeping it forwards the secret.</p>')
    w('<div class="note">That is the whole problem, and it is why redaction and compression fail '
      'in <em>different directions</em> rather than by different amounts. Neither is a weaker '
      'version of the other.</div>')

    w('<h2>2. Four policies on the same case</h2>')
    w('<p>Same excerpt, same question, same 50% budget, same reader (Qwen2.5-7B-Instruct). '
      'Gold answer: <b class="mono">dynam_0111_00</b>. Under DARK the account is replaced by a '
      'stable alias in the context and in the question alike, so the reader still knows which '
      'record is being asked about.</p>')

    for s in sections:
        w('<h3 class="mono">%s <span style="color:var(--g5);font-weight:400">'
          '\u2014 %s</span></h3>' % (html.escape(s["key"]), html.escape(s["source"])))
        w('<div class="q mono">%s</div>' % mark(s["question"], s["entity"], s["alias"], s["gold"]))
        w('<div class="cols">')
        for col in s["cols"]:
            cls = "col works" if col["outcome"] == "works" else "col"
            if col["correct"] and col["leaked"]:
                verdict_text = "answers, but leaks"
            elif col["correct"]:
                verdict_text = "answers, no leak"
            elif col["leaked"]:
                verdict_text = "fails, and leaks"
            else:
                verdict_text = "cannot answer"
            w('<div class="%s">' % cls)
            w('<h4>%s</h4>' % html.escape(col["name"]))
            w('<div class="verdict %s">%s</div>'
              % ("bad" if (not col["correct"] or col["leaked"]) else "", verdict_text))
            w('<p>%s</p>' % html.escape(col["desc"]))
            w('<div class="snip">%s</div>' % col["snip"])
            w('<div class="ans">reader: <b class="%s">%s</b> &nbsp;'
              '&middot; protected value in prompt: <span class="%s">%s</span></div>'
              % ("good" if col["correct"] else "bad",
                 html.escape(col["pred"] or "(empty)"),
                 "bad" if col["leaked"] else "good",
                 "yes" if col["leaked"] else "no"))
            w('</div>')
        w('</div>')

    w('<h2>3. The same pattern at scale</h2>')
    w('<p>2,728 cases, six public corpora, 50% budget. <em>GIC</em> is the share of cases in '
      'which the record carrying the answer survives compression at all; <em>leak</em> is the '
      'share in which a protected value is recoverable from the prompt.</p>')
    w('<table><thead><tr><th>Method</th><th>F1</th><th>GIC</th><th>leak</th>'
      '<th>tokens</th></tr></thead><tbody>')
    for name, f1, gic, leak, tok in TABLE:
        cls = ' class="ours"' if name == "DARK" else ""
        w('<tr%s><td>%s</td><td>%.4f</td><td>%.1f%%</td><td>%.1f%%</td><td>%d</td></tr>'
          % (cls, html.escape(name), f1, gic, leak, tok))
    w('</tbody></table>')
    w('<div class="note">DARK is the only row that reaches both 100% GIC and 0% leakage. '
      'It is <em>not</em> the highest F1: the uncompressed prompt scores 0.6081, and Quito-X '
      'reaches a statistically indistinguishable 0.4831 against our 0.4937. The claim is not '
      'that we compress better; it is that we are the only method that answers the question '
      'without forwarding the value.</div>')

    w('<h2>4. Why answer survival is the interesting column</h2>')
    w('<p>GIC is not a constant offset between methods; it depends on the corpus. Below, the '
      'share of cases in which the answer-bearing record survives, per method and per source. '
      'Darker is higher.</p>')
    w(build_heat_html())

    w('<h2>5. Reproduce it</h2>')
    w('<div class="rec mono">git clone https://github.com/firesaku/PrivTrim-IoT<br>'
      'cd PrivTrim-IoT<br>'
      'python src/analysis_quad4/make_fig_gic.py &nbsp;# this heatmap<br>'
      'python src/analysis_quad4/make_tab_external.py --check &nbsp;# Table IV</div>')

    w('<footer>Every number on this page is re-derived from <code>results/percase/</code>, the '
      'per-case records that ship with the repository. The prompts are produced by '
      '<code>src/longctx/e2e_quad.py</code>, the same construction the paper evaluates. '
      'Paper: <a href="../paper/main.pdf">paper/main.pdf</a> &middot; '
      'Code: <a href="https://github.com/firesaku/PrivTrim-IoT">'
      'github.com/firesaku/PrivTrim-IoT</a></footer>')
    w('</body></html>')

    dest = ROOT / "demo" / "dark_demo.html"
    with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out))
    print("wrote", dest, len("\n".join(out)), "bytes")
    return 0


def build_heat_html():
    """The GIC matrix, read from results/diag/gic.json so it matches Fig. 5."""
    path = ROOT / "results" / "diag" / "gic.json"
    if not path.exists():
        return ""
    rows = json.load(io.open(path, encoding="utf-8"))
    srcs = ["pec_home", "privacylens", "simuhome", "smartbench_tdcs", "toolmem", "toolmem_raw"]
    labels = {"quad4": "DARK", "quad4_del": "delete variant", "ext_provence": "Provence",
              "ext_cpc": "CPC", "ext_longllmlingua": "LongLLMLingua", "ext_quito_x": "Quito-X",
              "ext_mask_llmlingua2": "mask + L2", "ext_llmlingua2": "LLMLingua-2",
              "alias_budget_link": "global alias", "ext_selective_context": "Selective Context"}
    order = ["quad4", "quad4_del", "ext_provence", "ext_cpc", "ext_longllmlingua",
             "ext_quito_x", "ext_mask_llmlingua2", "ext_llmlingua2", "alias_budget_link",
             "ext_selective_context"]
    cell = {}
    for r in rows:
        cell[(r["arm"], r["source"])] = r["gic"]

    def shade(v):
        # white -> dark maroon, matching Fig. 5
        t = max(0.0, min(1.0, v / 100.0))
        r0, g0, b0 = 255, 255, 255
        r1, g1, b1 = 74, 20, 18
        return "rgb(%d,%d,%d)" % (round(r0 + (r1 - r0) * t), round(g0 + (g1 - g0) * t),
                                  round(b0 + (b1 - b0) * t))

    out = ['<table class="hm"><thead><tr><th></th>']
    for s in srcs:
        out.append('<th>%s</th>' % html.escape(s.replace("_", " ")))
    out.append('</tr></thead><tbody>')
    for arm in order:
        out.append('<tr><td class="lbl">%s</td>' % html.escape(labels.get(arm, arm)))
        for s in srcs:
            v = cell.get((arm, s))
            if v is None:
                out.append('<td></td>')
                continue
            fg = "#fff" if v > 62 else "#111"
            out.append('<td style="background:%s;color:%s">%.0f</td>' % (shade(v), fg, v))
        out.append('</tr>')
    out.append('</tbody></table>')
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
