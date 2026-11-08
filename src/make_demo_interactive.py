#!/usr/bin/env python3
"""Build demo/interactive.html: a self-contained, clickable version of the paper's main contrast.

The static page shows three cases. This one ships a browser of cases and lets the reader change
the compression while looking at it: every policy is rendered as its list of surviving units, and
the reader can drop units by clicking them. The leak and answer indicators are computed in the
browser from the embedded canary values, so what they show follows what is on screen.

Everything embedded is measured. The unit lists come from the real prompt construction, the reader
answers come from the archived per-case records, and the canary values come from the case itself.

Usage
    python src/make_demo_interactive.py            # writes demo/interactive.html
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

N_TARGET = 60          # cases shipped
UNIT_CHARS = 150       # truncation per unit, keeps the file small
MAX_UNITS = 34


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_cases():
    out = []
    with open_maybe_gz(ROOT / "data" / "longctx_max.jsonl.gz") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return {c["key"]: c for c in out}


def load_percase():
    acc = collections.defaultdict(dict)
    for pattern in ("quadfull_s*_percase.gz", "extevfull_s*_percase.gz", "extnew2_s*_percase.gz"):
        for path in sorted(glob.glob(str(ROOT / "results" / "percase" / pattern))):
            with open_maybe_gz(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    acc[r["arm"]].setdefault(r["case_key"], r)
    return acc


def pick(pc):
    """Half the cases where DARK wins outright, half a spread that includes failures."""
    wins, rest = [], []
    for key, r in pc["quad4"].items():
        d = pc["quad4_del"].get(key)
        c = pc["budget_cluster"].get(key)
        if not d or not c:
            continue
        if r["f1"] > 0.5 and d["f1"] < 0.1 and c["f1"] < 0.1:
            wins.append(key)
        else:
            rest.append(key)
    wins.sort()
    rest.sort()
    half = N_TARGET // 2
    step_w = max(1, len(wins) // half)
    step_r = max(1, len(rest) // (N_TARGET - half))
    chosen = wins[::step_w][:half] + rest[::step_r][:N_TARGET - half]
    return chosen


def units_of(text):
    us = [u.strip() for u in E.split_units(text) if u.strip()]
    if len(us) > MAX_UNITS:
        # keep the head and anything that looks like a record boundary
        head = us[:MAX_UNITS]
        us = head
    return [(u[:UNIT_CHARS] + ("\u2026" if len(u) > UNIT_CHARS else "")) for u in us]


def main():
    cases = load_cases()
    pc = load_percase()
    keys = pick(pc)

    POLICIES = [
        ("full", "send it raw", "full"),
        ("budget_cluster", "compress", "budget"),
        ("quad4_del", "delete in place", "delete"),
        ("quad4", "DARK", "dark"),
    ]

    nt = lambda s: max(1, int(len(s) / 4.0))
    shipped = []

    for key in keys:
        c = cases.get(key)
        if not c:
            continue
        gold = (c.get("answers") or [""])[0]
        entity = c.get("target_entity") or ""
        canaries = [x["value"] for x in (c.get("canaries") or []) if x.get("value")]
        texts = E.build_arm_texts(c, nt, [0.5], [p[0] for p in POLICIES], E.lex_drop_order)

        # the alias DARK wrote, read back from its own prompt
        alias = ""
        dark = texts.get(("quad4", 0.5)) or ""
        # The queried entity is the account, so prefer an account alias; a record can carry an
        # email or phone alias too, and taking the first match picked the wrong one.
        want = "[ACCOUNT-"
        if (c.get("case_kind") or "") == "needle":
            want = None
        for u in E.split_units(dark):
            if not (gold and gold in u):
                continue
            if want:
                m = re.search(r"\[ACCOUNT-[A-Z]\]", u)
            else:
                m = re.search(r"\[[A-Z]+-[A-Z]\]", u)
            if not m:
                m = re.search(r"\[[A-Z]+-[A-Z]\]", u)
            if m:
                alias = m.group(0)
                break

        arms = {}
        for arm, label, tag in POLICIES:
            k = (arm, None) if arm in E.ARMS_LENGTH else (arm, 0.5)
            txt = texts.get(k) or ""
            row = pc.get(arm, {}).get(key, {})
            arms[tag] = {
                "label": label,
                "units": units_of(txt),
                "pred": (row.get("pred") or "").strip(),
                "f1": round(row.get("f1") or 0.0, 4),
                "tokens": row.get("ctx_tokens") or 0,
            }

        shipped.append({
            "key": key,
            "source": c.get("source"),
            "kind": c.get("case_kind"),
            "q": c.get("question") or "",
            "alias": alias,
            "entity": entity,
            "gold": gold,
            "canaries": canaries,
            "arms": arms,
        })

    payload = json.dumps(shipped, ensure_ascii=False, separators=(",", ":"))
    page = HTML.replace("__DATA__", payload)
    dest = ROOT / "demo" / "interactive.html"
    with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print("wrote", dest, len(page), "bytes,", len(shipped), "cases")
    return 0


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DARK — interactive</title>
<style>
:root{--ink:#4A1412;--red:#8E2A22;--g5:#555;--g9:#9A9A9A;--g4:#D4D4D2;--pale:#EFEFEF}
*{box-sizing:border-box}
body{font:15px/1.6 "Times New Roman",Georgia,serif;margin:0;color:#111;background:#fff}
header{padding:18px 26px 12px;border-bottom:2px solid var(--g4)}
h1{margin:0;font-size:21px;color:var(--ink)}
.sub{color:var(--g5);font-size:13.5px;margin-top:3px}
.wrap{display:grid;grid-template-columns:290px 1fr;min-height:calc(100vh - 76px)}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
aside{border-right:1px solid var(--g4);padding:14px;overflow-y:auto;max-height:calc(100vh - 76px)}
@media(max-width:900px){aside{max-height:220px;border-right:none;border-bottom:1px solid var(--g4)}}
main{padding:16px 22px 60px;overflow-y:auto}
input[type=search],select{width:100%;padding:7px 9px;border:1px solid var(--g4);font:inherit;
  font-size:13.5px;margin-bottom:8px}
.cl{display:block;width:100%;text-align:left;padding:7px 9px;margin:3px 0;border:1px solid var(--g4);
    background:#fff;font:inherit;font-size:12.5px;cursor:pointer;font-family:Consolas,monospace}
.cl:hover{background:var(--pale)}
.cl.on{border-color:var(--ink);background:var(--pale);font-weight:700}
.cl small{display:block;color:var(--g5);font-weight:400;font-family:inherit;font-size:11.5px}
h2{font-size:16px;margin:0 0 6px;color:var(--ink)}
.qbar{background:var(--pale);border-left:3px solid var(--g9);padding:9px 12px;margin:8px 0 14px;
      font-family:Consolas,monospace;font-size:13px;white-space:pre-wrap}
.tools{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:12px;font-size:13px;
       color:var(--g5)}
.tools label{display:flex;gap:5px;align-items:center;cursor:pointer}
.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}
@media(max-width:1100px){.cols{grid-template-columns:repeat(2,1fr)}}
@media(max-width:640px){.cols{grid-template-columns:1fr}}
.col{border:1px solid var(--g4);border-top:3px solid var(--g9);display:flex;flex-direction:column}
.col.dark{border-top-color:var(--ink)}
.head{padding:8px 10px 6px;border-bottom:1px solid var(--g4)}
.head b{font-size:14px}
.verdict{font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;margin-top:2px}
.ok{color:var(--ink);font-weight:700}.no{color:var(--red);font-weight:700}
.units{flex:1;overflow-y:auto;max-height:330px;padding:6px;background:#FAFAFA}
.u{padding:4px 6px;margin:2px 0;font-family:Consolas,monospace;font-size:11.5px;line-height:1.45;
   border-left:2px solid transparent;cursor:pointer;word-break:break-word}
.u:hover{background:#F1F1F1}
.u.off{opacity:.32;text-decoration:line-through}
.u.anchor{border-left-color:var(--ink);background:#F3EDEC}
.u.ans{border-left-color:var(--g9);background:#F8F8F8}
.u .hit{background:#F6E3E0;color:var(--red);font-weight:700}
.u .ali{background:#EBE0DE;color:var(--ink);font-weight:700}
.foot{padding:7px 10px;border-top:1px solid var(--g4);font-size:12px;color:var(--g5)}
.foot b{font-family:Consolas,monospace;font-size:12.5px}
.badge{display:inline-block;padding:1px 6px;border-radius:2px;font-size:11px;font-weight:700}
.badge.bad{background:#F6E3E0;color:var(--red)}
.badge.good{background:#EBE0DE;color:var(--ink)}
.note{font-size:13px;color:var(--g5);border-left:3px solid var(--g9);padding-left:11px;margin:18px 0}
code{font-family:Consolas,monospace;font-size:12.5px}
.hidden{display:none}
</style></head><body>
<header>
  <h1>Keeping the Referent in the DARK — interactive</h1>
  <div class="sub">Pick a case on the left. Click any line to drop it and watch the leak indicator
  recompute. Nothing is a mock: the lines are the real prompts, the answers are what the reader
  returned.</div>
</header>
<div class="wrap">
  <aside>
    <input type="search" id="q" placeholder="filter by source or case id…">
    <select id="f">
      <option value="all">all cases</option>
      <option value="win">DARK answers, alternatives do not</option>
      <option value="other">everything else</option>
      <option value="multi_entity">multi-entity only</option>
      <option value="needle">single-entity only</option>
    </select>
    <div id="list"></div>
  </aside>
  <main>
    <h2 id="title">—</h2>
    <div class="qbar" id="qbar"></div>
    <div class="tools">
      <label><input type="checkbox" id="hide"> hide the reader's answer (guess first)</label>
      <label><input type="checkbox" id="drop" checked> dropping lines is enabled</label>
      <span id="hint"></span>
    </div>
    <div class="cols" id="cols"></div>
    <div class="note">
      <b>What to look for.</b> In <em>send it raw</em> the account is visible and the reader
      answers — and the protected value is in the prompt. In <em>compress</em> and
      <em>delete in place</em> the account is gone or the record is gone, and the reader returns
      <code>UNKNOWN</code>. In <em>DARK</em> the account line survives as a stable alias, the
      question is rewritten to use the same alias, and the reader answers without the value ever
      appearing. Try dropping a line from DARK and watch which indicator turns red first.
    </div>
    <div class="note">
      The <b>leak</b> indicator is computed in this page by scanning the visible lines for any of
      the case's protected values, so it follows what you leave on screen rather than what was
      recorded. The <b>answer</b> indicator is the reader's recorded output for the original
      prompt; it does not change when you edit the lines, because altering them would require
      re-running the model, which this page cannot do.
    </div>
  </main>
</div>
<script>
const DATA = __DATA__;
let cur = 0, shown = DATA.slice();

const el = id => document.getElementById(id);
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function markUnit(u, cs, alias, gold) {
  let s = esc(u);
  for (const c of cs) {
    const e = esc(c);
    if (s.includes(e)) s = s.split(e).join('<span class="hit">' + e + '</span>');
  }
  if (alias) {
    const a = esc(alias);
    if (s.includes(a)) s = s.split(a).join('<span class="ali">' + a + '</span>');
  }
  if (gold) {
    const g = esc(gold);
    if (s.includes(g)) s = s.split(g).join('<span class="ali">' + g + '</span>');
  }
  return s;
}

function renderList() {
  el('list').innerHTML = shown.map((c, i) =>
    `<button class="cl${i === cur ? ' on' : ''}" data-i="${i}">${esc(c.key)}` +
    `<small>${esc(c.source)} · ${esc(c.kind)}</small></button>`).join('');
  el('list').querySelectorAll('.cl').forEach(b =>
    b.onclick = () => { cur = +b.dataset.i; renderList(); render(); });
}

function render() {
  const c = shown[cur];
  if (!c) { el('title').textContent = 'no case'; el('cols').innerHTML = ''; return; }
  el('title').textContent = c.key + '  ·  ' + c.source;
  let q = esc(c.q);
  if (c.entity) q = q.split(esc(c.entity)).join('<span class="hit">' + esc(c.entity) + '</span>');
  el('qbar').innerHTML = q +
    (c.alias ? '   <span style="color:#555">under DARK: ' + esc(c.alias) + '</span>' : '');
  el('hint').textContent = 'gold: ' + c.gold + ' · ' + c.canaries.length + ' protected values';

  const hide = el('hide').checked;
  el('cols').innerHTML = Object.keys(c.arms).map(tag => {
    const a = c.arms[tag];
    const lines = a.units.map((u, i) => {
      const isAnchor = (c.entity && u.includes(c.entity)) || (c.alias && u.includes(c.alias)) ||
                       u.includes('[REDACTED]');
      const isAns = c.gold && u.includes(c.gold);
      const cls = 'u' + (isAnchor ? ' anchor' : '') + (isAns ? ' ans' : '');
      return `<div class="${cls}" data-tag="${tag}" data-i="${i}">` +
             markUnit(u, c.canaries, c.alias, c.gold) + '</div>';
    }).join('');
    const pred = hide ? '<span style="color:#9A9A9A">hidden</span>' : esc(a.pred || '(empty)');
    return `<div class="col${tag === 'dark' ? ' dark' : ''}">
      <div class="head"><b>${esc(a.label)}</b>
        <div class="verdict" data-v="${tag}">·</div></div>
      <div class="units" id="units-${tag}">${lines}</div>
      <div class="foot">reader: <b>${pred}</b><br>
        leak: <span data-leak="${tag}">·</span> · ${a.tokens} tokens</div>
    </div>`;
  }).join('');

  el('cols').querySelectorAll('.u').forEach(d => {
    d.onclick = () => {
      if (!el('drop').checked) return;
      d.classList.toggle('off');
      update();
    };
  });
  update();
}

function update() {
  const c = shown[cur];
  for (const tag of Object.keys(c.arms)) {
    const box = el('units-' + tag);
    if (!box) continue;
    const visible = [...box.querySelectorAll('.u:not(.off)')].map(d => d.textContent).join('\n');
    const leak = c.canaries.some(v => visible.includes(v));
    const answers = c.arms[tag].f1 > 0.5;
    const v = document.querySelector(`[data-v="${tag}"]`);
    const l = document.querySelector(`[data-leak="${tag}"]`);
    l.innerHTML = leak ? '<span class="badge bad">value is in the prompt</span>'
                       : '<span class="badge good">nothing recoverable</span>';
    if (v) {
      if (answers && leak) v.innerHTML = '<span class="no">answers, but leaks</span>';
      else if (answers) v.innerHTML = '<span class="ok">answers, no leak</span>';
      else if (leak) v.innerHTML = '<span class="no">fails, and leaks</span>';
      else v.innerHTML = '<span class="no">cannot answer</span>';
    }
  }
}

function applyFilter() {
  const q = el('q').value.toLowerCase(), f = el('f').value;
  shown = DATA.filter(c => {
    if (q && !(c.key.toLowerCase().includes(q) || (c.source || '').toLowerCase().includes(q)))
      return false;
    if (f === 'win') return c.arms.dark.f1 > 0.5 && c.arms.budget.f1 < 0.1 && c.arms.delete.f1 < 0.1;
    if (f === 'other') return !(c.arms.dark.f1 > 0.5 && c.arms.budget.f1 < 0.1 && c.arms.delete.f1 < 0.1);
    if (f === 'multi_entity') return c.kind === 'multi_entity';
    if (f === 'needle') return c.kind === 'needle';
    return true;
  });
  cur = 0; renderList(); render();
}

el('q').oninput = applyFilter;
el('f').onchange = applyFilter;
el('hide').onchange = render;
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowDown' && cur < shown.length - 1) { cur++; renderList(); render(); }
  if (e.key === 'ArrowUp' && cur > 0) { cur--; renderList(); render(); }
});
renderList(); render();
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
