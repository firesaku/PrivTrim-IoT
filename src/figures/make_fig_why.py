"""Generate Fig. 1 as an editable .drawio file, then export to PDF/PNG.

Units: 1 unit = 0.01 in (draw.io exports 100 units/inch), so a font that must print at
N pt carries fontSize = N / 0.72.  Page is 710 x 410 units = 7.10 x 4.10 in.

Icons are Feather-style 24x24 stroke glyphs embedded as base64 SVG data URIs, so they stay
crisp and identical on every machine and the file remains fully editable in draw.io.
"""
import base64
import subprocess
from pathlib import Path

W, H = 710, 430
OUT = Path(r'D:\KU-Assignment\CyberSecurity\Paper\Latex V1\figures')
DRAWIO = Path(r'D:\HPC\drawio\fig_why.drawio')
EXE = r'C:\Users\Yuki\AppData\Local\Programs\draw.io\draw.io.exe'

INK, RED = '#4A1412', '#8E2A22'
GREEN_ = '#4A1412'   # 与全文统一：黑色代表我们的结果，红色代表泄漏
G_D, G_M, G_L, PALE = '#555555', '#9A9A9A', '#D4D4D2', '#EFEFEF'
FF = 'Times New Roman'
FM = 'Courier New'


def fs(pt):
    return round(pt / 0.72, 2)


# ------------------------------------------------------------------ icons
ICONS = {
    'moon': '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
    'terminal': '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
    'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
                '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
                '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    'scissors': '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/>'
                '<line x1="20" y1="4" x2="8.12" y2="15.88"/>'
                '<line x1="14.47" y1="14.48" x2="20" y2="20"/>'
                '<line x1="8.12" y1="8.12" x2="12" y2="12"/>',
    'filter': '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    'cloud': '<path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/>',
    'chat': '<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1'
            '-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 '
            '0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    'key': '<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 '
           '7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>',
    'link': '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
            '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    'grid': '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>'
            '<rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    'layers': '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
              '<polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    'shield': '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    'alert': '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 '
             '3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/>'
             '<line x1="12" y1="17" x2="12.01" y2="17"/>',
    'lock': '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>'
            '<path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    'bars': '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="14"/>',
    'book': '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
            '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    'scale': '<path d="M12 3v18"/><path d="M5 7h14"/>'
             '<path d="M5 7l-3 6a3 3 0 0 0 6 0z"/><path d="M19 7l-3 6a3 3 0 0 0 6 0z"/>'
             '<path d="M8 21h8"/>',
    'arrow-down': '<line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/>',
}


def icon_uri(name, color=INK, sw=1.7):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           'stroke="%s" stroke-width="%s" stroke-linecap="round" stroke-linejoin="round">'
           '%s</svg>' % (color, sw, ICONS[name]))
    enc = svg.replace('<', '%3C').replace('>', '%3E').replace('#', '%23').replace('"', '%22')
    return 'data:image/svg+xml,' + enc


# ------------------------------------------------------------------ cells
cells = []


def add(style, x, y, w, h, value=''):
    """y is given in bottom-up coordinates and flipped here, because draw.io counts from
    the top-left corner."""
    cid = 'c%d' % (len(cells) + 2)
    cells.append((cid, value, style, x, H - y - h, w, h))


def rect(x, y, w, h, color=INK, fill='#FFFFFF', lw=0.9, arc=8):
    add('rounded=0;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=%s;strokeWidth=%s;'
        % (fill, color, lw), x, y, w, h)


def txt(x, y, w, h, s, pt=10, color=INK, bold=False, align='center', va='middle',
        family=FF, italic=False):
    st = 1 if bold else 0
    if italic:
        st += 2
    add('text;html=1;strokeColor=none;fillColor=none;align=%s;verticalAlign=%s;'
        'fontSize=%s;fontFamily=%s;fontColor=%s;fontStyle=%d;whiteSpace=wrap;'
        % (align, va, fs(pt), family, color, st), x, y, w, h, s)


def ic(name, x, y, size=26, color=INK):
    add('shape=image;image=%s;aspect=fixed;html=1;' % icon_uri(name, color), x, y, size, size)


def arrow(x, y, w=10, h=10, color=G_M):
    add('shape=image;image=%s;aspect=fixed;html=1;'
        % icon_uri('arrow-down', color), x, y, w, h)


def hline(x, y, w, color=G_M, lw=0.8):
    add('line;strokeWidth=%s;html=1;fillColor=none;strokeColor=%s;' % (lw, color), x, y, w, 0)


# ------------------------------------------------------------------ frame
add('rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;', 0, 0, W, H)
rect(4, 4, W - 8, H - 8, color=RED, fill='none', lw=1.1)
ic('moon', 24, 398, 20)
txt(50, 397, 70, 22, 'DARK', 11, INK, bold=True, align='left')

# ------------------------------------------------------------------ bands
TY, TH = 284, 108
BY, BH = 52, 222
tops = [(4, 172), (180, 172), (356, 172), (532, 174)]
bots = [(4, 136), (144, 136), (284, 136), (424, 136), (564, 142)]

for x, w in tops:
    rect(x, TY, w, TH)
for x, w in bots:
    rect(x, BY, w, BH)
for gx in (176, 352, 528):
    arrow(gx + 1, TY + TH // 2 - 5, 9, 11)

rect(5, 47, W - 10, 1.2, color=G_L, fill=G_L, lw=0, arc=0)
txt(6, 16, W - 12, 20,
    'DARK matches the strongest compressor on F1 at zero measured leakage, on half the tokens.',
    10.5, INK, bold=True)
txt(6, 4, W - 12, 12,
    '6 corpora \u00b7 2,728 evaluation cases \u00b7 112 main \u00b7 3 \u00d7 300 public variants',
    7.5, G_D)

# ------------------------------------------------------------------ top panels
x, w = tops[0]
txt(x, TY + TH - 24, w, 18, 'ANALYST REQUEST', 10.5, RED, bold=True)
ic('terminal', x + w // 2 - 20, TY + 34, 40)
ic('chat', x + 6, TY + 12, 16, RED)
txt(x + 26, TY + 2, w - 32, 32, 'which device is bound to ACCT-4cd0?', 10, INK, align='left')

x, w = tops[1]
txt(x, TY + TH - 24, w, 18, 'LOG STORE', 10.5, INK, bold=True)
ic('database', x + w // 2 - 20, TY + 34, 40)
txt(x, TY + 16, w, 16, '19,838 tokens', 10, INK)
txt(x, TY + 2, w, 14, '6 entities on every line', 10, G_D)

x, w = tops[2]
txt(x, TY + TH - 24, w, 18, 'THE CONFLICT', 10.5, RED, bold=True)
ic('scissors', x + 36, TY + 34, 38, RED)
hline(x + w // 2, TY + 38, 1, G_L)
ic('filter', x + w - 74, TY + 34, 38, RED)
txt(x, TY + 16, w, 16, 'redact \u2192 no handle', 10, RED, bold=True)
txt(x, TY + 2, w, 14, 'compress \u2192 88.8\u2013100% leak', 10, RED)

x, w = tops[3]
txt(x, TY + TH - 24, w, 18, 'CLOUD READER', 10.5, INK, bold=True)
ic('cloud', x + w // 2 - 22, TY + 32, 44)
txt(x, TY + 16, w, 16, "receives (q', P)", 10, INK)
txt(x, TY + 2, w, 14, 'one short answer', 10, G_D)

# ------------------------------------------------------------------ I
x, w = bots[0]
txt(x, BY + BH - 30, w, 16, 'I.  DATA EXAMPLE', 10, INK, bold=True)
ic('layers', x + 6, BY + BH - 66, 25, G_D)
ic('key', x + 35, BY + BH - 66, 25, INK)
ic('link', x + 64, BY + BH - 66, 23, INK)
rect(x + 6, BY + 62, w - 12, 62, color=G_M, fill=PALE, lw=0.7)
txt(x + 6, BY + 96, w - 12, 16, 'media_player', 10, INK, family=FM)
txt(x + 6, BY + 80, w - 12, 16, 'ACCT-4cd0', 10, INK, bold=True, family=FM)
txt(x + 6, BY + 64, w - 12, 16, 'temp 21.4', 10, INK, family=FM)
txt(x + 6, BY + 42, w - 12, 16, 'same block', 10, G_D)
txt(x + 6, BY + 26, w - 12, 16, 'in every case', 10, INK)
txt(x + 6, BY + 10, w - 12, 14, 'multi / needle', 10, G_D)

# ------------------------------------------------------------------ II  the rule
x, w = bots[1]
txt(x, BY + BH - 30, w, 16, 'II.  THE RULE', 10, INK, bold=True)

CW, CH, GP = 48, 54, 4
GX = x + 34
ROW_TOP = BY + 124
ROW_BOT = BY + 66
grid_cells = [
    (GX,                 ROW_TOP, 'DELETE', 'scissors', False),
    (GX + CW + GP,       ROW_TOP, 'ALIAS',  'key',      True),
    (GX,                 ROW_BOT, 'KEEP',   'shield',   False),
    (GX + CW + GP,       ROW_BOT, 'REFILL', 'layers',   False),
]
for cx, cy, label, icon_name, hero in grid_cells:
    if hero:
        rect(cx, cy, CW, CH, color=INK, fill=INK, lw=1.0)
        ic(icon_name, cx + CW // 2 - 7, cy + CH - 21, 15, '#FFFFFF')
        txt(cx, cy + 7, CW, 13, label, 9, '#FFFFFF', bold=True)
    else:
        rect(cx, cy, CW, CH, color=G_L, fill='#FBFBFB', lw=0.8)
        ic(icon_name, cx + CW // 2 - 7, cy + CH - 21, 15, G_M)
        txt(cx, cy + 7, CW, 13, label, 9, G_D, bold=True)

txt(x + 1, ROW_TOP + CH // 2 - 6, 31, 12, 'high', 7.5, G_D, align='left')
txt(x + 1, ROW_BOT + CH // 2 - 6, 31, 12, 'low',  7.5, G_D, align='left')
txt(GX, BY + 48, 2 * CW + GP, 12, 'relevance   low \u2192 high', 7.5, G_D)

# ------------------------------------------------------------------ III
x, w = bots[2]
txt(x, BY + BH - 30, w, 16, 'III.  NO SAFE', 10, RED, bold=True)
txt(x, BY + BH - 48, w, 16, 'CHOICE', 10, RED, bold=True)
ic('shield', x + 8, BY + BH - 84, 25, G_D)
ic('scale', x + w // 2 - 12, BY + BH - 84, 25, G_D)
ic('alert', x + w - 33, BY + BH - 84, 25, RED)
txt(x, BY + 104, w, 16, 'two failures, one cause', 9, G_D, italic=True)
rect(x + 10, BY + 90, w - 20, 1.8, color=G_M, fill=G_M, lw=0, arc=0)
add('shape=ellipse;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=none;' % G_M,
    x + 26, BY + 86, 9, 9)
add('shape=ellipse;whiteSpace=wrap;html=1;fillColor=%s;strokeColor=none;' % RED,
    x + w - 35, BY + 86, 9, 9)
txt(x + 2, BY + 68, 64, 18, '0.1712', 10, INK, bold=True)
txt(x + w - 66, BY + 68, 64, 18, '0.4831', 10, RED, bold=True)
txt(x + 2, BY + 50, 64, 18, 'mask', 10, G_D)
txt(x + w - 66, BY + 50, 64, 18, 'Quito-X', 10, G_D)
txt(x + 2, BY + 32, 64, 18, '0%', 10, G_D)
txt(x + w - 66, BY + 32, 64, 18, '98.1%', 10, RED)
txt(x, BY + 12, w, 16, 'leak rate', 10, G_D, italic=True)

# ------------------------------------------------------------------ IV  leaderboard
x, w = bots[3]
txt(x, BY + BH - 30, w, 16, 'IV.  DARK', 10, INK, bold=True)
ic('lock', x + 8, BY + BH - 62, 20, INK)
ic('scale', x + w // 2 - 10, BY + BH - 62, 20, G_D)
ic('book', x + w - 30, BY + BH - 62, 20, G_D)
board = [('DARK', '0.4937', True), ('Quito-X', '0.4831', False),
         ('CPC', '0.4740', False), ('Provence', '0.4080', False),
         ('LLMLingua-2', '0.2651', False)]
for k, (name, val, ours) in enumerate(board):
    ry = BY + 132 - k * 17
    if ours:
        rect(x + 4, ry - 2, w - 8, 17, color=INK, fill=INK, lw=0.6)
        txt(x + 6, ry, 90, 15, name, 10, '#FFFFFF', bold=True, align='left')
        txt(x + w - 40, ry, 38, 15, val, 11, '#FFFFFF', bold=True, align='right')
    else:
        txt(x + 6, ry, 90, 15, name, 10, G_D, align='left')
        txt(x + w - 40, ry, 38, 15, val, 10, G_D, align='right')
txt(x + 4, BY + 48, w - 8, 12, 'vs Quito-X: tie (n.s.)', 8, G_D, italic=True, align='left')
txt(x + 4, BY + 30, w - 8, 13, 'leak', 9, G_D, align='left')
txt(x + 28, BY + 29, 26, 15, '0%', 10, GREEN_, bold=True, align='left')
txt(x + 50, BY + 30, 84, 13, 'vs 98.1%', 9, RED, align='left')
txt(x + 4, BY + 14, w - 8, 12, 'others: 88.8-100%', 8, RED, align='left')

# ------------------------------------------------------------------ V
x, w = bots[4]
txt(x, BY + BH - 30, w, 16, 'V.  WHERE IT', 10, INK, bold=True)
txt(x, BY + BH - 48, w, 16, 'STAYS HARD', 10, INK, bold=True)
ic('bars', x + 8, BY + BH - 84, 25, G_D)
ic('alert', x + w - 33, BY + BH - 84, 25, RED)
base, maxh = BY + 84, 46
for k, (val, col, lab) in enumerate([(0.3273, RED, 'multi'), (0.6970, G_M, 'needle')]):
    bx = x + 12 + k * 62
    hgt = int(maxh * val / 0.70)
    rect(bx, base, 50, hgt, color='#FFFFFF', fill=col, lw=0.6)
    txt(bx, base + hgt // 2 - 8, 50, 16, '%.4f' % val, 10,
        '#FFFFFF' if k == 0 else INK, bold=True)
    txt(bx, base - 14, 50, 14, lab, 10, G_D)
txt(x + 6, BY + 46, w - 12, 16, 'toolmem_raw 0.263', 10, G_D)
txt(x + 6, BY + 28, w - 12, 16, 'every arm < 0.45', 10, G_D)

# ------------------------------------------------------------------ write
body = '\n'.join(
    '        <mxCell id="%s" value="%s" style="%s" vertex="1" parent="1">\n'
    '          <mxGeometry x="%s" y="%s" width="%s" height="%s" as="geometry"/>\n'
    '        </mxCell>' % (cid, v.replace('&', '&amp;').replace('<', '&lt;')
                        .replace('>', '&gt;').replace('"', '&quot;'),
                        st, x, y, w, h)
    for cid, v, st, x, y, w, h in cells)

xml = ('<mxfile host="app.diagrams.net" version="24.7.17">\n'
       '  <diagram id="figwhy" name="Fig1">\n'
       '    <mxGraphModel dx="1000" dy="700" grid="0" gridSize="10" guides="1" '
       'tooltips="1" connect="0" arrows="1" fold="1" page="1" pageScale="1" '
       'pageWidth="%d" pageHeight="%d" math="0" shadow="0">\n'
       '      <root>\n'
       '        <mxCell id="0"/>\n'
       '        <mxCell id="1" parent="0"/>\n'
       '%s\n'
       '      </root>\n'
       '    </mxGraphModel>\n'
       '  </diagram>\n'
       '</mxfile>\n') % (W, H, body)

DRAWIO.parent.mkdir(parents=True, exist_ok=True)
DRAWIO.write_text(xml, encoding='utf-8', newline='\n')
print('wrote', DRAWIO, len(xml), 'bytes')

for fmt, target in (('pdf', OUT / 'fig_why.pdf'), ('png', OUT / 'fig_why.png')):
    cmd = [EXE, '--export', '--format', fmt, '--output', str(target),
           str(DRAWIO), '--no-sandbox']
    if fmt == 'png':
        cmd.insert(-2, '--scale')
        cmd.insert(-2, '3')
    subprocess.run(cmd, check=False, capture_output=True)
    print(fmt, 'exported' if target.exists() else 'FAILED', target.stat().st_size
          if target.exists() else '')
