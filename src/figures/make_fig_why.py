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

INK, RED = '#141414', '#8E2A22'
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


# ======================================================================
# Layout: three narrative tiers, unequal weight.
#   A  the problem        four stages, thin separators, no cards
#   B  the argument       example (narrow) | two failures | DARK (widest)
#   C  evidence/boundary  one line, secondary
# ======================================================================
W, H = 710, 302

# add() writes in flipped y, so wrap the primitives to take "distance from the top".
_F = lambda y, h: H - y - h
_rect_o, _txt_o, _ic_o, _hline_o, _arrow_o = rect, txt, ic, hline, arrow
def rect(x, y, w, h, *a, **k):   return _rect_o(x, _F(y, h), w, h, *a, **k)
def txt(x, y, w, h, *a, **k):    return _txt_o(x, _F(y, h), w, h, *a, **k)
def ic(n, x, y, size=26, color=INK): return _ic_o(n, x, _F(y, size), size, color)
def hline(x, y, w, *a, **k):     return _hline_o(x, _F(y, 0), w, *a, **k)
def arrow(x, y, w=10, h=10, *a, **k): return _arrow_o(x, _F(y, h), w, h, *a, **k)
INK   = '#1F2937'
G_D   = '#64748B'
G_M   = '#697586'
G_L   = '#D8DEE6'
RED   = '#B6403A'
BLUE  = '#0072B2'
GREEN = '#009E73'
GOLD  = '#D69A35'

add('rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;', 0, 0, W, H)

# ---------------------------------------------------------------- A  problem
AY = 10
stages = [
    (8,   'analyst request', 'which device is bound', 'to ACCT-4cd0?', 'terminal'),
    (186, 'log store',       '19,838 tokens',        '6 entries per line', 'database'),
    (364, 'the conflict',    'redact: no handle',    'compress: value sent', 'alert'),
    (542, 'cloud reader',    'receives (q-prime, P)', 'returns one answer', 'cloud'),
]
for x, title, l1, l2, ic_name in stages:
    txt(x, AY, 168, 12, title.upper(), 8, G_D, bold=True, align='left')
    ic(ic_name, x, AY + 13, 15, RED if title == 'the conflict' else G_M)
    txt(x + 21, AY + 12, 145, 10, l1, 9, INK, align='left')
    col = RED if title == 'the conflict' else G_D
    txt(x + 21, AY + 23, 145, 10, l2, 9, col, align='left')
for gx in (176, 354, 532):
    arrow(gx, AY + 17, 8, 9, G_L)

hline(8, AY + 40, W - 16, G_L, 0.8)

# ---------------------------------------------------------------- B  argument
BY = 62
BH = 176
# widths are deliberately unequal: 168 / 208 / 302
rect(8,   BY, 168, BH, G_L, '#FFFFFF', 0.9)
rect(184, BY, 208, BH, G_L, '#FFFFFF', 0.9)
rect(400, BY, 302, BH, GREEN, '#FFFFFF', 1.1)     # the payoff gets the accent

# --- B1 the example (auxiliary)
txt(16, BY + 7, 152, 11, 'THE EXAMPLE', 8, G_D, bold=True, align='left')
rect(16, BY + 22, 152, 74, G_L, '#FAFBFC', 0.8)
txt(22, BY + 26, 140, 15, 'ACCT-4cd0', 10, RED, bold=True, align='left', family='Courier New')
txt(22, BY + 43, 140, 15, 'media_player', 10, GREEN, bold=True, align='left', family='Courier New')
txt(22, BY + 60, 140, 14, 'temp 21.4', 9, G_D, align='left', family='Courier New')
txt(22, BY + 76, 140, 14, 'status active', 9, G_D, align='left', family='Courier New')
txt(16, BY + 101, 152, 10, 'one record holds both', 8, G_D, align='left', italic=True)
txt(16, BY + 113, 152, 26,
    'the value to hide is also the anchor that locates the answer', 7.5, G_D, align='left')

# --- B2 two failures (strengthened, qualitative only)
txt(192, BY + 7, 192, 11, 'TWO WAYS TO FAIL', 8, RED, bold=True, align='left')
for i, (op, what, why) in enumerate([
        ('redact', 'handle lost', 'the reader cannot tell\nwhich record is meant'),
        ('compress', 'value exposed', 'the handle arrives\ntogether with the value')]):
    yy = BY + 26 + i * 62
    txt(192, yy, 90, 11, op, 10, INK, bold=True, align='left')
    txt(192, yy + 13, 190, 10, what, 9, RED, align='left')
    txt(192, yy + 24, 190, 22, why.replace('\n', ' '), 7.5, G_D, align='left')
hline(192, BY + 150, 190, G_L, 0.8)
txt(192, BY + 156, 190, 14, 'each keeps one and loses the other', 7.5, G_D,
    align='left', italic=True)

# --- B3 DARK (the visual centre)
txt(410, BY + 7, 280, 11, 'DARK', 9, INK, bold=True, align='left')
txt(410, BY + 20, 120, 34, '0.4937', 22, GREEN, bold=True, align='left')
txt(410, BY + 54, 120, 10, 'F1 at 50% budget', 8, G_D, align='left')

txt(536, BY + 24, 156, 20, '0.4831', 15, G_M, bold=True, align='left')
txt(536, BY + 44, 156, 10, 'strongest compressor', 8, G_D, align='left')
txt(536, BY + 55, 156, 10, 'difference not significant', 7.5, G_D, align='left', italic=True)

hline(410, BY + 74, 282, G_L, 0.8)
txt(410, BY + 80, 140, 10, 'MEASURED LEAKAGE', 8, G_D, bold=True, align='left')
txt(410, BY + 92, 60, 24, '0%', 17, GREEN, bold=True, align='left')
txt(452, BY + 99, 60, 12, 'vs 98.1%', 10, RED, align='left')
txt(516, BY + 99, 176, 12, 'for that compressor', 8, G_D, align='left')

txt(410, BY + 124, 282, 10, 'REFERENT', 8, G_D, bold=True, align='left')
txt(410, BY + 136, 282, 11, 'retained, not deleted', 9, INK, align='left')
txt(410, BY + 150, 282, 22,
    'the record stays, the value inside it is replaced', 7.5, G_D, align='left')

# ---------------------------------------------------------------- C  evidence
CY = 252
hline(8, CY, W - 16, G_L, 0.8)
txt(8, CY + 6, 350, 11,
    '2,728 cases  |  6 corpora  |  50% token budget  |  no answer key  |  no training',
    8, G_D, align='left')
txt(360, CY + 6, 342, 22,
    'boundary: the gain tracks the record layout, and Section VIII measures where it stops',
    7.5, G_D, align='left', italic=True)

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

