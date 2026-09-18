"""
a2_digitize_source.py — Extract the plotted data of the delivered English
Appendix Figure A2 (03_Supplementary/Appendix_Figure_A2.png) at pixel level.

Why: the generator of the delivered English A2 is not in the repo and its
sample construction (n=19 / n=1120 / n=211 / n=909) could not be reproduced
from the saved parquet artifacts. To redraw the figure without inventing any
number, we digitize the delivered PNG itself: axes are calibrated from tick
marks, curves are tracked per column by colour continuity, and legend line
samples are removed via connected-component analysis before tracking.

Output:
  04_Code_and_Data/figures_public/A2_digitized_source.json   (extracted data)
  04_Code_and_Data/figures_public/A2_digitize_overlay.png    (QC overlay)

Read-only with respect to 02_Figures / 03_Supplementary.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "03_Supplementary" / "Appendix_Figure_A2.png"
OUTDIR = ROOT / "figures_public"
OUT = OUTDIR / "A2_digitized_source.json"
QC = OUTDIR / "A2_digitize_overlay.png"
if not OUTDIR.exists():
    OUTDIR.mkdir()

im_img = Image.open(SRC).convert("RGB")
im = np.array(im_img).astype(int)
H, W, _ = im.shape

# ---------------------------------------------------------------- calibration
# Axes frames: a cols 185-726 | b cols 884-1425 | c cols 1583-2124; rows 228/666
# x tick mark centroids: b [907,1091,1275] -> 0/2000/4000 d; c [1606,1790,1974]
# y tick mark centroids: rows [275,349,424,498,572,647]
#   panel a: 5000..0 days | panels b,c: 1.0..0.0
ROW_TOP, ROW_BOT = 228.0, 666.0
def px2days_b(px): return (px - 907.0) * 2000.0 / 184.0
def px2days_c(px): return (px - 1606.0) * 2000.0 / 184.0
def py2surv(py):   return (ROW_BOT - py) / (ROW_BOT - ROW_TOP)
def py2days_a(py): return (647.0 - py) * 1000.0 / 74.4

def color_mask(img, rgb, tol=50):
    return (np.abs(img - np.array(rgb)).max(axis=2) <= tol)

def remove_legend_samples(mask, region, max_size=600):
    """Zero out small same-colour components inside a legend region."""
    lab, n = ndimage.label(mask)
    out = mask.copy()
    y0, y1, x0, x1 = region
    for i in range(1, n + 1):
        comp = lab == i
        size = comp.sum()
        if size > max_size:
            continue
        cy, cx = ndimage.center_of_mass(comp)
        if y0 <= cy <= y1 and x0 <= cx <= x1:
            out[comp] = False
    return out

def track_curve(mask, col_start, col_end, x2days, y2val, name="", snap_n=None):
    """Per-column interval-distance tracking with edge snapping (rows -> y2val)."""
    xs, levels, prev = [], [], None
    for c in range(col_start, col_end):
        rows = np.where(mask[:, c - col_start])[0]
        if rows.size == 0:
            if prev is not None:
                xs.append(c); levels.append(prev)   # bridge occlusions
            continue
        clusters, cur = [], [rows[0]]
        for r in rows[1:]:
            if r - cur[-1] <= 2:
                cur.append(r)
            else:
                clusters.append(cur); cur = [r]
        clusters.append(cur)
        if prev is None:
            sel = clusters[0]
        else:
            # distance from prev to each cluster interval; snap to nearest edge
            def d(cl):
                if cl[0] >= prev: return cl[0] - prev, cl[0]
                if cl[-1] <= prev: return prev - cl[-1], cl[-1]
                return 0.0, prev
            _, snap = min((d(cl) for cl in clusters), key=lambda t: t[0])
            sel = [snap]
        prev = float(sel[0])
        xs.append(c); levels.append(prev)
    xs = np.array(xs, float); levels = y2val(np.array(levels, float))
    # terminal drop: if the last real column's lowest pixel sits well below the
    # tracked level, the curve ends with a vertical drop there (KM -> 0);
    # everything after that column is empty bridging and gets truncated.
    real = [i for i, c in enumerate(xs) if mask[:, int(round(c)) - col_start].any()]
    last_real = real[-1] if real else len(xs) - 1
    col_lr = int(round(xs[last_real])) - col_start
    rows_lr = np.where(mask[:, col_lr])[0]
    if rows_lr.size:
        level_lr = y2val(float(rows_lr.max()))          # bottom of last column
        if levels[last_real] - level_lr > 0.04:          # a real vertical drop
            xs = xs[:last_real + 1]; levels = levels[:last_real + 1]
            xs = np.append(xs, xs[-1]); levels = np.append(levels, level_lr)
        else:
            xs = xs[:last_real + 1]; levels = levels[:last_real + 1]
    # compress to steps: keep first point, then points where level changes
    keep = [0]
    for i in range(1, len(xs)):
        if abs(levels[i] - levels[keep[-1]]) > 0.006:
            keep.append(i)
    keep.append(len(xs) - 1)
    t = [max(0.0, x2days(xs[i])) for i in keep]
    s = [round(float(levels[i]), 4) for i in keep]
    if snap_n:                         # all-events KM: S must be k/n
        s = [round(round(v * snap_n) / snap_n, 4)
             if abs(v * snap_n - round(v * snap_n)) < 0.3 else v for v in s]
    if s[0] >= 0.98:                   # visible 1.0 plateau (line centre overlaps
        s[0] = 1.0                     # the top spine; snap to exact 1.0)
        t[0] = 0.0
    elif s[0] >= 0.9:                  # curve emerges from under the spine:
        t.insert(0, 0.0); s.insert(0, 1.0)
    if s[-1] > 0.02:                   # terminal event: drop to 0
        t.append(t[-1]); s.append(0.0)
    print(f"  {name}: {len(t)} steps, t=[{t[0]}, {t[-1]}], S range [{min(s)}, {max(s)}]")
    return t, s

out = {}

# ---------------------------------------------------------------- panel (a)
pa = im[229:665, 186:725]
pa_off_y, pa_off_x = 229, 186
dark = (pa.sum(axis=2) < 260)

def hlines_in(colslice, rows):
    """rows (local) with a long horizontal dark run inside colslice."""
    out_rows = []
    for r in rows:
        if dark[r, colslice].sum() > 0.7 * (colslice.stop - colslice.start):
            out_rows.append(r)
    return out_rows

def box_stats(cx_full):
    cx = cx_full - pa_off_x
    half = 52
    cs = slice(cx - half, cx + half + 1)
    # horizontal borders: exclude the plot spines (first/last rows of the panel)
    inner = range(8, pa.shape[0] - 8)
    borders = hlines_in(cs, inner)
    fill = color_mask(pa, (162, 213, 242), 40) | color_mask(pa, (231, 166, 115), 40)
    fill_rows = np.where(fill[:, cx - 20:cx + 21].sum(axis=1) > 30)[0]
    ftop, fbot = fill_rows.min(), fill_rows.max()
    top = max([r for r in borders if r <= ftop + 3], default=ftop)
    bot = min([r for r in borders if r >= fbot - 3], default=fbot)
    # median: dark border strictly between top and bot
    mids = [r for r in borders if top + 4 < r < bot - 4]
    med = int(np.median(mids)) if mids else (top + bot) // 2
    # whiskers: walk outward from the box along the centre column while dark
    vcol = dark[:, cx - 1:cx + 2].any(axis=1)
    r = top - 2
    while r > 0 and (vcol[r] or vcol[r - 1]):
        r -= 1
    w_hi = r
    r = bot + 2
    while r < pa.shape[0] - 1 and (vcol[r] or vcol[r + 1]):
        r += 1
    w_lo = r
    return {"whisker_hi": round(py2days_a(w_hi + pa_off_y), 1),
            "q3":  round(py2days_a(top + pa_off_y), 1),
            "median": round(py2days_a(med + pa_off_y), 1),
            "q1":  round(py2days_a(bot + pa_off_y), 1),
            "whisker_lo": round(py2days_a(w_lo + pa_off_y), 1),
            "_rows": [int(w_hi), int(top), int(med), int(bot), int(w_lo)]}

out["panel_a"] = {"labels": ["Single-holder", "Multi-source"],
                  "single": box_stats(320), "multi": box_stats(591)}
# outlier of Single-holder: dark ring above its whisker, near cols 300-340
ring_region = dark[:out["panel_a"]["single"]["_rows"][0] - 3,
                   300 - pa_off_x - 25:320 - pa_off_x + 26]
rr = np.where(ring_region.any(axis=1))[0]
if rr.size:
    out["panel_a"]["single_outlier"] = round(py2days_a(rr.mean() + 1 + pa_off_y), 1)
for k in ("single", "multi"):
    out["panel_a"][k].pop("_rows")

def detect_markers(mask, col_start, col_end, x2days, y2val):
    """Markers survive a 7x7 erosion that removes the ~6 px curve line."""
    sub = mask[:, col_start:col_end]
    core = ndimage.binary_erosion(sub, structure=np.ones((7, 7)))
    lab, n = ndimage.label(core)
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 3:
            continue
        ys, xs = np.where(comp)
        if ys.max() - ys.min() > 24 or xs.max() - xs.min() > 24:
            continue                       # not a compact marker blob
        c = xs.mean() + col_start
        r = ys.mean()
        out.append([round(x2days(c), 1), round(float(y2val(r)), 4)])
    out.sort(key=lambda p: p[0])
    return out

# ---------------------------------------------------------------- panels (b), (c)
BLUE, ORANGE, GREEN = (0, 114, 178), (213, 94, 0), (0, 158, 115)
def curve(rgb, c0, c1, x2f, legend_region, name, snap_n=None, markers=False):
    m = remove_legend_samples(color_mask(im, rgb), legend_region)
    d = dict(zip(["t", "S"], track_curve(m[:, c0:c1], c0, c1, x2f, py2surv,
                                         name, snap_n)))
    if markers:
        mk = detect_markers(m, c0, c1, x2f, py2surv)
        d["markers"] = mk
        print(f"    {len(mk)} markers")
    return d

print("tracking curves:")
LEG_B = (560, 700, 880, 1010)
LEG_C = (250, 430, 1680, 1800)   # legend line samples only; real curves' big
                                 # connected component exceeds max_size
out["panel_b"] = {
    "Single-holder (n=19)":  curve(BLUE,   886, 1424, px2days_b, LEG_B, "b blue", 19),
    "Multi-source (n=1120)": curve(ORANGE, 886, 1424, px2days_b, LEG_B, "b orange"),
}
out["panel_c"] = {
    "1 holder (n=19)":       curve(BLUE,   1585, 2123, px2days_c, LEG_C, "c blue", 19, markers=True),
    "2-3 holders (n=211)":   curve(ORANGE, 1585, 2123, px2days_c, LEG_C, "c orange", markers=True),
    ">=4 holders (n=909)":   curve(GREEN,  1585, 2123, px2days_c, LEG_C, "c green", markers=True),
}

OUT.write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- QC overlay
ov = im_img.copy()
dr = ImageDraw.Draw(ov)
for pan, cols, xf, yf in (("panel_b", 886, px2days_b, py2surv),
                          ("panel_c", 1585, px2days_c, py2surv)):
    for lab, d in out[pan].items():
        pts = [(cols + t / 2000.0 * 184.0, ROW_BOT - s * (ROW_BOT - ROW_TOP))
               for t, s in zip(d["t"], d["S"])]
        dr.line(pts, fill=(255, 0, 255), width=1)
# panel a: mark five-number rows
for key, cx in (("single", 320), ("multi", 591)):
    st = out["panel_a"][key]
    for val, col in ((st["whisker_hi"], (255,0,255)), (st["q3"], (255,0,0)),
                     (st["median"], (255,0,0)), (st["q1"], (255,0,0)),
                     (st["whisker_lo"], (255,0,255))):
        r = 647.0 - val * 74.4 / 1000.0
        dr.line([(cx - 40, r), (cx + 40, r)], fill=col, width=1)
if "single_outlier" in out["panel_a"]:
    r = 647.0 - out["panel_a"]["single_outlier"] * 74.4 / 1000.0
    dr.ellipse([320 - 6, r - 6, 320 + 6, r + 6], outline=(255, 0, 255), width=2)
ov.save(QC)
print("wrote", OUT)
print("wrote", QC)
