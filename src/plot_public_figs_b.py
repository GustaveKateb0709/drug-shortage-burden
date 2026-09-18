"""
plot_public_figs_b.py — Redraw English final figures for the Health Policy
submission package, from saved result files (no re-estimation):

  Figure_2.png/.tif            (2160x1020) exposure / burden / gradient panel
  Appendix_Figure_A2.png/.tif  (2160x810)  stock-age composition, 3 panels
  Appendix_Figure_A6.png/.tif  (1050x900)  MNAR envelope dot plot (14 rows)
  Appendix_Figure_A7.png/.tif  (1050x840)  exploratory event study

Data sources (read-only):
  Fig2 : results/table3_chapter_panel.xlsx sheet "ChapterPanel_gradient"
         (slope_aar_1sd / slope_aar_p / annual_deaths_avg) + artifacts/
         h3_results.json "equity_rank_table" (category, n_shortage_units,
         tbd_share). Psychiatry (F01-F99) is plotted at the pooled
         2003-2016 slope -0.596 (p = 0.55), matching manuscript Table 3
         row F (flagged with a dagger; see plot comment).
  A2   : figures_public/A2_digitized_source.json - pixel-level digitization of
         the delivered PNG (its generator is not in the repo; see
         src/a2_digitize_source.py). No number is invented.
  A6   : artifacts/robustness_h3.json "H1_mnar_envelope" (14 rows).
         Fix 1: extra x headroom so the bottom I00-I99 point (x=12.958) sits
         ~7% of the canvas width away from the right spine.
         Fix 2: the duplicated rows C00-D48 / I00-I99 get window suffixes
         taken from the JSON "window" field (2003-2016 vs 2017-2019).
  A7   : artifacts/h3_results.json "event_study.coef" hi_k_* entries.

Output pixel geometry matches the delivered files exactly; files are tagged
300 dpi (each canvas therefore prints at 300 ppi) and the TIFFs are LZW.
Outputs go to 04_Code_and_Data/figures_public/ only — never 02_Figures /
03_Supplementary.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ART, RES = ROOT / "artifacts", ROOT / "results"
OUT = ROOT / "figures_public"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# Okabe-Ito palette (as delivered)
BLUE, ORANGE, LIGHTBLUE, GREEN = "#0072B2", "#D55E00", "#56B4E9", "#009E73"
LINE_BLUE = "#1f77b4"

def export(fig, stem, w_px, h_px, design_dpi):
    """Render at design_dpi (exact pixel geometry), tag files as 300 dpi."""
    fig.set_size_inches(w_px / design_dpi, h_px / design_dpi)
    tmp = OUT / f"_{stem}_tmp.png"
    fig.savefig(tmp, dpi=design_dpi)
    fig.savefig(OUT / f"{stem}.pdf", dpi=design_dpi)
    plt.close(fig)
    im = Image.open(tmp)
    assert im.size == (w_px, h_px), f"{stem}: got {im.size}"
    im.save(OUT / f"{stem}.png", dpi=(300, 300))
    im.save(OUT / f"{stem}.tif", compression="tiff_lzw", dpi=(300, 300))
    tmp.unlink()
    print(f"wrote {stem}.png/.tif  ({w_px}x{h_px}, tagged 300 dpi)")

h3 = json.load(open(ART / "h3_results.json"))

# ================================================================ Figure 2
rank = pd.DataFrame(h3["equity_rank_table"])
panel = pd.read_excel(RES / "table3_chapter_panel.xlsx",
                      sheet_name="ChapterPanel_gradient").set_index("chapter")

# Row order and labels exactly as delivered (top -> bottom)
ORDER = [
    ("Cardiovascular",             "Cardiovascular (56)"),
    ("Anti-Infective",             "Anti-infective (48)"),
    ("Oncology",                   "Oncology (45)"),
    ("Urology",                    "Urology (2)"),
    ("Pulmonary/Allergy",          "Pulmonary (6)"),
    ("Gastroenterology",           "Gastro (16)"),
    ("Endocrinology/Metabolism",   "Endocrine (46)"),
    ("Dermatology",                "Dermatology (10)"),
    ("Musculoskeletal",            "Musculoskel. (2)"),
    ("Psychiatry",                 "Psychiatry (44)"),
    ("Neurology",                  "Neurology (27)"),
]
rows = []
for cat, lab in ORDER:
    r = rank[rank["category"] == cat]
    assert len(r) == 1, f"category {cat} not unique in equity_rank_table"
    r = r.iloc[0]
    ch = r["chapter"]
    assert int(r["n_shortage_units"]) == int(lab.split("(")[1].split(")")[0])
    # All 11 chapters (incl. Psychiatry F01-F99) read the chapter-panel
    # 2003-2016 slope from the xlsx, matching manuscript Table 3 row F:
    # -0.596 (pooled), flagged with a dagger in the text ("full-window
    # pooled slope is uninterpretable"). The narrative post-2017 D76
    # value (-5.002, p=0.0004) is a robustness figure of merit, not the
    # Table 3 entry, and is therefore NOT plotted here.
    slope = float(panel.loc[ch, "slope_aar_1sd"])
    p = float(panel.loc[ch, "slope_aar_p"])
    rows.append({"lab": lab, "slope": slope, "p": p,
                 "deaths": float(panel.loc[ch, "annual_deaths_avg"]),
                 "tbd": float(r["tbd_share"])})
d2 = pd.DataFrame(rows)

# numeric-fidelity gate: the 11 plotted gradients / deaths must equal the xlsx
assert np.allclose(d2["slope"],
                   [15.669, 3.813, 2.262, 1.735, 0.921, 0.729, 0.54, 0.066,
                    -0.426, -0.596, -0.881], atol=6e-4)
assert np.allclose(d2["deaths"],
                   [821889, 68256, 588367, 64101, 244969, 93524, 104988,
                    4198, 13810, 111954, 143224], atol=0.5)

def tbd_style(tbd):                    # colour and shape bound (dual channel)
    if tbd < 0.30:
        return LIGHTBLUE, "o"
    if tbd < 0.60:
        return BLUE, "s"
    return ORANGE, "^"

fig, (axa, axb) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.85, 1]})
fig.subplots_adjust(left=0.15, right=0.985, top=0.82, bottom=0.14,
                    wspace=0.05)

y = np.arange(len(d2))[::-1]           # first row on top
for yi, r in zip(y, d2.itertuples()):
    col, mk = tbd_style(r.tbd)
    filled = r.p < 0.05
    axa.scatter(r.slope, yi, marker=mk, s=64,
                facecolor=col if filled else "white",
                edgecolor=col, linewidths=1.3, zorder=3)
axa.axvline(0, color="gray", ls="--", lw=1.1, zorder=1)
axa.set_yticks(y)
axa.set_yticklabels(d2["lab"], fontsize=13)
axa.set_xlim(-2.9, 16.8)               # left margin sized for the open Psychiatry
                                       # marker at -0.596 (pooled, Table 3 row F);
                                       # right side keeps headroom for 15.669
axa.set_ylim(-0.6, len(d2) - 0.4)
axa.set_xticks(np.arange(0, 16.1, 2.5))
axa.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axa.set_xlabel("AAR gradient per 1-SD of SVI", fontsize=14)
axa.tick_params(labelsize=12)
handles = [
    Line2D([], [], marker="o", ls="none", mfc=LIGHTBLUE, mec=LIGHTBLUE,
           ms=10, label="TBD < 30%"),
    Line2D([], [], marker="s", ls="none", mfc=BLUE, mec=BLUE,
           ms=10, label="TBD 30\u201360%"),
    Line2D([], [], marker="^", ls="none", mfc=ORANGE, mec=ORANGE,
           ms=10, label="TBD \u2265 60%"),
    Line2D([], [], marker="o", ls="none", mfc="gray", mec="gray",
           ms=10, label="p < 0.05"),
    Line2D([], [], marker="o", ls="none", mfc="white", mec="gray",
           ms=10, label="n.s."),
]
axa.legend(handles=handles, loc="lower right", fontsize=11.5,
           frameon=False, borderaxespad=0.3, labelspacing=0.4,
           handletextpad=0.5)

colors = [tbd_style(t)[0] for t in d2["tbd"]]
axb.barh(y, d2["deaths"], height=0.62, color=colors, edgecolor="white")
axb.set_xscale("log")
axb.set_xlim(3e3, 1.4e6)
axb.set_xticks([1e4, 1e5, 1e6])
axb.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$"])
axb.set_ylim(-0.6, len(d2) - 0.4)
axb.set_yticklabels([])
axb.set_xlabel("Annual deaths (2003\u20132016)", fontsize=14)
axb.tick_params(labelsize=12)

fig.text(0.5, 0.955, "Exposure, burden, and gradient by category",
         ha="center", fontsize=15)
fig.text(0.075, 0.895, "(a)", ha="center", fontsize=14, fontweight="bold")
fig.text(0.72, 0.895, "(b)", ha="center", fontsize=14, fontweight="bold")
export(fig, "Figure_2", 2160, 1020, design_dpi=200)

# ================================================================ Appendix A2
dig = json.load(open(OUT / "A2_digitized_source.json"))

fig, (axa, axb, axc) = plt.subplots(1, 3)
fig.subplots_adjust(left=0.08, right=0.985, top=0.79, bottom=0.19,
                    wspace=0.31)

pa = dig["panel_a"]
KEYMAP = {"median": "med", "whisker_lo": "whislo", "whisker_hi": "whishi"}
stats = []
for src, name, has_out in ((pa["single"], "Single-holder", True),
                           (pa["multi"], "Multi-source", False)):
    st = {KEYMAP.get(k, k): [v] for k, v in src.items()}
    st["label"] = name
    st["fliers"] = [pa["single_outlier"]] if has_out else []
    stats.append(st)
bp = axa.bxp(stats, patch_artist=True, widths=0.5,
             boxprops=dict(lw=1.2), medianprops=dict(lw=1.6, color="black"),
             whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
             flierprops=dict(marker="o", markersize=7,
                             mfc="white", mec="black", mew=1.2))
for box, fc in zip(bp["boxes"], ["#A2D5F2", "#E7A673"]):
    box.set_facecolor(fc)
axa.set_ylabel("Median stock age per market (days)", fontsize=12)
axa.tick_params(labelsize=11)

MK_OF = {"1 holder": "o", "2-3": "s", ">=": "^"}
for ax, pan in ((axb, dig["panel_b"]), (axc, dig["panel_c"])):
    for lab, d in pan.items():
        col = BLUE if lab.startswith(("Single", "1 holder")) else (
              ORANGE if lab.startswith(("Multi", "2-3")) else GREEN)
        lab_txt = lab.replace(">=", "\u2265")
        if " (n=" in lab_txt:      # two-line legend labels keep panels compact
            lab_txt = lab_txt.replace(" (n=", "\n(n=")
        ax.step(d["t"], d["S"], where="post", color=col, lw=2.0,
                label=lab_txt)
        if "markers" in d:
            mt, ms_ = zip(*d["markers"])
            mk = MK_OF[next(k for k in MK_OF if lab.startswith(k))]
            ax.plot(mt, ms_, ls="none", marker=mk, ms=5.5, color=col)
    ax.set_xlim(-250, 5630)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 2000, 4000])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Days on the current list", fontsize=12)
    ax.tick_params(labelsize=11)
axb.set_ylabel("Empirical survival", fontsize=12)
axb.legend(loc="upper right", fontsize=11.5, frameon=False,
           labelspacing=0.55, borderaxespad=0.3)
axc.legend(loc="upper right", fontsize=10.5, frameon=False,
           labelspacing=0.5, borderaxespad=0.3)

fig.text(0.5, 0.93, "Stock-age composition by market structure",
         ha="center", fontsize=14)
fig.text(0.042, 0.855, "(a)", fontsize=13, fontweight="bold")
fig.text(0.395, 0.855, "(b)", fontsize=13, fontweight="bold")
fig.text(0.735, 0.855, "(c)", fontsize=13, fontweight="bold")
export(fig, "Appendix_Figure_A2", 2160, 810, design_dpi=200)

# ================================================================ Appendix A6
rob = json.load(open(ART / "robustness_h3.json"))
env = rob["H1_mnar_envelope"]
assert len(env) == 14

seen = Counter(e["chapter"] for e in env)
dup = {ch for ch, n in seen.items() if n > 1}
labels6 = [f"{e['chapter']} ({'2003\u20132016' if 'D140' in e['window'] else '2017\u20132019'})"
           if e["chapter"] in dup else e["chapter"] for e in env]
avail6 = [e["slope_available_case"] for e in env]
med6 = [e["slope_中位(章节年均值)"] for e in env]
lo6 = [min(e["slope_下界(sup=0)"], e["slope_上界(sup=9例)"]) for e in env]
hi6 = [max(e["slope_下界(sup=0)"], e["slope_上界(sup=9例)"]) for e in env]
# numeric-fidelity gate against the delivered figure
assert avail6[-1] == 12.958 and med6[-1] == 12.958      # bottom I00-I99 row

fig, ax = plt.subplots()
yy = np.arange(len(env))[::-1]         # JSON order, first row on top
for yi, lo, hi in zip(yy, lo6, hi6):
    ax.hlines(yi, lo, hi, color=LIGHTBLUE, lw=7, zorder=1)
ax.scatter(med6, yy, marker="o", s=110, color=BLUE, zorder=3)
ax.scatter(avail6, yy, marker="D", s=90, color=ORANGE, zorder=4)
ax.axvline(0, color="gray", ls="--", lw=1.2, zorder=2)
ax.set_yticks(yy)
ax.set_yticklabels(labels6, fontsize=12)
ax.set_xlim(-6.2, 14.5)        # I00-I99 at 12.958 -> ~7% clear of right spine
ax.set_ylim(-0.7, len(env) - 0.3)
ax.set_xlabel("Slope per 1-SD of SVI (sub-row proxy era)", fontsize=13)
ax.tick_params(labelsize=12)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.legend(handles=[
    Line2D([], [], marker="D", ls="none", mfc=ORANGE, mec=ORANGE, ms=11,
           label="Available case"),
    Line2D([], [], marker="o", ls="none", mfc=BLUE, mec=BLUE, ms=11,
           label="Envelope median"),
    Line2D([], [], color=LIGHTBLUE, lw=7, label="Worst-case envelope")],
    loc="upper right", fontsize=12, frameon=False, labelspacing=0.75,
    borderaxespad=0.7)
ax.set_title("The sub-row proxy and its missing-data envelope",
             fontsize=14, pad=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
export(fig, "Appendix_Figure_A6", 1050, 900, design_dpi=150)

# ================================================================ Appendix A7
es = h3["event_study"]["coef"]
ks = sorted(int(float(k.split("_")[2])) for k in es if k.startswith("hi_k_"))
assert ks == [-5, -4, -3, -2, -1, 1, 2, 3, 4]
b7 = [es[f"hi_k_{float(k)}"]["b"] for k in ks]
p7 = [es[f"hi_k_{float(k)}"]["p"] for k in ks]
assert all(pi >= 0.05 for pi in p7)    # delivered figure has 0 filled points
assert b7 == [-4.024, -3.196, -2.886, -2.504, -2.868,
              -0.764, 0.669, -2.713, -0.221]

fig, ax = plt.subplots()
for k, bi, pi in zip(ks, b7, p7):
    filled = pi < 0.05
    ax.scatter(k, bi, s=90, facecolor=LINE_BLUE if filled else "white",
               edgecolor=LINE_BLUE, linewidths=1.5, zorder=3)
ax.axhline(0, color="gray", ls="--", lw=1.2, zorder=1)
ax.axvline(0, color="gray", ls=":", lw=1.2, zorder=1)
ax.set_xlim(-5.6, 4.6)
ax.set_ylim(-4.45, 0.95)
ax.set_xlabel("Years since category first posting", fontsize=13)
ax.set_ylabel("Coefficient (hi-SVI states)", fontsize=13)
ax.tick_params(labelsize=12)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", mfc=LINE_BLUE, mec=LINE_BLUE,
           ms=10, label="p < 0.05"),
    Line2D([], [], marker="o", ls="none", mfc="white", mec=LINE_BLUE,
           ms=10, label="n.s.")],
    loc="lower right", fontsize=12, frameon=False, borderaxespad=0.5)
ax.set_title("Exploratory event study around category first-posting years",
             fontsize=14, pad=14)
fig.tight_layout(rect=[0, 0, 1, 0.955])
export(fig, "Appendix_Figure_A7", 1050, 840, design_dpi=150)

print("DONE")


# ================================================================
# Appendix Figure A4 (English final) — dumbbell: Crude vs Age-adjusted
# ================================================================
def a4_english():
    import json as _json
    xlsx = RES / "table3_chapter_panel.xlsx"
    grad = pd.read_excel(xlsx, sheet_name="ChapterPanel_gradient")
    h3 = _json.loads((ART / "h3_results.json").read_text())
    catmap = {r["chapter"]: r["category"] for r in h3["equity_rank_table"]}
    need = {"chapter", "slope_crude_1sd", "slope_aar_1sd"}
    assert need.issubset(grad.columns), f"xlsx missing columns: {need - set(grad.columns)}"
    g = grad[grad["slope_aar_1sd"].notna() & grad["chapter"].isin(catmap)].sort_values("slope_aar_1sd").reset_index(drop=True)
    assert len(g) == 11, f"expect 11 chapters, got {len(g)}"
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    yy = np.arange(len(g))
    for i, (c, a_) in enumerate(zip(g["slope_crude_1sd"], g["slope_aar_1sd"])):
        ax.plot([c, a_], [i, i], color="#c9ced6", lw=1.4, zorder=1)
    ax.scatter(g["slope_crude_1sd"], yy, marker="s", s=46, color="#8d99ae",
               edgecolor="k", linewidth=0.5, label="Crude (population-weighted WLS)", zorder=2)
    ax.scatter(g["slope_aar_1sd"], yy, marker="o", s=52, color="#0072B2",
               edgecolor="k", linewidth=0.5, label="Age-adjusted (OLS, 2000 US std)", zorder=3)
    for i, (c, a_) in enumerate(zip(g["slope_crude_1sd"], g["slope_aar_1sd"])):
        if abs(c - a_) < 1.0:
            # tiny dumbbell: labels go outside the pair so they cannot collide
            # with each other, the markers, or the zero line
            ax.text(c - 0.55, i, f"{c:+.2f}", ha="right", va="center",
                    fontsize=7, color="#555559")
            ax.text(a_ + 0.55, i, f"{a_:+.2f}", ha="left", va="center",
                    fontsize=7, color="#00517e")
        else:
            ax.text(c, i + 0.26, f"{c:+.2f}", ha="center", va="bottom", fontsize=7, color="#555559")
            ax.text(a_, i - 0.30, f"{a_:+.2f}", ha="center", va="top", fontsize=7, color="#00517e")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(yy)
    ax.set_ylim(-0.75, len(g) - 0.25)   # headroom for the end-row value labels
    ax.set_yticklabels([catmap.get(ch, ch) for ch in g["chapter"]], fontsize=8.5)
    ax.set_xlabel("Mortality difference per 1-SD of state SVI (chapter panel 2003\u20132016, full window)")
    ax.set_title("Chapter-level gradients: crude versus age-adjusted")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_png = FIGPUB / "Appendix_Figure_A4.png"
    fig.savefig(out_png, dpi=300)
    fig.savefig(FIGPUB / (out_png.stem + ".pdf"), dpi=300)
    fig.savefig(FIGPUB / "Appendix_Figure_A4.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("A4 written:", out_png)

FIGPUB = ROOT / "figures_public"
a4_english()
