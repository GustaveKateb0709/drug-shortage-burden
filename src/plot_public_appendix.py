"""
plot_public_appendix.py — Redraw English appendix figures A1 / A5 / A3 from saved
result files (no re-estimation).

Outputs (PNG + LZW-compressed TIFF, 300 dpi) to 04_Code_and_Data/figures_public/:
  Appendix_Figure_A1.png/.tif  — TBD share by posting cohort (orange bars)
  Appendix_Figure_A5.png/.tif  — Yearly age-adjusted anti-infective gradients
  Appendix_Figure_A3.png/.tif  — Specification curve across 64 model variants

Data sources (read-only):
  A1: artifacts/robustness_results.json        -> E_entry_composition
  A5: artifacts/robustness_h3_chapter.json     -> C5_yearly (chapter A00-B99)
  A3: results/table5_robustness.xlsx           -> sheet "C_spec_curve" (64 rows)
Cross-checks against artifacts/robustness_results.json C_speccurve summary.

Fixes relative to the previous delivered PNGs:
  A1: single two-line x tick labels "cohort\n(n=..)" with proper line spacing
      (no overlap); in-figure title "The 2025-26 discontinuation wave".
  A5: mean-line annotations moved to empty regions (no data overlap); title
      adds "age-adjusted".
  A3: legend now explains the marker-shape channel (circle = log_holders,
      square = single_holder).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ART, RES = ROOT / "artifacts", ROOT / "results"
OUT = ROOT / "figures_public"
if not OUT.exists():
        OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

BLUE, ORANGE = "#0072B2", "#D55E00"   # Okabe-Ito
LINE_BLUE = "#1f77b4"

def export(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tif", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("wrote", OUT / f"{stem}.png", "and .tif")

# ================================================================ A1
# TBD share by posting cohort. Source: E_entry_composition.
rob = json.load(open(ART / "robustness_results.json"))
comp = rob["E_entry_composition"]           # cohort -> {Current, TBD, tbd_share}
cohorts = list(comp.keys())
shares = [comp[c]["tbd_share"] for c in cohorts]
ns = [comp[c]["Current"] + comp[c]["TBD"] for c in cohorts]

fig, ax = plt.subplots(figsize=(7.0, 5.8))
x = np.arange(len(cohorts))
ax.bar(x, shares, width=0.62, color=ORANGE, edgecolor="white")
for xi, s in zip(x, shares):
    ax.text(xi, s + 0.02, f"{s:.0%}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cohorts, ns)],
                   fontsize=10, linespacing=1.6)
ax.set_ylim(0, 1.08)
ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_ylabel("Share flagged To Be Discontinued", fontsize=11)
ax.set_title("The 2025\u201326 discontinuation wave", fontsize=13, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
export(fig, "Appendix_Figure_A1")

# ================================================================ A5
# Yearly age-adjusted anti-infective (A00-B99) gradients. Source: C5_yearly.
ch = json.load(open(ART / "robustness_h3_chapter.json"))
rows = sorted((r for r in ch["C5_yearly"] if r["chapter"] == "A00-B99"),
              key=lambda r: r["year"])
years = np.array([r["year"] for r in rows])
slopes = np.array([r["slope_per_1sd"] for r in rows])
assert len(years) == 14 and years[0] == 2003 and years[-1] == 2016
m1, m2 = slopes[:7].mean(), slopes[7:].mean()   # 4.58 / 3.05
assert abs(round(m1, 2) - 4.58) < 0.005 and abs(round(m2, 2) - 3.05) < 0.005

fig, ax = plt.subplots(figsize=(7.0, 5.6))
ax.plot(years, slopes, "-o", color=LINE_BLUE, lw=2, ms=7, zorder=3)
ax.axhline(m1, color="gray", ls=":", lw=1.4, zorder=1)
ax.axhline(m2, color="gray", ls=":", lw=1.4, zorder=1)
# Annotations in verified empty regions: left band below the early data
# (2003-07 points are all >= 4.37) and right band above the late data
# (2011-16 points are all <= 3.35). Both inside xlim/ylim.
ax.text(2005.0, m2 + 0.08, f"last-7 mean {m2:.2f}", color="dimgray",
        fontsize=10.5, ha="center", va="bottom", zorder=4)
ax.text(2013.7, m1 + 0.08, f"first-7 mean {m1:.2f}", color="dimgray",
        fontsize=10.5, ha="center", va="bottom", zorder=4)
ax.set_xlim(2002.4, 2016.6)
ax.set_ylim(2.45, 5.25)
ax.set_xticks([2003, 2006, 2009, 2012, 2015])
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("AAR gradient per 1-SD of SVI", fontsize=11)
ax.set_title("Yearly age-adjusted anti-infective gradients, 2003\u20132016",
             fontsize=13, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
export(fig, "Appendix_Figure_A5")

# ================================================================ A3
# Specification curve across 64 model variants. Source: sheet C_spec_curve.
spec = pd.read_excel(RES / "table5_robustness.xlsx", sheet_name="C_spec_curve")
assert len(spec) == 64
sc = rob["C_speccurve"]                     # cross-check vs JSON summary
assert abs(spec["coef"].min() - sc["coef_min"]) < 1e-6
assert abs(spec["coef"].max() - sc["coef_max"]) < 1e-6
assert abs((spec["p"] < 0.05).mean() - sc["share_p_lt_0.05"]) < 1e-6

Cc = spec.sort_values("coef").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7.0, 8.8))
for var, marker in [("log_holders", "o"), ("single_holder", "s")]:
    sub = Cc[Cc["变量"] == var]
    for fe, col in [(0, BLUE), (1, ORANGE)]:
        s2 = sub[sub["入榜期FE"] == fe]
        sig = s2["p"] < 0.05
        ax.scatter(s2.index, s2["coef"], marker=marker, s=46, zorder=3,
                   facecolors=np.where(sig, col, "white"), edgecolors=col,
                   linewidths=1.2)
ax.axhline(0, color="gray", ls="--", lw=0.9, zorder=1)
handles = [
    Line2D([], [], marker="o", ls="none", markerfacecolor="black",
           markeredgecolor="black", markersize=9, label="p < 0.05"),
    Line2D([], [], marker="o", ls="none", markerfacecolor="white",
           markeredgecolor="black", markersize=9, label="n.s."),
    Line2D([], [], marker="o", ls="none", markerfacecolor=BLUE,
           markeredgecolor=BLUE, markersize=9, label="No cohort FE"),
    Line2D([], [], marker="o", ls="none", markerfacecolor=ORANGE,
           markeredgecolor=ORANGE, markersize=9, label="Cohort FE"),
    Line2D([], [], marker="o", ls="none", markerfacecolor="dimgray",
           markeredgecolor="dimgray", markersize=9,
           label="Circle: log_holders"),
    Line2D([], [], marker="s", ls="none", markerfacecolor="dimgray",
           markeredgecolor="dimgray", markersize=9,
           label="Square: single_holder"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9.5, frameon=True)
ax.set_xlabel("Specification (sorted by coefficient)", fontsize=11)
ax.set_ylabel("Coefficient", fontsize=11)
ax.set_title("Specification curve across 64 model variants",
             fontsize=13, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
export(fig, "Appendix_Figure_A3")

print("DONE")
