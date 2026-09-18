# -*- coding: utf-8 -*-
# LEGACY bubble-chart draft (superseded); kept for provenance only — do not use for the manuscript figure.
"""
fig6 regeneration (manuscript Figure 4): chapter-panel direct-specification version
- Composition/color scheme/annotation style consistent with the legacy version (bubble = number of
  shortage units, color = TBD share, x = annual death burden on a log axis, y = AAR gradient)
- Data sources: results/table3_chapter_panel.xlsx ChapterPanel_gradient (2003-2016 OLS primary
  specification, consistent with the main Table 3 column)
  + artifacts/h3_results.json equity_rank_table (shortage-exposure side: n_units / tbd_share)
- The burden column switches to the chapter-panel full-specification annual average deaths (legacy
  values came from an incomplete sub-row sum, e.g. chapters E/G/K missing 35-128%)
- Chapter F displays the full-window OLS -0.596 (window-specification adjudication handled in the
  robustness review; re-generate if it changes after being locked)
- Writes to figures/fig6_bubble_legacy.png (legacy output; deliberately renamed from
  fig6_equity_summary.png so it cannot overwrite the manuscript figure)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG, RES, ART = ROOT / "figures", ROOT / "results", ROOT / "artifacts"

for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# Data
grad = pd.read_excel(RES / "table3_chapter_panel.xlsx", sheet_name="ChapterPanel_gradient")
eq = pd.DataFrame(json.loads((ART / "h3_results.json").read_text())["equity_rank_table"])
eq = eq.drop(columns=[c for c in eq.columns if c.startswith("slope")])   # drop all legacy proxy gradient columns
d = eq.merge(grad[["chapter", "slope_aar_1sd", "slope_aar_p", "annual_deaths_avg"]]
             .rename(columns={"annual_deaths_avg": "annual_deaths_panel"}),
             on="chapter", how="inner")
assert len(d) == 11, f"expected 11 rows after join, got {len(d)}"
d = d.dropna(subset=["slope_aar_1sd"])
d["annual_deaths"] = d["annual_deaths_panel"]   # chapter-panel full-specification burden (the legacy sub-row sum misses 35-128% for E/G/K/N)

# Specification lock (final ruling, 2026-09-18): Chapter F (Psychiatry) follows
# manuscript Table 3's main column = the pooled full-window 2003-2016 OLS slope
# -0.596 (p = 0.547), flagged with a dagger for the time-varying structure
# ("emerged around 2015"). No window substitution is applied: the legacy
# narrative value -5.002 (D76 2017-2019, p = 0.0004) is a robustness figure of
# merit only and is never plotted.
fmask = d["chapter"] == "F01-F99"

fig, ax = plt.subplots(figsize=(11.5, 7))
sizes = 60 + 900 * (d["n_shortage_units"] / d["n_shortage_units"].max())
sc = ax.scatter(d["annual_deaths"], d["slope_aar_1sd"], s=sizes,
                c=d["tbd_share"], cmap="RdYlBu_r", edgecolor="k", linewidth=0.6, alpha=0.85)
for _, r in d.iterrows():
    star = "*" if r["slope_aar_p"] < 0.05 else ""
    dag = "†" if r["chapter"] == "F01-F99" else ""
    ax.annotate(f"{r['category']}\n(units={int(r['n_shortage_units'])}){star}{dag}",
                (r["annual_deaths"], r["slope_aar_1sd"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("Annual average death burden (deaths/year, chapter-panel full specification, log axis)")
ax.set_ylabel("SVI gradient: age-adjusted mortality (AAR) difference per +1-SD SVI")
ax.set_title("fig6 | Shortage exposure - mortality burden - state SVI gradient ternary summary (11 directly mapped categories, chapter-panel direct AAR specification 2003-2016, zero suppression, descriptive)\n"
             "Bubble size = number of shortage units; color = TBD share (signal of permanent exit); * = p<0.05; positive gradient = burden concentrated in more vulnerable states; "
             "†Psychiatry shows the pooled 2003-2016 slope -0.596 (time-varying gradient, 'emerged around 2015', see the Table 3 footnote)", fontsize=10)
fig.colorbar(sc, label="TBD share")
fig.tight_layout()
out = FIG / "fig6_bubble_legacy.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote (legacy filename; does not overwrite the manuscript figure):", out)
print(d[["category", "chapter", "n_shortage_units", "annual_deaths", "tbd_share",
         "slope_aar_1sd", "slope_aar_p"]].sort_values("slope_aar_1sd").to_string(index=False))
