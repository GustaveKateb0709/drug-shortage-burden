"""
paper18 robustness_h3_chapter.py -- Phase 4c: full-battery audit of the chapter-panel direct gradient (upgrade gate)
Run: python3 src/robustness_h3_chapter.py

Input: data/state_mortality_chapter.parquet (chapter-level 2003-2016; A/I/N/C/E/G/J/K/F/M have 0
       suppressed cells, L has 52)
Scope (as specified by the project lead):
  C1 Baseline replication + all-chapter direct gradient (mean-over-years unweighted OLS, per-1SD SVI)
  C2 SVI permutation (shuffled across 51 states, B=1000)
  C3 State subsets: drop the 10 smallest states (population) / the 10 highest-SVI states / LOO range
  C4 Weighting comparison: unweighted / population-weighted WLS / death-weighted -- formal
     characterization of the Chapter I attenuation
  C5 Year-by-year gradients (each chapter, each single year; sign stability; Chapter A secular-drift check)
  C6 Proxy vs direct comparison table (legacy 113 sub-row specification vs MNAR envelope vs chapter direct)
  C7 Chapter A sub-row composition decomposition (each sub-row's death share x its own gradient ->
     what drives +3.813)
  C8 Verdict-gate outputs (JSON; README written manually)
Outputs: artifacts/robustness_h3_chapter.json; Phase4c_* sheets appended to table5_robustness.xlsx;
      figures/fig10_chapter_yearly.png. Does not touch any existing files.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm

for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]:
    if any(f.lower() == x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, RES, FIG = ROOT / "data", ROOT / "artifacts", ROOT / "results", ROOT / "figures"
SEED = 20260911
R = {}

chp = pd.read_parquet(DATA / "state_mortality_chapter.parquet")
sv = pd.read_parquet(DATA / "state_svi.parquet").set_index("state_fips")["rpl_themes_pw"]
print("source:", chp.source.unique(), "| years:", chp.year.min(), "-", chp.year.max())

TARGETS = ["A00-B99", "I00-I99", "N00-N98", "C00-D48", "E00-E88", "G00-G98", "J00-J98", "K00-K92",
           "F01-F99", "L00-L98", "M00-M99"]

def state_table(chapter, value="aar", weight="deaths"):
    sub = chp[(chp.cause_code == chapter) & chp[value].notna()]
    st = sub.groupby("state_fips").agg(v=(value, "mean"), pop=("population", "first"),
                                       deaths_tot=("deaths", "sum"))
    st["svi"] = st.index.map(sv)
    return st.dropna(subset=["v", "svi"])

def fit_slope(st, weighted=False, weights=None, svi_perm=None):
    svi_v = st["svi"].values if svi_perm is None else np.asarray(svi_perm, dtype=float)
    X = sm.add_constant(svi_v)
    y = st["v"].values.astype(float)
    if weights is not None:
        m = sm.WLS(y, X, weights=np.asarray(weights, dtype=float)).fit()
    elif weighted:
        m = sm.WLS(y, X, weights=st["pop"].values.astype(float)).fit()
    else:
        m = sm.OLS(y, X).fit()
    sd = float(svi_v.std(ddof=0))
    return {"slope": float(m.params[1] * sd), "se": float(m.bse[1] * sd), "p": float(m.pvalues[1]),
            "n": int(len(st)), "sd": sd}

# ================================================================ C1 Baseline replication + all-chapter direct gradient
print("\n=== C1 Baseline (mean-over-years, unweighted) ===")
base = {}
for c in TARGETS:
    st = state_table(c)
    base[c] = fit_slope(st)
    print(f"  {c}: slope/SD={base[c]['slope']:.3f} (SE {base[c]['se']:.3f}, p={base[c]['p']:.4f}) n={base[c]['n']}")
assert abs(base["A00-B99"]["slope"] - 3.813) < 0.01, "Chapter A baseline replication failed"
R["C1_baseline"] = {c: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for c, d in base.items()}

# ================================================================ C2 SVI permutation
print("\n=== C2 SVI permutation (B=1000) ===")
rng = np.random.default_rng(SEED)
perm_rows = []
for c in TARGETS:
    st = state_table(c)
    b_obs = fit_slope(st)["slope"]
    cnt = 0
    for _ in range(1000):
        mp = fit_slope(st, svi_perm=rng.permutation(st["svi"].values))
        if abs(mp["slope"]) >= abs(b_obs):
            cnt += 1
    perm_rows.append({"chapter": c, "slope_direct": round(b_obs, 3), "OLS_p": round(base[c]["p"], 4),
                      "permutation_p(two-sided)": round(cnt / 1000, 4), "B": 1000})
C2 = pd.DataFrame(perm_rows)
R["C2_perm"] = C2.to_dict("records")
print(C2.to_string(index=False))

# ================================================================ C3 State subsets
print("\n=== C3 State subsets ===")
sub_rows = []
for c in TARGETS:
    st = state_table(c)
    row = {"chapter": c, "slope_baseline": round(base[c]["slope"], 3), "p_baseline": round(base[c]["p"], 4)}
    small10 = st.sort_values("pop").index[:10]
    row["slope_drop_10_smallest"] = round(fit_slope(st.drop(small10))["slope"], 3)
    row["p_drop_10_smallest"] = round(fit_slope(st.drop(small10))["p"], 4)
    hiSVI10 = st.sort_values("svi", ascending=False).index[:10]
    row["slope_drop_10_highest_SVI"] = round(fit_slope(st.drop(hiSVI10))["slope"], 3)
    row["p_drop_10_highest_SVI"] = round(fit_slope(st.drop(hiSVI10))["p"], 4)
    slopes = [fit_slope(st.drop(index=s))["slope"] for s in st.index]
    row["LOO_min"] = round(min(slopes), 3)
    row["LOO_max"] = round(max(slopes), 3)
    row["LOO_crosses_zero"] = bool(min(slopes) < 0 < max(slopes))
    sub_rows.append(row)
C3 = pd.DataFrame(sub_rows)
R["C3_subsets"] = C3.to_dict("records")
print(C3.to_string(index=False))

# ================================================================ C4 Weighting comparison
print("\n=== C4 Weighting comparison (characterizing the Chapter I attenuation) ===")
w_rows = []
for c in TARGETS:
    st = state_table(c)
    sub = chp[(chp.cause_code == c) & chp.aar.notna()]
    # Death weighting: death-weighted average AAR within a state across years
    dw = sub.groupby("state_fips").apply(lambda g: np.average(g["aar"], weights=g["deaths"]))
    stw = pd.DataFrame({"v": dw})
    stw["svi"] = stw.index.map(sv)
    stw = stw.dropna()
    m_u = fit_slope(st)
    m_w = fit_slope(st, weighted=True)
    m_d = fit_slope(stw)
    w_rows.append({"chapter": c, "unweighted": round(m_u["slope"], 3), "p_unweighted": round(m_u["p"], 4),
                   "population_weighted_WLS": round(m_w["slope"], 3), "p_WLS": round(m_w["p"], 4),
                   "WLS_over_unweighted": round(m_w["slope"] / m_u["slope"], 2) if abs(m_u["slope"]) > 1e-9 else np.nan,
                   "death_weighted": round(m_d["slope"], 3), "p_death_weighted": round(m_d["p"], 4)})
C4 = pd.DataFrame(w_rows)
R["C4_weighting"] = C4.to_dict("records")
print(C4.to_string(index=False))

# ================================================================ C5 Year-by-year gradients
print("\n=== C5 Year by year (sign stability) ===")
yr_rows = []
YEARLY_CH = ["A00-B99", "I00-I99", "N00-N98", "F01-F99", "L00-L98", "M00-M99"]
for c in YEARLY_CH:
    slopes = {}
    for y in range(2003, 2017):
        sub = chp[(chp.cause_code == c) & (chp.year == y) & chp.aar.notna()]
        st = sub.groupby("state_fips").agg(v=("aar", "mean"), pop=("population", "first"))
        st["svi"] = st.index.map(sv)
        st = st.dropna()
        slopes[y] = fit_slope(st)
        yr_rows.append({"chapter": c, "year": y, "slope_per_1sd": round(slopes[y]["slope"], 3),
                        "p": round(slopes[y]["p"], 4), "n": slopes[y]["n"]})
    ss = [slopes[y]["slope"] for y in range(2003, 2017)]
    ps = [slopes[y]["p"] for y in range(2003, 2017)]
    neg = base[c]["slope"] < 0
    n_sign = sum(1 for s in ss if (s < 0) == neg)
    n_sig = sum(1 for p_ in ps if p_ < 0.05)
    print(f"  {c}: {n_sign}/14 years share the baseline sign, {n_sig} years singly significant; range [{min(ss):.2f},{max(ss):.2f}]")
C5 = pd.DataFrame(yr_rows)
stab = C5.groupby("chapter").apply(lambda g: pd.Series({
    "n_sign_consistent": int(sum(1 for s in g["slope_per_1sd"] if (s < 0) == (base[g.name]["slope"] < 0))),
    "n_singly_significant": int((g["p"] < 0.05).sum()),
    "slope_min": round(g["slope_per_1sd"].min(), 2), "slope_max": round(g["slope_per_1sd"].max(), 2),
    "mean_first_7_years": round(g[g.year <= 2009]["slope_per_1sd"].mean(), 3),
    "mean_last_7_years": round(g[g.year >= 2010]["slope_per_1sd"].mean(), 3)}))
R["C5_yearly"] = C5.to_dict("records")
R["C5_stability"] = stab.reset_index().to_dict("records")
print(stab.to_string())

# fig10: Chapter A/I year-by-year slopes
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, c, col in zip(axes, ["A00-B99", "I00-I99"], ["#1f6f8b", "#b23a48"]):
    g = C5[C5.chapter == c]
    ax.plot(g["year"], g["slope_per_1sd"], "o-", color=col, lw=1.8)
    sig = g["p"] < 0.05
    ax.scatter(g["year"][sig], g["slope_per_1sd"][sig], c=col, s=55, zorder=3)
    ax.scatter(g["year"][~sig], g["slope_per_1sd"][~sig], facecolors="white", edgecolors=col, s=55, zorder=3)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_title(f"{c} year-by-year direct gradient (filled = single-year p<0.05)\nbaseline (pooled) = {base[c]['slope']:.2f}", fontsize=10)
    ax.set_xlabel("Year"); ax.set_ylabel("AAR difference per +1-SD SVI")
fig.suptitle("fig10 | Annual stability of the chapter-panel direct gradients (2003-2016, zero suppression)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig10_chapter_yearly.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================================ C6 Proxy vs direct comparison
print("\n=== C6 Proxy vs direct comparison ===")
h3r = json.load(open(ART / "robustness_h3.json"))
env = {r["chapter"]: r for r in h3r["H1_mnar_envelope"] if "D140" in str(r.get("window", ""))}
old = {r["chapter"]: r for r in json.load(open(ART / "h3_results.json"))["aar_gradient"]["d140only_2003_2016"]}
cmp_rows = []
for c in TARGETS:
    e = env.get(c, {})
    o = old.get(c, {})
    cmp_rows.append({"chapter": c,
                     "legacy_proxy(113 sub-rows equal-weight)": o.get("slope_aar_per_1sd"),
                     "legacy_proxy_p": o.get("slope_aar_p"),
                     "envelope_lower_bound": e.get("slope_lower_bound(sup=0)"),
                     "envelope_upper_bound": e.get("slope_upper_bound(sup=9)"),
                     "direct(chapter panel)": round(base[c]["slope"], 3),
                     "direct_p": round(base[c]["p"], 4),
                     "direct_over_proxy_ratio": (round(base[c]["slope"] / o["slope_aar_per_1sd"], 1)
                                     if o.get("slope_aar_per_1sd") not in (None, 0) else None)})
C6 = pd.DataFrame(cmp_rows)
R["C6_proxy_vs_direct"] = C6.to_dict("records")
print(C6.to_string(index=False))

# ================================================================ C7 Chapter A sub-row composition decomposition
print("\n=== C7 Chapter A sub-row composition decomposition (what drives +3.813) ===")
aar_sub = pd.read_parquet(DATA / "state_mortality_aar.parquet")
a_sub = aar_sub[(aar_sub.source.str.contains("D140"))].copy()
a_sub["chapter_letter"] = a_sub["cause_label"].map(lambda s: str(s)[:1])
import re as _re
def a_row_chapter(label):
    s = str(label).lower()
    if "immunodeficiency" in s:      # the HIV row label contains "(HIV)", which the regex misassigns to chapter H; explicitly assign to A/B
        return "B"
    m = _re.search(r"\(([A-Z])", str(label))
    return m.group(1) if m else None
a_sub["letter"] = a_sub["cause_label"].map(a_row_chapter)
a_rows = a_sub[(a_sub["letter"].isin(["A", "B"])) & a_sub["deaths"].notna()].copy()
total_a = a_rows.groupby(["state_fips", "year"])["deaths"].sum().rename("chap_deaths")
a_rows = a_rows.join(total_a, on=["state_fips", "year"])
a_rows["share"] = a_rows["deaths"] / a_rows["chap_deaths"]
agg = a_rows.groupby("cause_label").agg(
    mean_share=("share", "mean"), tot_deaths=("deaths", "sum")).sort_values("tot_deaths", ascending=False)
dec_rows = []
for label, g_ in a_rows.groupby("cause_label"):
    st = g_.groupby("state_fips").agg(v=("share", "mean"), pop=("population", "first"))
    st["svi"] = st.index.map(sv)
    st = st.dropna()
    if len(st) < 40:
        continue
    m = fit_slope(st)
    dec_rows.append({"sub_row": label, "national_death_share": round(float(agg.loc[label, "mean_share"]), 4),
                     "share_SVI_slope(per_1sd)": round(m["slope"], 4), "p": round(m["p"], 4)})
C7 = pd.DataFrame(dec_rows).sort_values("national_death_share", ascending=False)
R["C7_a_decomp"] = C7.to_dict("records")
print(C7.head(12).to_string(index=False))

# ================================================================ C8 Verdict-gate inputs (written to JSON)
R["C8_verdict_inputs"] = {
    "A_wls_p": float(C4.loc[C4.chapter == "A00-B99", "p_WLS"].iloc[0]),
    "A_wls_slope": float(C4.loc[C4.chapter == "A00-B99", "population_weighted_WLS"].iloc[0]),
    "A_loo_range": [float(C3.loc[C3.chapter == "A00-B99", "LOO_min"].iloc[0]),
                    float(C3.loc[C3.chapter == "A00-B99", "LOO_max"].iloc[0])],
    "A_perm_p": float(C2.loc[C2.chapter == "A00-B99", "permutation_p(two-sided)"].iloc[0]),
    "A_drop_10_highest_SVI": float(C3.loc[C3.chapter == "A00-B99", "slope_drop_10_highest_SVI"].iloc[0]),
}

# ================================================================ C9 Chapter-level crude vs AAR narrowing comparison + Chapter A mean AAR
print("\n=== C9 Chapter-level crude vs AAR comparison ===")
rows9 = []
for c in TARGETS:
    sub = chp[chp.cause_code == c]
    st = sub.groupby("state_fips").agg(deaths=("deaths", "sum"), popsum=("population", "sum"), aar=("aar", "mean"))
    st["crude"] = st["deaths"] / st["popsum"] * 1e5
    st["svi"] = st.index.map(sv); st = st.dropna()
    sd = st.svi.std(ddof=0)
    mc = sm.OLS(st["crude"], sm.add_constant(st["svi"].astype(float))).fit()
    ma = sm.OLS(st["aar"], sm.add_constant(st["svi"].astype(float))).fit()
    rows9.append({"chapter": c, "crude_slope": round(float(mc.params.iloc[1] * sd), 3),
                  "crude_p": round(float(mc.pvalues.iloc[1]), 4),
                  "aar_slope": round(float(ma.params.iloc[1] * sd), 3),
                  "aar_p": round(float(ma.pvalues.iloc[1]), 4),
                  "ratio_|AAR|/|crude|": round(abs(float(ma.params.iloc[1] * sd)) / abs(float(mc.params.iloc[1] * sd)), 3),
                  "verdict": "narrowed" if abs(float(ma.params.iloc[1] * sd)) < abs(float(mc.params.iloc[1] * sd)) else "amplified"})
C9 = pd.DataFrame(rows9)
a9 = chp[chp.cause_code == "A00-B99"]
st9 = a9.groupby("state_fips").agg(aar=("aar", "mean"), pp=("population", "first"))
a_mean_pw = round(float(np.average(st9["aar"], weights=st9["pp"])), 2)
a_mean_ew = round(float(st9["aar"].mean()), 2)
R["C9_crude_vs_aar_chapter"] = {"table": C9.to_dict("records"),
                                "n_narrowed": int((C9["verdict"] == "narrowed").sum()), "n_total": len(C9),
                                "exceptions": C9.loc[C9["verdict"] == "amplified", "chapter"].tolist(),
                                "A_mean_AAR_popweighted": a_mean_pw, "A_mean_AAR_state_equal": a_mean_ew}
print(C9.to_string(index=False))
print(f"Narrowed {(C9['verdict'] == 'narrowed').sum()}/{len(C9)}; exceptions: {R['C9_crude_vs_aar_chapter']['exceptions']}")
print(f"Chapter A national mean AAR: population-weighted {a_mean_pw} / state equal-weight {a_mean_ew}")

# ================================================================ Write out
print("\n=== Write out ===")
(ART / "robustness_h3_chapter.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
existing = pd.read_excel(RES / "table5_robustness.xlsx", sheet_name=None)
with pd.ExcelWriter(RES / "table5_robustness.xlsx", engine="openpyxl") as w:
    for name, df_ in existing.items():
        df_.to_excel(w, sheet_name=name, index=False)
    pd.DataFrame(R["C1_baseline"]).T.reset_index().rename(columns={"index": "chapter"}).to_excel(
        w, sheet_name="Phase4c_C1_baseline", index=False)
    C2.to_excel(w, sheet_name="Phase4c_C2_perm", index=False)
    C3.to_excel(w, sheet_name="Phase4c_C3_subsets", index=False)
    C4.to_excel(w, sheet_name="Phase4c_C4_weighting", index=False)
    C5.to_excel(w, sheet_name="Phase4c_C5_yearly", index=False)
    stab.reset_index().to_excel(w, sheet_name="Phase4c_C5_stability", index=False)
    C6.to_excel(w, sheet_name="Phase4c_C6_proxy_vs_direct", index=False)
    C7.to_excel(w, sheet_name="Phase4c_C7_A_decomp", index=False)
    C9.to_excel(w, sheet_name="Phase4c_C9_crude_vs_aar", index=False)
print("Wrote:", ART / "robustness_h3_chapter.json", "| table5 (+8 Phase4c sheets) | fig10")
print("DONE")
