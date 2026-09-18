"""
paper18 robustness_h3.py -- Phase 4b: H3 robustness audit (does not touch estimate_h3.py / table3_equity.xlsx)
Run: python3 src/robustness_h3.py

Scope (as specified by the project lead; all computed directly):
  H3-1 MNAR boundary envelope: for each suppressed row, set aar to 0 (lower bound) / to the "AAR
       converted from 9 deaths" (upper bound) / to the chapter-year mean (median scenario),
       rebuild the chapter x state AAR and re-estimate the SVI slope -> decide whether +0.335 can flip
  H3-2 State-subset sensitivity: drop the 10 smallest states / the 10 states with the highest
       suppression rates / leave-one-out over all 51 states
  H3-3 SVI order permutation test: shuffle SVI labels within a chapter, B=1000; permutation quantile
       of the true slope
  H3-4 Weighting comparison: unweighted (primary specification) / population-weighted WLS / complete-state sample
  H3-5 F/L/M year-by-year consistency + Neurology data-window clarification (the Chapter G main gradient
       is in fact D140 2003-2016, 14 years)
  H3-6 EquityRank 4 alternative composites (baseline product / equal-weight z / rank sum / excluding unit count):
       rank cross-checks
  H3-7 crude -> AAR "artifact verdict" chapter-by-chapter check (is narrowing direction-consistent; name exceptions)
  H3-8 2020 sensitivity vs main-window consistency

Outputs: artifacts/robustness_h3.json; appends H3_* sheets to results/table5_robustness.xlsx
      (original sheets preserved); figures/fig9_mnar_envelope.png; H3 verdict appended to
      results/robustness_README.md (README updated manually).
CPU only, <1GB.
"""
import json
import re
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

# ================================================================ 0. Baseline replication (chapter mapping and AAR aggregation = estimate_h3.py 7b)
print("=== 0. Baseline replication ===")
aar_df = pd.read_parquet(DATA / "state_mortality_aar.parquet")
svi = pd.read_parquet(DATA / "state_svi.parquet")
SV = svi.set_index("state_fips")["rpl_themes_pw"]

CHAPTER = {"A": "A00-B99", "B": "A00-B99", "C": "C00-D48", "D": None,
           "E": "E00-E88", "F": "F01-F99", "G": "G00-G98", "H": "H00-H93",
           "I": "I00-I99", "J": "J00-J98", "K": "K00-K92", "L": "L00-L98",
           "M": "M00-M99", "N": "N00-N98", "O": "O00-O99", "P": "P00-P96",
           "Q": "Q00-Q99", "R": "R00-R99", "U": None, "V": "V01-Y89",
           "W": "V01-Y89", "X": "V01-Y89", "Y": "V01-Y89"}

def d140_chapter(label):
    s = str(label)
    if "Residual" in s:
        return "Residual"
    m = re.search(r"\(([A-Z][0-9]{2})", s)
    if not m:
        if "*U" in s:
            return "V01-Y89"
        return None
    letter, num = m.group(1)[0], int(m.group(1)[1:])
    if letter == "C":
        return "C00-D48"
    if letter == "D":
        return "D50-D89" if num >= 50 else "C00-D48"
    return CHAPTER.get(letter)

da = aar_df.copy()
d140 = da[da["source"].str.contains("D140")].copy()
d76 = da[da["source"].str.contains("D76")].copy()
d140["chapter"] = d140["cause_label"].map(d140_chapter)
d140.loc[d140["cause_code"] == "GR113-111", "chapter"] = "Residual"
d76["chapter"] = d76["cause_code"].map(lambda c: {"H00-H57": "H00-H93", "H60-H93": "H00-H93"}.get(c, c))

CHAP140 = ["A00-B99", "C00-D48", "D50-D89", "E00-E88", "G00-G98", "I00-I99", "J00-J98", "K00-K92", "N00-N98"]
FLM = ["F01-F99", "L00-L98", "M00-M99"]

def slope_of(st, weighted=False, svi_perm=None):
    """st: DataFrame indexed by state with aar & svi (state level, years already pooled). Returns (model, sd)."""
    svi_v = st["svi"] if svi_perm is None else svi_perm
    X = sm.add_constant(pd.Series(np.asarray(svi_v, dtype=float), index=st.index))
    y = st["aar"].astype(float)
    if weighted:
        m = sm.WLS(y, X, weights=st["pop"]).fit()
    else:
        m = sm.OLS(y, X).fit()
    sd = float(np.asarray(svi_v, dtype=float).std(ddof=0))
    return m, sd

def state_table(rows_df, y0, y1, chapter, value_col="aar"):
    """Row-level -> state-level table (consistent with the baseline: equal-weight average of all row cells
    within a state, years pooled). Rows with NaN value_col are dropped."""
    sub = rows_df[(rows_df["year"] >= y0) & (rows_df["year"] <= y1) & (rows_df["chapter"] == chapter)].copy()
    g = sub.groupby(["state_fips", "year"]).agg(v=(value_col, "mean"), n_avail=(value_col, "count"),
                                                n_rows=(value_col, "size"), pop=("population", "first")).reset_index()
    st = g.groupby("state_fips").agg(aar=("v", "mean"), n_avail=("n_avail", "sum"), n_rows=("n_rows", "sum"),
                                     pop=("pop", "first"))
    st["svi"] = st.index.map(SV)
    return st.dropna(subset=["aar", "svi"]), sub

def aar_slope(rows_df, y0, y1, chapter, weighted=False, svi_perm=None):
    st, _ = state_table(rows_df, y0, y1, chapter)
    m, sd = slope_of(st, weighted, svi_perm)
    return {"slope_per_1sd": float(m.params.iloc[1] * sd), "se": float(m.bse.iloc[1] * sd),
            "p": float(m.pvalues.iloc[1]), "n_states": int(len(st))}

# Baseline anchor replication
anchor = {}
anchor["A00-B99_D140only"] = aar_slope(d140, 2003, 2016, "A00-B99")
anchor["G00-G98_D140only"] = aar_slope(d140, 2003, 2016, "G00-G98")
for ch in FLM:
    anchor[f"{ch}_D76_17_19"] = aar_slope(d76, 2017, 2019, ch)
R["H0_anchor"] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                  for k, v in anchor.items()}
for k, v in anchor.items():
    print(f"  anchor {k}: slope/SD={v['slope_per_1sd']:.3f} p={v['p']:.4f} n={v['n_states']}")
assert abs(anchor["A00-B99_D140only"]["slope_per_1sd"] - 0.335) < 0.005, "Chapter A baseline replication failed"

# ================================================================ H3-1 MNAR boundary envelope
print("\n=== H3-1 MNAR boundary envelope ===")

def build_envelope_rows(df, y0, y1):
    """For row-level data in [y0,y1], apply three imputations: lo=0 / hi=AAR converted from 9 deaths /
    mid=chapter-year national mean."""
    d = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()
    d["aar_lo"] = d["aar"]
    d["aar_hi"] = d["aar"]
    d["aar_mid"] = d["aar"]
    sup = d["aar_suppressed"].astype(bool)
    if not sup.any():
        return d
    # Conversion ratio: state-year-chapter available rows sum(aar)/sum(crude); fallback chapter-year national;
    # final fallback 1.0
    avail = d[~sup & d["aar"].notna() & d["crude_rate"].notna()]
    ratio_sy = avail.groupby(["state_fips", "year", "chapter"]).apply(
        lambda g: g["aar"].sum() / g["crude_rate"].sum() if g["crude_rate"].sum() > 0 else np.nan)
    ratio_ch = avail.groupby(["chapter", "year"]).apply(
        lambda g: g["aar"].sum() / g["crude_rate"].sum() if g["crude_rate"].sum() > 0 else np.nan)
    ch_mean = avail.groupby(["chapter", "year"])["aar"].mean()
    ratio_sy = {k: float(v) for k, v in ratio_sy.items() if pd.notna(v)}
    ratio_ch = {k: float(v) for k, v in ratio_ch.items() if pd.notna(v)}
    ch_mean = {k: float(v) for k, v in ch_mean.items() if pd.notna(v)}
    hi_vals, mid_vals = [], []
    for _, r in d[sup].iterrows():
        pop = r["population"]
        rat = ratio_sy.get((r["state_fips"], r["year"], r["chapter"]), np.nan)
        if not np.isfinite(rat):
            rat = ratio_ch.get((r["chapter"], r["year"]), 1.0)
        if not np.isfinite(rat):
            rat = 1.0
        hi_vals.append(9.0 / pop * 1e5 * rat)
        mid_vals.append(ch_mean.get((r["chapter"], r["year"]), np.nan))
    d.loc[sup, "aar_lo"] = 0.0
    d.loc[sup, "aar_hi"] = hi_vals
    d.loc[sup, "aar_mid"] = mid_vals
    return d

env_rows = []
d140_env = build_envelope_rows(d140, 2003, 2016)
for ch in CHAP140:
    row = {"chapter": ch, "window": "2003-2016 (D140)"}
    for tag, col in [("available_case", "aar"), ("lower_bound(sup=0)", "aar_lo"),
                     ("upper_bound(sup=9)", "aar_hi"), ("median(chapter-year mean)", "aar_mid")]:
        s = aar_slope(d140_env.assign(aar=d140_env[col]), 2003, 2016, ch)
        row[f"slope_{tag}"] = round(s["slope_per_1sd"], 3)
        row[f"p_{tag}"] = round(s["p"], 4)
    sub = d140_env[(d140_env["chapter"] == ch)]
    row["suppressed_share"] = round(float(sub["aar_suppressed"].astype(bool).mean()), 4)
    lo_, hi_ = sorted([row["slope_lower_bound(sup=0)"], row["slope_upper_bound(sup=9)"]])
    row["envelope_contains_0(sign_flippable)"] = bool(lo_ < 0 < hi_)
    row["available_case_sign_agrees_with_envelope"] = bool(np.sign(row["slope_available_case"]) == np.sign((lo_ + hi_) / 2))
    env_rows.append(row)
d76_env = build_envelope_rows(d76, 2017, 2019)
for ch in FLM + ["C00-D48", "I00-I99"]:
    row = {"chapter": ch, "window": "2017-2019 (D76)"}
    for tag, col in [("available_case", "aar"), ("lower_bound(sup=0)", "aar_lo"),
                     ("upper_bound(sup=9)", "aar_hi"), ("median(chapter-year mean)", "aar_mid")]:
        s = aar_slope(d76_env.assign(aar=d76_env[col]), 2017, 2019, ch)
        row[f"slope_{tag}"] = round(s["slope_per_1sd"], 3)
        row[f"p_{tag}"] = round(s["p"], 4)
    sub = d76_env[(d76_env["chapter"] == ch)]
    row["suppressed_share"] = round(float(sub["aar_suppressed"].astype(bool).mean()), 4)
    lo_, hi_ = sorted([row["slope_lower_bound(sup=0)"], row["slope_upper_bound(sup=9)"]])
    row["envelope_contains_0(sign_flippable)"] = bool(lo_ < 0 < hi_)
    row["available_case_sign_agrees_with_envelope"] = bool(np.sign(row["slope_available_case"]) == np.sign((lo_ + hi_) / 2))
    env_rows.append(row)
H1 = pd.DataFrame(env_rows)
R["H1_mnar_envelope"] = H1.to_dict("records")
print(H1[["chapter", "window", "suppressed_share", "slope_available_case", "slope_lower_bound(sup=0)",
          "slope_upper_bound(sup=9)", "slope_median(chapter-year mean)", "envelope_contains_0(sign_flippable)"]].to_string(index=False))

# fig9 envelope figure
fig, ax = plt.subplots(figsize=(11, 6))
H1p = H1[~H1["chapter"].isin(["C00-D48"])].copy()   # C/D48 D76 chapter scale too large; annotated separately
H1p = H1p[H1p["window"].str.contains("D140") | H1p["chapter"].isin(FLM)]
yy = np.arange(len(H1p))
for i, (_, r) in enumerate(H1p.iterrows()):
    lo, hi, obs = r["slope_lower_bound(sup=0)"], r["slope_upper_bound(sup=9)"], r["slope_available_case"]
    ax.plot([lo, hi], [i, i], color="#bdbdbd", lw=6, alpha=.6, zorder=1)
    ax.scatter([obs], [i], c="#b23a48" if r["envelope_contains_0(sign_flippable)"] else "#1f6f8b", s=52, zorder=3)
    ax.text(hi, i, f"  sup={r['suppressed_share']:.0%}", va="center", fontsize=8, color="#555")
ax.set_yticks(yy)
ax.set_yticklabels([f"{r.chapter}\n({r.window[:9]})" for _, r in H1p.iterrows()], fontsize=8)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("AAR difference per +1-SD SVI (per 100,000): gray band = MNAR boundary envelope [sup=0, sup=9], dot = available-case")
ax.set_title("fig9 | MNAR boundary envelope for H3 gradients (red dot = sign flippable within the envelope)\n"
             "Upper bound: each suppressed cell converted from 9 deaths using the state-chapter AAR/crude conversion ratio", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig9_mnar_envelope.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================================ H3-2 State-subset sensitivity
print("\n=== H3-2 State-subset sensitivity ===")
TARGETS = [("d140", "A00-B99", 2003, 2016), ("d140", "G00-G98", 2003, 2016),
           ("d76", "F01-F99", 2017, 2019), ("d76", "M00-M99", 2017, 2019), ("d140", "N00-N98", 2003, 2016)]
sub_rows = []
for src, ch, y0, y1 in TARGETS:
    df = d140 if src == "d140" else d76
    st, subrows = state_table(df, y0, y1, ch)
    st = st.copy()
    st["sup_share"] = 1 - st["n_avail"] / st["n_rows"]
    base = aar_slope(df, y0, y1, ch)
    row = {"chapter": ch, "window": f"{y0}-{y1}", "slope_baseline": round(base["slope_per_1sd"], 3),
           "p_baseline": round(base["p"], 4)}
    # Drop the 10 smallest states (by population)
    small10 = st.sort_values("pop").index[:10]
    st2 = st.drop(small10)
    m, sd = slope_of(st2)
    row["slope_drop_10_smallest"] = round(float(m.params.iloc[1] * sd), 3); row["p_drop_10_smallest"] = round(float(m.pvalues.iloc[1]), 4)
    # Drop the 10 states with the highest suppression rates
    hi10 = st.sort_values("sup_share", ascending=False).index[:10]
    st3 = st.drop(hi10)
    m, sd = slope_of(st3)
    row["slope_drop_10_most_suppressed"] = round(float(m.params.iloc[1] * sd), 3); row["p_drop_10_most_suppressed"] = round(float(m.pvalues.iloc[1]), 4)
    # leave-one-out
    slopes = []
    for s in st.index:
        m, sd = slope_of(st.drop(index=s))
        slopes.append(float(m.params.iloc[1] * sd))
    row["slope_LOO_min"] = round(min(slopes), 3); row["slope_LOO_max"] = round(max(slopes), 3)
    row["LOO_crosses_zero"] = bool(min(slopes) < 0 < max(slopes))
    sub_rows.append(row)
H2 = pd.DataFrame(sub_rows)
R["H2_state_subsets"] = H2.to_dict("records")
print(H2.to_string(index=False))

# ================================================================ H3-3 SVI permutation test
print("\n=== H3-3 SVI permutation test (B=1000) ===")
rng = np.random.default_rng(SEED)
perm_rows = []
for src, ch, y0, y1 in TARGETS:
    df = d140 if src == "d140" else d76
    st, _ = state_table(df, y0, y1, ch)
    m, sd = slope_of(st)
    b_obs = float(m.params.iloc[1])
    cnt = 0
    for _ in range(1000):
        perm_svi = pd.Series(rng.permutation(st["svi"].values), index=st.index)
        mp, _ = slope_of(st, svi_perm=perm_svi)
        if abs(float(mp.params.iloc[1])) >= abs(b_obs):
            cnt += 1
    perm_rows.append({"chapter": ch, "window": f"{y0}-{y1}", "slope": round(b_obs * sd, 3),
                      "OLS_p": round(float(m.pvalues.iloc[1]), 4),
                      "permutation_p(two-sided)": round(cnt / 1000, 4), "B": 1000})
H3 = pd.DataFrame(perm_rows)
R["H3_perm_svi"] = H3.to_dict("records")
print(H3.to_string(index=False))

# ================================================================ H3-4 Weighting comparison
print("\n=== H3-4 Weighting comparison ===")
w_rows = []
for src, ch, y0, y1 in TARGETS:
    df = d140 if src == "d140" else d76
    st, _ = state_table(df, y0, y1, ch)
    m_u, sd = slope_of(st)
    m_w, _ = slope_of(st, weighted=True)
    cc = st[st["n_avail"] == st["n_rows"]]
    if len(cc) >= 2:
        m_c, sd_c = slope_of(cc)
        complete = {"slope": round(float(m_c.params.iloc[1] * sd_c), 3), "p": round(float(m_c.pvalues.iloc[1]), 4),
                    "n": int(len(cc))}
    else:
        complete = {"slope": None, "p": None, "n": int(len(cc))}
    w_rows.append({"chapter": ch, "window": f"{y0}-{y1}",
                   "unweighted(primary)": round(float(m_u.params.iloc[1] * sd), 3),
                   "p_unweighted": round(float(m_u.pvalues.iloc[1]), 4),
                   "population_weighted_WLS": round(float(m_w.params.iloc[1] * sd), 3),
                   "p_WLS": round(float(m_w.pvalues.iloc[1]), 4),
                   "complete_states_slope": complete["slope"], "complete_states_p": complete["p"], "n_complete_states": complete["n"]})
H4 = pd.DataFrame(w_rows)
R["H4_weighting"] = H4.to_dict("records")
print(H4.to_string(index=False))

# ================================================================ H3-5 F/L/M year-by-year consistency + Neurology window clarification
print("\n=== H3-5 Year-by-year consistency ===")
yr_rows = []
for src, ch, years in [("d76", "F01-F99", [2017, 2018, 2019]), ("d76", "M00-M99", [2017, 2018, 2019]),
                       ("d76", "L00-L98", [2017, 2018, 2019]),
                       ("d140", "G00-G98", [2003, 2006, 2009, 2012, 2015])]:  # G as a 3-year-window approximation: single years + sub-intervals
    df = d140 if src == "d140" else d76
    for y in years:
        s = aar_slope(df, y, y, ch)
        yr_rows.append({"chapter": ch, "data_source": "D140" if src == "d140" else "D76", "year": y,
                        "slope_per_1sd": round(s["slope_per_1sd"], 3), "p": round(s["p"], 4), "n": s["n_states"]})
# Sub-interval slices (Chapter G main-window consistency)
for y0, y1 in [(2003, 2006), (2007, 2010), (2011, 2014), (2013, 2016)]:
    s = aar_slope(d140, y0, y1, "G00-G98")
    yr_rows.append({"chapter": "G00-G98", "data_source": "D140", "year": f"{y0}-{y1}",
                    "slope_per_1sd": round(s["slope_per_1sd"], 3), "p": round(s["p"], 4), "n": s["n_states"]})
# 2020 (COVID) vs 2017-19 for F/M/G
for ch in ["F01-F99", "M00-M99", "G00-G98"]:
    src_df = d76 if ch != "G00-G98" else d76   # 2020 available in D76 only
    s = aar_slope(d76, 2020, 2020, ch)
    yr_rows.append({"chapter": ch, "data_source": "D76", "year": 2020,
                    "slope_per_1sd": round(s["slope_per_1sd"], 3), "p": round(s["p"], 4), "n": s["n_states"]})
H5 = pd.DataFrame(yr_rows)
R["H5_yearly"] = H5.to_dict("records")
print(H5.to_string(index=False))

# ================================================================ H3-6 EquityRank alternative composites
print("\n=== H3-6 EquityRank 4 alternative composites ===")
cmap = json.load(open(ART / "category_cause_map.json"))
cat_map = {k: v["cause_group"] for k, v in cmap["mapping"].items()
           if (not v.get("review", True)) and v.get("cause_group")}
CH_OF = {cat: cmap["cause_group_reference"][g]["chapter"] for cat, g in cat_map.items()}
frame = pd.read_parquet(DATA / "shortages_linked.parquet")
frame = frame[frame["status"].isin(["Current", "To Be Discontinued"])].copy()
frame["tbd"] = (frame["status"] == "To Be Discontinued").astype(int)
frame["unit_id"] = frame.groupby(["ingredient", "form_norm", "company_name", "initial_posting_date"]).ngroup()
units = frame.drop_duplicates("unit_id").copy()
units["first_cat"] = units["therapeutic_category"].str.split("; ").str[0].fillna("Missing")
units = units[units["first_cat"].isin(cat_map)]
expo = units.groupby("first_cat").agg(n_units=("unit_id", "nunique"), tbd_share=("tbd", "mean"))

eq_rows = []
for cat, g_ in cat_map.items():
    ch = cmap["cause_group_reference"][g_]["chapter"]
    if ch == "H00-H93":
        continue
    src_aar = d76 if ch in FLM else d140
    y0, y1 = (2017, 2019) if ch in FLM else (2003, 2016)
    try:
        s = aar_slope(src_aar, y0, y1, ch)
    except Exception:
        continue
    if cat not in expo.index:
        continue
    # Annual average deaths (baseline specification: sum of crude available-case deaths / years; from
    # chapter totals in state_mortality.parquet)
    eq_rows.append({"category": cat, "chapter": ch, "n_units": int(expo.loc[cat, "n_units"]),
                    "slope_aar": s["slope_per_1sd"], "slope_p": s["p"]})
eq = pd.DataFrame(eq_rows)
# Annual average deaths (aligned with the baseline annual_deaths_avg: taken directly from h3_results.json
# to avoid specification drift)
h3j = json.load(open(ART / "h3_results.json"))
eqj = {r["category"]: r for r in h3j["equity_rank_table"] if "annual_deaths_avg" in r}
eq["annual_deaths"] = eq["category"].map(lambda c: eqj.get(c, {}).get("annual_deaths_avg"))
eq = eq.dropna(subset=["annual_deaths"])
pos = np.maximum(eq["slope_aar"], 0.0)
z = lambda x: (x - x.mean()) / x.std(ddof=0)
eq["comp_baseline_product"] = pos * eq["annual_deaths"] * eq["n_units"]
eq["comp_equal_weight_z"] = (z(pos) + z(np.log1p(eq["annual_deaths"])) + z(eq["n_units"])) / 3
eq["comp_rank_sum"] = (pos.rank() + np.log1p(eq["annual_deaths"]).rank() + eq["n_units"].rank()) / 3
eq["comp_excluding_units"] = pos * eq["annual_deaths"]
H6 = eq.sort_values("comp_baseline_product", ascending=False)[
    ["category", "chapter", "n_units", "annual_deaths", "slope_aar", "slope_p",
     "comp_baseline_product", "comp_equal_weight_z", "comp_rank_sum", "comp_excluding_units"]].copy()
for c in ["comp_baseline_product", "comp_equal_weight_z", "comp_rank_sum", "comp_excluding_units"]:
    H6[f"rank_{c}"] = H6[c].rank(ascending=False).astype(int)
R["H6_equityrank_alt"] = H6.round(4).to_dict("records")
print(H6.round(3).to_string(index=False))

# ================================================================ H3-7 crude->AAR artifact verdict check
print("\n=== H3-7 crude->AAR narrowing check ===")
cmp_rows = h3j["aar_gradient"]["crude_vs_aar_compare"]
H7 = pd.DataFrame(cmp_rows)
H7["|AAR|/|crude|"] = (H7["slope_aar_per_1sd"].abs() / H7["slope_crude_per_1sd"].abs()).round(3)
H7["verdict"] = np.where(H7["|AAR|/|crude|"] < 0.5, "substantially narrowed (age confounding dominant)",
              np.where(H7["|AAR|/|crude|"] < 1.0, "narrowed",
               np.where((H7["slope_crude_per_1sd"] * H7["slope_aar_per_1sd"]) < 0, "sign flip", "amplified / not narrowed")))
R["H7_crude_vs_aar"] = H7.to_dict("records")
print(H7[["chapter", "window", "slope_crude_per_1sd", "slope_aar_per_1sd", "|AAR|/|crude|", "verdict"]].to_string(index=False))

# ================================================================ H3-8 2020 sensitivity consistency
print("\n=== H3-8 2020 vs main window ===")
g140 = pd.DataFrame(h3j["aar_gradient"]["d140only_2003_2016"]).set_index("chapter")
g76 = pd.DataFrame(h3j["aar_gradient"]["d76_2017_2019_FLM"]).set_index("chapter")
g20 = pd.DataFrame(h3j["aar_gradient"]["sens_2020"]).set_index("chapter")
rows8 = []
for ch in sorted(set(g140.index) | set(g76.index)):
    main = g140.loc[ch] if ch in g140.index else g76.loc[ch]
    if ch not in g20.index:
        continue
    y20 = g20.loc[ch]
    rows8.append({"chapter": ch, "main_window_slope": main["slope_aar_per_1sd"], "main_window_p": main["slope_aar_p"],
                  "slope_2020": y20["slope_aar_per_1sd"], "p_2020": y20["slope_aar_p"],
                  "sign_consistent": bool(np.sign(main["slope_aar_per_1sd"]) == np.sign(y20["slope_aar_per_1sd"]))})
H8 = pd.DataFrame(rows8)
R["H8_2020_consistency"] = H8.to_dict("records")
print(H8.to_string(index=False))

# ================================================================ Write out
print("\n=== Write out ===")
(ART / "robustness_h3.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))

# Append H3 sheets to table5_robustness.xlsx (original sheets preserved)
existing = pd.read_excel(RES / "table5_robustness.xlsx", sheet_name=None)
with pd.ExcelWriter(RES / "table5_robustness.xlsx", engine="openpyxl") as w:
    for name, df_ in existing.items():
        df_.to_excel(w, sheet_name=name, index=False)
    H1.to_excel(w, sheet_name="H3_1_MNAR_envelope", index=False)
    H2.to_excel(w, sheet_name="H3_2_state_subsets", index=False)
    H3.to_excel(w, sheet_name="H3_3_perm_SVI", index=False)
    H4.to_excel(w, sheet_name="H3_4_weighting", index=False)
    H5.to_excel(w, sheet_name="H3_5_yearly_FLM_G", index=False)
    H6.round(4).to_excel(w, sheet_name="H3_6_equityrank_alt", index=False)
    H7.to_excel(w, sheet_name="H3_7_crude_vs_aar", index=False)
    H8.to_excel(w, sheet_name="H3_8_2020_sens", index=False)
print("Wrote:", ART / "robustness_h3.json", "| table5_robustness.xlsx (+8 H3 sheets) | fig9")
print("DONE")
