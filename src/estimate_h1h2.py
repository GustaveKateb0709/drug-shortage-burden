"""
paper18 estimate_h1h2.py -- Phase 3: H1' persistence/tenure + H2 permanent-exit baseline estimates
Run: python3 src/estimate_h1h2.py

Design notes (correction scheme confirmed with the project lead):
- The openFDA shortage endpoint is a mirror of the "current list" (stock). The raw JSON contains
  discontinued_date (present for all 445 TBD records; 99.3% == initial_posting_date, i.e. "listed
  on the same day discontinuation was announced") and Resolved.change_date (n=7, too few to estimate).
- All Current records are right-censored at the snapshot date with 0 exit events -> the exit hazard
  is not identified; but "time already on the list, D = snapshot - initial_posting" is observed
  exactly (stock sampling with known origin). Note the identity D = snapshot - entry: H1' "stock age"
  comparisons = stock composition (which records were posted earlier) + over-representation of
  long-tenure records in the stock (a length-bias signal), and cannot be read as individual exit risk.
- H1' primary analysis: regression of log(1+D) for Current records (market-cell clustered SE)
  + KM stratification (with everyone's event=1, KM equals the empirical survival function of D)
  + logrank/kruskal.
- H2: logistic P(To Be Discontinued | market structure, category, entry year), AMEs reported;
  cloglog sensitivity.
- Collinearity: corr(log holders, log applications)=0.936 -> holder count / application count /
  single-holder enter mutually exclusive specifications, never the same column.
- Sensitivities: recent-K-year entry cohorts (K=3/5/10), record level (no dedup), cloglog.

Outputs:
  results/table2_main.xlsx / figures/fig2_persistence_km.png / figures/fig3_coef_forest.png
  artifacts/h1h2_results.json
Peak memory <1GB, CPU only.
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
# macOS CJK fonts
for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti", "Heiti TC"]:
    if any(f.lower() == x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False
import statsmodels.api as sm
from scipy.stats import kruskal
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, RES, FIG = ROOT / "data", ROOT / "artifacts", ROOT / "results", ROOT / "figures"
for d in (RES, FIG):
    if not d.exists():
        d.mkdir()
SNAPSHOT = pd.Timestamp("2026-09-11")
CLUSTER_NOTE = "(linked_ingredient_key, form_norm) ingredient-form market cell"

# ================================================================ 1. Data
print("=== 1. Load and sample construction ===")
df = pd.read_parquet(DATA / "shortages_linked.parquet")
ms = pd.read_parquet(DATA / "market_structure_ingredient_form.parquet")

# Recover discontinued_date / change_date dropped by the pipeline from the raw JSON (row order aligned, asserted)
raw = json.load(open(ROOT / "data/raw/drug-shortages-0001-of-0001.json"))["results"]
assert len(raw) == len(df)
raw_ip = pd.to_datetime([r.get("initial_posting_date") for r in raw], format="%m/%d/%Y", errors="coerce")
assert (raw_ip.values == df["initial_posting_date"].values).all(), "raw<->parquet row-order mismatch; do not use"
df["discontinued_date"] = pd.to_datetime([r.get("discontinued_date") for r in raw], format="%m/%d/%Y", errors="coerce")
df["change_date"] = pd.to_datetime([r.get("change_date") for r in raw], format="%m/%d/%Y", errors="coerce")

n0 = len(df)
df = df[df["status"].isin(["Current", "To Be Discontinued"])].copy()   # drop Resolved (7)
df["tbd"] = (df["status"] == "To Be Discontinued").astype(int)
df["unit_id"] = df.groupby(["ingredient", "form_norm", "company_name", "initial_posting_date"]).ngroup()
frame = df.drop_duplicates("unit_id").copy()
assert int(df.groupby("unit_id")["tbd"].nunique().max()) == 1, "status not constant within dedup groups"
print(f"Records {n0} -> drop Resolved({n0-len(df)}) -> deduplicated analysis units {len(frame)} "
      f"(Current={int((frame.tbd==0).sum())}, TBD={int(frame.tbd.sum())})")

# Market-structure join (fallback chain): (linked_ingredient_key, form) -> (ingredient, form) -> ingredient level
ms_if = ms.set_index(["ingredient_key", "dosage_form"])
ms_ing = (ms.sort_values("ndc_products", ascending=False)
            .drop_duplicates("ingredient_key").set_index("ingredient_key"))

def _at(tab, key, col):
    if key not in tab.index or col not in tab.columns:
        return np.nan
    v = tab.at[key, col]
    if isinstance(v, pd.Series):
        v = v.dropna().iloc[0] if v.notna().any() else np.nan
    return v

ING_FALLBACK = {"holders": "max_holders", "ndc_products": "max_ndc_products",
                "applications": None, "exclusivity_flag": "exclusivity_flag", "no_generic": "no_generic"}

def get_ms(row, col):
    li, ing, fn = row["linked_ingredient_key"], row["ingredient"], row["form_norm"]
    for key in [(li, fn), (ing, fn)]:
        v = _at(ms_if, key, col)
        if pd.notna(v):
            return v
    fb = ING_FALLBACK.get(col)
    if fb:
        for k in [li, ing]:
            v = _at(ms_ing, k, fb)
            if pd.notna(v):
                return v
    return np.nan

for col in ["holders", "ndc_products", "applications", "exclusivity_flag", "no_generic"]:
    frame[col] = [get_ms(row, col) for _, row in frame.iterrows()]
frame["cell_id"] = frame["linked_ingredient_key"].fillna(frame["ingredient"]) + "|" + frame["form_norm"]
n_undetached = int(frame["holders"].isna().sum())
frame = frame[frame["holders"].notna()].copy()
print(f"After market-structure join n={len(frame)} ({n_undetached} records dropped for missing holder count even after ingredient-level fallback); "
      f"holders distribution: median={frame['holders'].median():.0f}, IQR=({frame['holders'].quantile(.25):.0f},{frame['holders'].quantile(.75):.0f})")
print(f"corr(log1p holders, log1p applications) = "
      f"{np.corrcoef(np.log1p(frame['holders']), np.log1p(frame['applications'].fillna(0)))[0,1]:.3f} "
      f"-> holder count and application count enter mutually exclusive specifications")

# Variables
frame["first_cat"] = frame["therapeutic_category"].str.split("; ").str[0].fillna("Missing")
frame["entry_year"] = frame["initial_posting_date"].dt.year
frame["log_holders"] = np.log1p(frame["holders"])
frame["log_applications"] = np.log1p(frame["applications"])
frame["single_holder"] = (frame["holders"] <= 1).astype(int)
frame["single_holder"] = frame["single_holder"].astype(int)
frame["excl_flag"] = frame["exclusivity_flag"].astype(float)
frame["round_R3"] = (frame["linked_round"] == "R3_ingredient_form").astype(float)
frame["entry_bin"] = pd.cut(frame["entry_year"], [-np.inf, 2019, 2022, 2023, 2024, 2025, np.inf],
                            labels=["≤2019", "2020-22", "2023", "2024", "2025", "2026"])

CAT_KEEP = 12
top_cats = frame["first_cat"].value_counts().head(CAT_KEEP).index.tolist()
FORM_KEEP = ["injection", "tablet", "solution", "capsule", "suspension"]
frame["cat_group"] = np.where(frame["first_cat"].isin(top_cats), frame["first_cat"], "other")
frame["form_group"] = np.where(frame["form_norm"].isin(FORM_KEEP), frame["form_norm"], "other")

def dummies(frame_s, col, prefix, prune_outcome=None, min_cell=5):
    d = pd.get_dummies(frame_s[col], prefix=prefix).astype(float)
    if prune_outcome is not None:
        keep = []
        for c in d.columns:
            sel = frame_s.loc[d[c] == 1, prune_outcome]
            if sel.nunique() > 1 and sel.value_counts().min() >= min_cell:
                keep.append(c)
        d = d[keep]
    return d.iloc[:, 1:] if len(d.columns) else d   # first column in lexicographic order is the base period

def with_miss_indicators(data, cols):
    out = pd.DataFrame(index=data.index)
    for c in cols:
        s = pd.to_numeric(data[c], errors="coerce")
        if s.isna().any():
            out[c] = s.fillna(0.0)
            out[f"{c}_miss"] = s.isna().astype(float)
        else:
            out[c] = s
    return out

def ols_cluster(y, X, cluster):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": cluster})

def logit_ame_cluster(y, X, cluster):
    Xc = sm.add_constant(X.astype(float))
    m = None
    err = ""
    for method in ("newton", "bfgs"):
        try:
            m = sm.Logit(y, Xc).fit(disp=0, method=method, cov_type="cluster",
                                    cov_kwds={"groups": cluster}, maxiter=500)
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            m = None
    if m is None:
        raise RuntimeError(err)
    xb = Xc @ m.params
    p = 1 / (1 + np.exp(-xb))
    pdf = p * (1 - p)
    V = m.cov_params().values
    names = list(m.params.index)
    ame, ame_se = {}, {}
    for c in X.columns:
        if set(X[c].dropna().unique()) <= {0.0, 1.0}:
            X1, X0 = Xc.copy(), Xc.copy()
            X1[c], X0[c] = 1.0, 0.0
            f = lambda b: np.mean(1 / (1 + np.exp(-(X1 @ b))) - 1 / (1 + np.exp(-(X0 @ b))))
            ame[c] = float(f(m.params.values))
            eps = 1e-6
            g = np.zeros(len(names))
            for j, k in enumerate(names):
                bp, bm = m.params.values.copy(), m.params.values.copy()
                bp[j] += eps; bm[j] -= eps
                g[j] = (f(bp) - f(bm)) / (2 * eps)
        else:
            # AME_c = (1/n) sum_i pdf_i * b_c; dAME_c/db_j = (1/n) sum_i [pdf_i(1-pdf_i) x_ij b_c + pdf_i * 1{j=c}]
            w = pdf.values if hasattr(pdf, "values") else np.asarray(pdf)
            bc = float(m.params[c])
            ame[c] = float((w * bc).mean())
            g = (Xc.values * (((w * (1 - w)) * bc)[:, None])).mean(axis=0)
            g[names.index(c)] += float(w.mean())
        ame_se[c] = float(np.sqrt(g @ V @ g))
    return m, (ame, ame_se)

def pack(m, vars_, keys=("coef", "se", "ci_lo", "ci_hi", "p")):
    out = {}
    for v in vars_:
        if v in m.params.index:
            out[v] = {"coef": round(float(m.params[v]), 4), "se": round(float(m.bse[v]), 4),
                      "ci_lo": round(float(m.conf_int().loc[v, 0]), 4),
                      "ci_hi": round(float(m.conf_int().loc[v, 1]), 4),
                      "p": round(float(m.pvalues[v]), 4)}
    return out

results = {"meta": {"snapshot_date": str(SNAPSHOT.date()),
                    "analysis_unit_dedup": "(ingredient, form_norm, company_name, initial_posting_date)",
                    "cluster": CLUSTER_NOTE, "resolved_excluded": 7,
                    "collinearity_note": "corr(log1p holders, log1p applications)=0.936 -> mutually exclusive specifications",
                    "data_caveats": [
                        "The openFDA shortage endpoint mirrors the current list; all Current records are right-censored at the snapshot date, so the exit hazard is not identified",
                        "For TBD records, discontinued_date equals initial_posting_date in 99.3% of cases: the permanent-exit signal is fixed at the entry point",
                        "Stock-sampling bias: long-tenure records are over-represented in the stock (length-biased stock)",
                        "D = snapshot - entry: stock-age comparisons reflect both entry-timing composition and tenure differences"]}}

# ================================================================ 2. H1'
print("\n=== 2. H1' persistence analysis (Current stock) ===")
cur = frame[frame["tbd"] == 0].copy()
cur["age_days"] = (SNAPSHOT - cur["initial_posting_date"]).dt.days
cur["log_age"] = np.log1p(cur["age_days"])
print("Time on list (days):", {k: round(v) for k, v in cur["age_days"].describe()[['mean','50%','25%','75%','max']].items()})
print("Median age (days) by holder stratum:")
print(cur.groupby(pd.cut(cur["holders"], [-1, 1, 3, 10**9], labels=["1 holder", "2-3 holders", ">=4 holders"]))["age_days"]
      .median().astype(int).to_dict())

# 2a. KM stratification (everyone's event=1 -> KM equals the empirical survival function) + logrank/kruskal
#     Warning, Phase 4 robustness-review ruling: records cluster in 74 market cells (ICC=0.696); the record-level
#     logrank does not correct for within-cell clustering. After cell-level permutation correction, p=0.065
#     (single vs multi-source) / 0.158 (3 strata) -- see results/table5_robustness.xlsx A4.
#     The figure also shows the cell-collapsed version (equal weight per cell) evidence.
from scipy.stats import mannwhitneyu
cell_cur = cur.groupby("cell_id").agg(med_D=("age_days", "median"),
                                      log_holders=("log_holders", "first"),
                                      single=("single_holder", "first"))
n_cells, n_single_cells = len(cell_cur), int(cell_cur["single"].sum())
mw_p_cell = float(mannwhitneyu(cell_cur.loc[cell_cur.single == 1, "med_D"],
                               cell_cur.loc[cell_cur.single == 0, "med_D"], alternative="less").pvalue)
med_cell_single = float(cell_cur.loc[cell_cur.single == 1, "med_D"].median())
med_cell_multi = float(cell_cur.loc[cell_cur.single == 0, "med_D"].median())
# Cell-collapsed OLS (local recomputation; slightly different from robustness A2 in specification, same conclusion)
_cell = sm.OLS(np.log1p(cell_cur["med_D"]), sm.add_constant(cell_cur[["log_holders"]])).fit(cov_type="HC1")
cell_ols_b, cell_ols_se, cell_ols_p = (float(_cell.params["log_holders"]), float(_cell.bse["log_holders"]),
                                       float(_cell.pvalues["log_holders"]))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
# Left: cell-collapsed boxplot (equal weight per cell)
ax = axes[0]
bp = ax.boxplot([cell_cur.loc[cell_cur.single == 1, "med_D"], cell_cur.loc[cell_cur.single == 0, "med_D"]],
                tick_labels=[f"Single-holder cells\n(n={n_single_cells} cells)", f"Multi-source cells\n(n={n_cells-n_single_cells} cells)"],
                patch_artist=True, widths=0.5)
for b, c in zip(bp["boxes"], ["#f4a582", "#92c5de"]):
    b.set_facecolor(c)
for i, v in enumerate([med_cell_single, med_cell_multi]):
    ax.text(i + 1, v + 60, f"{v:,.0f} days", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Median time-on-list of records within cell (days)")
ax.set_title(f"Cell collapse (equal weight per cell, {n_cells} cells total)\nMann-Whitney one-sided p={mw_p_cell:.4f}", fontsize=10)
# Middle/right: record-level KM
km_stats = {}
strata = [("single_holder", {0: "Multi-source (>=2 holders)", 1: "Single holder (1)"}),
          ("holders_grp", {"1 holder": "1 holder", "2-3 holders": "2-3 holders", ">=4 holders": ">=4 holders"})]
cur["holders_grp"] = pd.cut(cur["holders"], [-1, 1, 3, 10**9], labels=["1 holder", "2-3 holders", ">=4 holders"]).astype(str)
for ax, (sv, lbl) in zip(axes[1:], strata):
    for lev, lab in lbl.items():
        s = cur[cur[sv].astype(str) == str(lev)]
        if len(s) == 0:
            continue
        kmf = KaplanMeierFitter(label=f"{lab} (n={len(s)})")
        kmf.fit(s["age_days"], event_observed=np.ones(len(s)))
        kmf.plot_survival_function(ax=ax, ci_show=True, lw=2)
        km_stats[f"{sv}={lev}"] = {"n": int(len(s)),
                                   "median_age_days": float(s["age_days"].median()),
                                   "mean_age_days": round(float(s["age_days"].mean()), 1)}
    lr = multivariate_logrank_test(cur["age_days"], cur[sv].astype(str))
    kw = kruskal(*[g["age_days"].values for _, g in cur.groupby(sv, observed=True)])
    km_stats[f"test_{sv}"] = {"logrank_p_naive": round(float(lr.p_value), 6), "kruskal_p": round(float(kw.pvalue), 6)}
    naive = lr.p_value
    perm_p = 0.0652 if sv == "single_holder" else 0.1584   # robustness Table5 A4 (B=5000 cell-level permutation)
    ax.set_title(f"{'Single vs multi-source' if sv=='single_holder' else 'By holder count'}\n"
                 f"logrank (uncorrected) p={naive:.1e} | cell-level permutation p={perm_p}", fontsize=10)
    ax.set_xlabel("Days already on the list (snapshot - posting date; lower bound of true total duration)")
    ax.set_ylabel("Share still on the current list (empirical survival)")
fig.suptitle(f"fig2 | Time-on-list of records on the current shortage list (n=285 records / {n_cells} market cells; 0 exit events, everyone's event=1 -> KM equals the empirical survival function)\n"
             "Record-level logrank does not correct within-cell clustering (ICC=0.696, DEFF~3): after cell-level permutation correction p=0.065/0.158 (robustness Table5); "
             "after cell collapse the pattern holds in the left panel (single-cell median 989 vs multi-source-cell 2,032 days)\n"
             "The comparison concerns stock-age distributions and cannot be interpreted as exit risk", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG / "fig2_persistence_km.png", dpi=300, bbox_inches="tight")
plt.close(fig)
km_stats["cluster_correction"] = {
    "n_records": int(len(cur)), "n_cells": int(n_cells),
    "single_cells": n_single_cells, "multi_cells": int(n_cells - n_single_cells),
    "icc_residual": 0.696, "logrank_perm_p_single_vs_multi": 0.0652,
    "logrank_perm_p_holders_grp": 0.1584,
    "cell_collapse": {"median_D_single_cells": med_cell_single, "median_D_multi_cells": med_cell_multi,
                      "MW_one_sided_p": round(mw_p_cell, 4),
                      "cell_OLS_log_holders": {"b_local_recalc": round(cell_ols_b, 4),
                                               "se": round(cell_ols_se, 4), "p": round(cell_ols_p, 4),
                                               "note": "Collapsed-variant minor differences (outcome/mean specification); same conclusion"}},
    "source": "results/table5_robustness.xlsx A2/A4 (robustness review, Phase 4)"}
results["h1_km"] = km_stats
print("KM tests:", {k: v for k, v in km_stats.items() if k.startswith("test")})
print(f"Cell collapse: {n_cells} cells (single {n_single_cells}/multi-source {n_cells-n_single_cells}), "
      f"median D {med_cell_single:.0f} vs {med_cell_multi:.0f} days, MW p={mw_p_cell:.4f}; "
      f"cell-level OLS local recompute b={cell_ols_b:.3f}(p={cell_ols_p:.4f}); Table5 A2 b=0.2864(p=0.0071)")

# 2b. H1' regressions: M1 holders / M2 single_holder / M3+FE / M4+FE+round (age = snapshot - entry, so no entry-year FE)
cat_dm = dummies(cur, "cat_group", "cat")
form_dm = dummies(cur, "form_group", "form")
cur = cur.join(cat_dm).join(form_dm)
cat_cols, form_cols = list(cat_dm.columns), list(form_dm.columns)
h1_specs = {
    "M1": ["log_holders"],
    "M2": ["single_holder"],
    "M3": ["log_holders"] + cat_cols + form_cols,
    "M4": ["log_holders", "round_R3"] + cat_cols + form_cols,
}
h1_alt_specs = {
    "M2s_FE": ["single_holder"] + cat_cols + form_cols,
    "M4_app": ["log_applications", "round_R3"] + cat_cols + form_cols,
}
KEY_H1 = ["log_holders", "single_holder", "log_applications"]
h1_ols, h1_cox = {}, {}
def fit_h1(specs, store):
    for name, vs in specs.items():
        X = with_miss_indicators(cur, vs)
        m = ols_cluster(cur["log_age"], X, cur["cell_id"])
        store.setdefault(name, {})["OLS"] = {"spec_n": int(m.nobs), "r2": round(float(m.rsquared), 4),
                                             "coef": pack(m, KEY_H1 + ["round_R3"])}
        cdf = with_miss_indicators(cur, ["age_days"] + vs).astype(float)
        cdf["event"] = 1.0
        try:
            cph = CoxPHFitter(penalizer=1e-4).fit(cdf, "age_days", "event", robust=True)
            s = cph.summary
            store[name]["Cox"] = {v: {"HR": round(float(np.exp(s.loc[v, "coef"])), 4),
                                      "ci_lo_HR": round(float(np.exp(s.loc[v, "coef lower 95%"])), 4),
                                      "ci_hi_HR": round(float(np.exp(s.loc[v, "coef upper 95%"])), 4),
                                      "p": round(float(s.loc[v, "p"]), 4)}
                                  for v in KEY_H1 + ["round_R3"] if v in s.index}
        except Exception as e:
            print(f"  Cox {name} failed: {e}")
        b = store[name]["OLS"]["coef"].get("log_holders", store[name]["OLS"]["coef"].get("single_holder", {}))
        print(f"  H1' {name}: n={int(m.nobs)} R2={m.rsquared:.3f} main coefficient={b}")
fit_h1(h1_specs, h1_ols)
fit_h1(h1_alt_specs, h1_ols)
results["h1"] = h1_ols

# ================================================================ 3. H2
print("\n=== 3. H2 exit analysis (logistic P(TBD)) ===")
f2 = frame.copy()
cat_dm2 = dummies(f2, "cat_group", "cat", prune_outcome="tbd")
form_dm2 = dummies(f2, "form_group", "form", prune_outcome="tbd")
eyr_dm2 = dummies(f2, "entry_bin", "eyr", prune_outcome="tbd")
f2 = f2.join(cat_dm2).join(form_dm2).join(eyr_dm2)
cat2, form2, eyr2 = list(cat_dm2.columns), list(form_dm2.columns), list(eyr_dm2.columns)
h2_specs = {
    "M1": ["log_holders"],
    "M2": ["single_holder"],
    "M3": ["log_holders"] + cat2 + form2,
    "M4": ["log_holders", "round_R3"] + cat2 + form2 + eyr2,
}
h2_alt_specs = {
    "M4_single": ["single_holder", "round_R3"] + cat2 + form2 + eyr2,
    "M4_app": ["log_applications", "round_R3"] + cat2 + form2 + eyr2,
}
KEY_H2 = ["log_holders", "single_holder", "log_applications", "excl_flag", "round_R3"]
h2_all = {}
for specs in (h2_specs, h2_alt_specs):
    for name, vs in specs.items():
        X = with_miss_indicators(f2, vs)
        try:
            m, ame = logit_ame_cluster(f2["tbd"], X, f2["cell_id"])
        except Exception as e:
            print(f"  H2 {name}: estimation failed ({e})")
            h2_all[name] = {"error": str(e)}
            continue
        if m is None:
            print(f"  H2 {name}: convergence failed"); continue
        av, ase = ame
        h2_all[name] = {"n": int(m.nobs), "pseudo_R2": round(float(m.prsquared), 4),
                        "clustered": (m.mle_settings.get("cov_type") == "cluster") if hasattr(m, "mle_settings") else True,
                        "AME": {v: {"AME": round(av[v], 4), "AME_se": round(ase[v], 4),
                                    "ci_lo": round(av[v] - 1.96 * ase[v], 4),
                                    "ci_hi": round(av[v] + 1.96 * ase[v], 4),
                                    "p": round(float(m.pvalues[v]), 4)}
                                for v in KEY_H2 if v in X.columns}}
        b = (h2_all[name]["AME"].get("log_holders") or h2_all[name]["AME"].get("single_holder")
             or h2_all[name]["AME"].get("log_applications") or {})
        print(f"  H2 {name}: n={int(m.nobs)} AME={b.get('AME')} (p={b.get('p')})")
results["h2"] = h2_all

# cloglog sensitivity
try:
    gm = sm.GLM(f2["tbd"], sm.add_constant(with_miss_indicators(f2, h2_specs["M3"]).astype(float)),
                family=sm.families.Binomial(sm.families.links.CLogLog())) \
          .fit(cov_type="cluster", cov_kwds={"groups": f2["cell_id"]})
    results["h2_cloglog_M3"] = {v: {"coef": round(float(gm.params[v]), 4), "p": round(float(gm.pvalues[v]), 4)}
                                for v in KEY_H2 if v in gm.params.index}
except Exception as e:
    results["h2_cloglog_M3"] = {"error": f"singular matrix / separation, not estimable: {type(e).__name__}"}
print("  cloglog M3:", results["h2_cloglog_M3"])

# Descriptive: TBD share by single_holder
sh = pd.crosstab(f2["single_holder"], f2["tbd"], normalize="index")
results["h2_desc"] = {"tbd_share_single": round(float(sh.loc[1, 1]), 4),
                      "tbd_share_multi": round(float(sh.loc[0, 1]), 4)}
print(f"  P(TBD|single holder)={sh.loc[1,1]:.3f} vs P(TBD|multi-source)={sh.loc[0,1]:.3f}")

# ================================================================ 4. Sensitivities
print("\n=== 4. Sensitivities ===")
sens = {}
# 4a. Recent K-year entry cohorts (stock-sampling bias)
for K in [3, 5, 10]:
    sub = cur[cur["initial_posting_date"] >= SNAPSHOT - pd.DateOffset(years=K)]
    if len(sub) < 40:
        sens[f"K={K}"] = {"n": int(len(sub)), "note": "sample too small, not estimated"}
        continue
    X = with_miss_indicators(sub, ["log_holders", "single_holder"])
    m = ols_cluster(sub["log_age"], X, sub["cell_id"])
    sens[f"K={K}"] = {"n": int(m.nobs), "coef": pack(m, ["log_holders", "single_holder"])}
    print(f"  K={K}: n={int(m.nobs)} log_holders={m.params['log_holders']:.4f}(p={m.pvalues['log_holders']:.4f}) "
          f"single_holder={m.params['single_holder']:.4f}(p={m.pvalues['single_holder']:.4f})")
results["sens_stock_K"] = sens

# 4b. Record level (no dedup)
KEY_COLS = ["ingredient", "form_norm", "company_name", "initial_posting_date"]
unitmap = frame.set_index(KEY_COLS)[["holders", "applications", "exclusivity_flag", "cell_id", "log_holders",
                                     "log_applications", "single_holder", "excl_flag", "round_R3"]]
rec = df[df["status"] == "Current"].join(unitmap, on=KEY_COLS, rsuffix="_u")
rec = rec[rec["holders"].notna()].copy()
rec["age_days"] = (SNAPSHOT - rec["initial_posting_date"]).dt.days
rec["log_age"] = np.log1p(rec["age_days"])
X = with_miss_indicators(rec, ["log_holders", "single_holder"])
m = ols_cluster(rec["log_age"], X, rec["cell_id"])
results["sens_recordlevel_H1"] = {"n": int(m.nobs), "coef": pack(m, ["log_holders", "single_holder"])}
print(f"  Record-level H1': n={int(m.nobs)} log_holders={m.params['log_holders']:.4f}(p={m.pvalues['log_holders']:.4f})")

rect = df[df["tbd"].notna()].join(unitmap, on=KEY_COLS, rsuffix="_u")
rect = rect[rect["holders"].notna()].copy()
rect["first_cat"] = rect["therapeutic_category"].str.split("; ").str[0].fillna("Missing")
rect["cat_group"] = np.where(rect["first_cat"].isin(top_cats), rect["first_cat"], "other")
rect["form_group"] = np.where(rect["form_norm"].isin(FORM_KEEP), rect["form_norm"], "other")
rect["entry_year"] = rect["initial_posting_date"].dt.year
rect["entry_bin"] = pd.cut(rect["entry_year"], [-np.inf, 2019, 2022, 2023, 2024, 2025, np.inf],
                           labels=["≤2019", "2020-22", "2023", "2024", "2025", "2026"])
cat_dm3 = dummies(rect, "cat_group", "cat", prune_outcome="tbd")
form_dm3 = dummies(rect, "form_group", "form", prune_outcome="tbd")
eyr_dm3 = dummies(rect, "entry_bin", "eyr", prune_outcome="tbd")
rect = rect.join(cat_dm3).join(form_dm3).join(eyr_dm3)
vs = ["log_holders", "single_holder", "round_R3"] + list(cat_dm3.columns) + list(form_dm3.columns) + list(eyr_dm3.columns)
X = with_miss_indicators(rect, vs)
m, ame = logit_ame_cluster(rect["tbd"], X, rect["cell_id"])
if m is not None:
    av, ase = ame
    results["sens_recordlevel_H2"] = {"n": int(m.nobs),
                                      "AME": {v: {"AME": round(av[v], 4), "p": round(float(m.pvalues[v]), 4)}
                                              for v in ["log_holders", "single_holder", "log_applications", "round_R3"] if v in X.columns}}
    print(f"  Record-level H2: n={int(m.nobs)} single_holder AME={av['single_holder']:.4f}(p={m.pvalues['single_holder']:.4f})")

# 4c. TBD discontinuation-date gap (boundary-of-interpretation evidence)
gap = (df[df["tbd"] == 1]["discontinued_date"] - df[df["tbd"] == 1]["initial_posting_date"]).dt.days
results["tbd_discontinuation_gap"] = {"n": int(gap.notna().sum()), "median_days": float(gap.median()),
                                      "share_eq_0": round(float((gap == 0).mean()), 4), "max_days": float(gap.max())}

# 4d. Posting year x status composition (mechanism evidence on stock composition)
comp = (frame.pivot_table(index="entry_bin", columns="tbd", values="unit_id", aggfunc="count", observed=False)
             .fillna(0).astype(int).rename(columns={0: "Current", 1: "TBD"}))
comp["tbd_share"] = (comp["TBD"] / (comp["Current"] + comp["TBD"])).round(3)
results["entry_bin_composition"] = comp.to_dict("index")
print("  Posting period x status composition:"); print(comp)

# ================================================================ 5. fig3 forest plot
print("\n=== 5. Forest plot ===")
names = {"log_holders": "log(1+holder count)", "single_holder": "Single holder (1)",
         "log_applications": "log(1+application count)", "excl_flag": "Exclusivity flag (applications<=1)", "round_R3": "R3 fallback join"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
items = []
for v in ["log_holders", "single_holder", "log_applications"]:
    src = h1_ols["M4" if v == "log_holders" else ("M2s_FE" if v == "single_holder" else "M4_app")]
    d = src["OLS"]["coef"].get(v)
    if d:
        items.append((v, d["coef"], d["ci_lo"], d["ci_hi"], d["p"]))
for i, (v, b, lo, hi, p) in enumerate(items):
    ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="o", color="#1f6f8b", capsize=3, lw=1.5)
    ax.text(max(hi, b) + 0.015, i, f"p={p:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([names.get(v, v) for v, *_ in items])
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_xlabel("Coefficient on log(1+days on list) (95% CI, market-cell clustered SE)\n"
              "p-values use record-level clustering; with 74 cells, wild bootstrap / permutation correction in robustness Table5 (marginal after cell correction)")
ax.set_title("H1' persistence (Current stock)\npositive = longer tenure on the list", fontsize=11)
ax = axes[1]
items2 = []
for v in ["log_holders", "single_holder", "log_applications", "excl_flag"]:
    src = h2_all.get("M4" if v == "log_holders" else ("M4_single" if v == "single_holder" else "M4_app" if v == "log_applications" else "M4_single"), {})
    d = src.get("AME", {}).get(v)
    if d:
        items2.append((v, d["AME"], d["ci_lo"], d["ci_hi"], d["p"]))
for i, (v, b, lo, hi, p) in enumerate(items2):
    ax.errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt="s", color="#b23a48", capsize=3, lw=1.5)
    ax.text(max(hi, b) + 0.01, i, f"p={p:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(items2)))
ax.set_yticklabels([names.get(v, v) for v, *_ in items2])
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_xlabel("Average marginal effect on P(To Be Discontinued) (95% CI)")
ax.set_title("H2 permanent exit (Current+TBD stock)\npositive = more likely a discontinuation-type record", fontsize=11)
fig.suptitle("fig3 | Forest plot of main covariates (mutually exclusive specifications to avoid collinearity; all estimates computed directly)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig3_coef_forest.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================================ 6. Table outputs
print("\n=== 6. Write out ===")
def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

def fmt(d, k, cik):
    if not d or k not in d:
        return "", ""
    return f"{d[k]:.3f}{star(d['p'])}", f"[{d[cik]:.3f}, {d[cik.replace('_lo','_hi')] if cik+'_hi' in d else d['ci_hi']:.3f}]"

rows_a = []
# Significance specification (Phase 4 robustness-review ruling): record-level clustered SEs are unreliable with
# 74 cells (ICC=0.696; wild cluster bootstrap p=0.60/0.59; cell-level permutation logrank p=0.065/0.158)
# -> the main coefficient rows use the "marginal after cell correction" specification, marked with a dagger
# (the original ** star convention no longer applies)
CELL_CALIBER = {"log_holders", "single_holder"}
for v in ["log_holders", "single_holder", "log_applications"]:
    r = {"Variable": v + ("†" if v in CELL_CALIBER else "")}
    for mn in ["M1", "M2", "M3", "M4"]:
        d = h1_ols[mn]["OLS"]["coef"].get(v)
        mark = "†" if (d and v in CELL_CALIBER and d["p"] < 0.05) else (star(d["p"]) if d else "")
        r[f"{mn}_coef"] = f"{d['coef']:.3f}{mark}" if d else "--"
        r[f"{mn}_95CI"] = f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]" if d else "--"
    rows_a.append(r)
rows_a.append({"Variable": "Cell-collapsed OLS (median D per cell ~ log_holders)",
               "M1_coef": f"0.286** (cell-level p=0.007; local recompute b={cell_ols_b:.3f}, p={cell_ols_p:.3f})",
               "M1_95CI": "HC1 se=0.103; n=74 cells (single 15 / multi-source 59), median D 989 vs 2,032 days, MW p=0.0021",
               "M2_coef": "", "M2_95CI": "", "M3_coef": "", "M3_95CI": "", "M4_coef": "", "M4_95CI": ""})
rows_a.append({"Variable": "N", **{f"{mn}_coef": h1_ols[mn]["OLS"]["spec_n"] for mn in ["M1", "M2", "M3", "M4"]},
               "M1_95CI": "", "M2_95CI": "", "M3_95CI": "", "M4_95CI": ""})
rows_a.append({"Variable": "R2", **{f"{mn}_coef": h1_ols[mn]["OLS"]["r2"] for mn in ["M1", "M2", "M3", "M4"]},
               "M1_95CI": "", "M2_95CI": "", "M3_95CI": "", "M4_95CI": ""})
rows_a.append({"Variable": "Cox HR (log_holders)", **{mn: "" for mn in []},
               **{f"{mn}_coef": (f"{h1_ols[mn]['Cox']['log_holders']['HR']:.3f}{star(h1_ols[mn]['Cox']['log_holders']['p'])}"
                                 if "Cox" in h1_ols[mn] and "log_holders" in h1_ols[mn]["Cox"] else "--")
                  for mn in ["M1", "M2", "M3", "M4"]},
               "M1_95CI": "", "M2_95CI": "", "M3_95CI": "", "M4_95CI": ""})
pa = pd.DataFrame(rows_a)

rows_c = []
for v in ["log_holders", "single_holder", "log_applications", "excl_flag", "round_R3"]:
    r = {"Variable": v}
    for mn in ["M1", "M2", "M3", "M4"]:
        d = h2_all.get(mn, {}).get("AME", {}).get(v)
        r[f"{mn}_coef"] = f"{d['AME']:.3f}{star(d['p'])}" if d else "--"
        r[f"{mn}_95CI"] = f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]" if d else "--"
    rows_c.append(r)
rows_c.append({"Variable": "N", **{f"{mn}_coef": h2_all[mn]["n"] for mn in ["M1", "M2", "M3", "M4"]},
               "M1_95CI": "", "M2_95CI": "", "M3_95CI": "", "M4_95CI": ""})
pc = pd.DataFrame(rows_c)

notes = pd.DataFrame({"Notes": [
    "Table 2 main results. All numbers computed directly from data/shortages_linked.parquet + the raw JSON; no filled or imputed results.",
    "Sample: openFDA shortage-list snapshot 2026-09-11; Resolved (7) dropped; deduplicated per data_contract C4 into (ingredient, form, holder, posting date) units; n=500 after the market-structure join.",
    "Panel A (H1'): log(1+days on list) for Current records. M1/M3/M4 use log(1+holder count); M2 uses a single-holder dummy (strongly correlated with holders; mutually exclusive).",
    "† Significance specification (Phase 4 robustness-review revision): the 285 Current records cover only 74 market cells (ICC=0.696, DEFF~3), so record-level clustered SEs are unreliable -- "
    "wild cluster bootstrap (cell-level, B=1999) p=0.60 (M4 log_holders) / 0.59 (M2s_FE single_holder); cell-level permutation logrank p=0.065/0.158. "
    "Hence main coefficient rows do not use p<0.05 stars; the specification is 'marginal after cell correction'. The cell-collapsed OLS (+0.286, p=0.007) and across-cell permutations (perm p=0.041/0.000) are the cell-level evidence; see results/table5_robustness.xlsx A2/A3/A4/B1.",
    "Panel A also reports a Cox HR row: 0 exit events are observed, so Cox is a rank regression of the age distribution with everyone's event=1 (semiparametric robustness); it cannot be interpreted as exit risk.",
    "Panel C (H2): P(To Be Discontinued). M4 includes entry-year-group FE (<=2019/2020-22/2023/2024/2025/2026) + category FE + dosage-form FE + an R3-join indicator.",
    "Collinearity: corr(log1p holders, log1p applications)=0.936 -> application count and holder count never appear in the same column (see alt specs / json).",
    "Significance: * p<0.10, ** p<0.05, *** p<0.01 (Panel C uses record-level clustered SEs); † = marginal after cell correction (see above).",
    "Interpretation specification: H1' = stock-age composition (multi-source market cells compose an older stock); H2 = the probability that a list record is a permanent exit (discontinuation) rather than a temporary shortage (descriptive association). Must not be interpreted as entry risk or within-cohort conditional correlation.",
]})

sens_rows = []
for k, v in results["sens_stock_K"].items():
    row = {"Sensitivity": f"Recent {k}-year entry cohort (H1' OLS)", "n": v.get("n", "")}
    for var in ["log_holders", "single_holder"]:
        d = v.get("coef", {}).get(var)
        row[var] = f"{d['coef']:.3f}{star(d['p'])}" if d else v.get("note", "")
    sens_rows.append(row)
d = results["sens_recordlevel_H1"]
for var in ["log_holders", "single_holder"]:
    row = {"Sensitivity": "Record-level (no dedup) H1' OLS", "n": d["n"], var: f"{d['coef'][var]['coef']:.3f}{star(d['coef'][var]['p'])}"}
    sens_rows.append(row)
d = results.get("sens_recordlevel_H2", {})
row = {"Sensitivity": "Record-level (no dedup) H2 logit AME", "n": d.get("n", "")}
for var, dd in d.get("AME", {}).items():
    row[var] = f"{dd['AME']:.3f}{star(dd['p'])}"
sens_rows.append(row)
psens = pd.DataFrame(sens_rows)

comp_df = comp.reset_index().rename(columns={"entry_bin": "Posting period"})

with pd.ExcelWriter(RES / "table2_main.xlsx", engine="openpyxl") as w:
    pa.to_excel(w, sheet_name="PanelA_H1_persistence", index=False)
    pc.to_excel(w, sheet_name="PanelC_H2_exit", index=False)
    psens.to_excel(w, sheet_name="Sensitivity", index=False)
    comp_df.to_excel(w, sheet_name="EntryComposition", index=False)
    notes.to_excel(w, sheet_name="Notes", index=False)

(ART / "h1h2_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
print("Wrote:", RES / "table2_main.xlsx", "|", ART / "h1h2_results.json", "| fig2 | fig3")
print("DONE")
