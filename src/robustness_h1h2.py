"""
paper18 robustness_h1h2.py -- Phase 4: robustness audit for H1' (stock persistence) / H2 (discontinuation probability)
Run: python3 src/robustness_h1h2.py

Scope (as specified by the project lead; all computed directly, zero imputation):
  A. Market-cell clustering corrections for KM/logrank and OLS
     A1 n_eff: record counts vs distinct market-cell counts per stratum
     A2 Cell collapse (median/mean D per cell) + cell-level tests
     A3 Wild cluster bootstrap (Rademacher, cell level, percentile-t)
     A4 Cell-level permutation logrank (permute stratum labels across cells, preserving cell structure)
  B. Placebos
     B1 Cell-level permutation tests: shuffle concentration labels across cells (holders is constant
        within a cell -> permutation only possible across cells), B=1000
     B2 Fake snapshot dates: re-estimate with +/-90/180-day shifts; also drop boundary records with D<=90 days
  C. Specification curve: outcome (log1p D / log1p(D+30)) x category FE x form FE x join round x
     posting-period FE = 2^5 = 64 specifications
  D. Sample and definition sensitivities
     D1 Exclude / keep the R3 fallback join
     D2 Worst-case envelope for the 41 OB/FDA-only fallback cells (holders all imputed as 1 vs all as p95)
     D3 Alternative D definitions: log(D+30), months, quarterly bins
  E. Posting-period decomposition: cohort-specific H1' coefficients + fig5 (posting year x status composition)
  F. H2 supplements: re-estimate excluding R3; single vs multi TBD share within 2025/2026 cohorts (Fisher exact)

Outputs: results/table5_robustness.xlsx / figures/fig4_spec_curve.png / figures/fig5_entry_cohort.png
      artifacts/robustness_results.json / results/robustness_README.md (conclusions maintained separately in the README)
CPU only, memory <1GB, ~8000 total OLS/logrank permutations, minutes-scale runtime.
"""
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"]:
    if any(f.lower() == x.name.lower() for x in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, ttest_ind, fisher_exact
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA, ART, RES, FIG = ROOT / "data", ROOT / "artifacts", ROOT / "results", ROOT / "figures"
SNAPSHOT = pd.Timestamp("2026-09-11")
SEED = 20260911
R = {}  # main results container

# ================================================================ 0. Data construction (replicates estimate_h1h2.py step by step, baseline unchanged)
print("=== 0. Data construction (replicating the baseline pipeline) ===")
df = pd.read_parquet(DATA / "shortages_linked.parquet")
raw = json.load(open(ROOT / "data/raw/drug-shortages-0001-of-0001.json"))["results"]
assert len(raw) == len(df)
raw_ip = pd.to_datetime([r.get("initial_posting_date") for r in raw], format="%m/%d/%Y", errors="coerce")
assert (raw_ip.values == df["initial_posting_date"].values).all(), "raw<->parquet row-order mismatch"
df["discontinued_date"] = pd.to_datetime([r.get("discontinued_date") for r in raw], format="%m/%d/%Y", errors="coerce")

n0 = len(df)
df = df[df["status"].isin(["Current", "To Be Discontinued"])].copy()
df["tbd"] = (df["status"] == "To Be Discontinued").astype(int)
df["unit_id"] = df.groupby(["ingredient", "form_norm", "company_name", "initial_posting_date"]).ngroup()
frame = df.drop_duplicates("unit_id").copy()
assert int(df.groupby("unit_id")["tbd"].nunique().max()) == 1

ms = pd.read_parquet(DATA / "market_structure_ingredient_form.parquet")
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
n_dropped_holders_na = int(frame["holders"].isna().sum())
frame["linked_round_isR3"] = (frame["linked_round"] == "R3_ingredient_form").astype(float)
frame_dropped = frame[frame["holders"].isna()].copy()          # 41 OB/FDA-only fallback cells
frame = frame[frame["holders"].notna()].copy()
print(f"Records {n0} -> dedup {frame_dropped.shape[0] + len(frame)} -> drop holders-missing {n_dropped_holders_na} -> n={len(frame)}")

# Common variables
for d in (frame, frame_dropped):
    d["first_cat"] = d["therapeutic_category"].str.split("; ").str[0].fillna("Missing")
    d["entry_year"] = d["initial_posting_date"].dt.year
    d["log_holders"] = np.log1p(d["holders"])
    d["log_applications"] = np.log1p(d["applications"])
    d["single_holder"] = (d["holders"] <= 1).astype(float)
    d["round_R3"] = d["linked_round_isR3"]
    d["entry_bin"] = pd.cut(d["entry_year"], [-np.inf, 2019, 2022, 2023, 2024, 2025, np.inf],
                            labels=["≤2019", "2020-22", "2023", "2024", "2025", "2026"])

CAT_KEEP, FORM_KEEP = 12, ["injection", "tablet", "solution", "capsule", "suspension"]
top_cats = frame["first_cat"].value_counts().head(CAT_KEEP).index.tolist()
for d in (frame, frame_dropped):
    d["cat_group"] = np.where(d["first_cat"].isin(top_cats), d["first_cat"], "other")
    d["form_group"] = np.where(d["form_norm"].isin(FORM_KEEP), d["form_norm"], "other")

def dummies(data, col, prefix, prune_outcome=None, min_cell=5):
    dd = pd.get_dummies(data[col], prefix=prefix).astype(float)
    if prune_outcome is not None:
        keep = []
        for c in dd.columns:
            sel = data.loc[dd[c] == 1, prune_outcome]
            if sel.nunique() > 1 and sel.value_counts().min() >= min_cell:
                keep.append(c)
        dd = dd[keep]
    return dd.iloc[:, 1:] if len(dd.columns) else dd

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
    m = sm.Logit(y, Xc).fit(disp=0, method="newton", cov_type="cluster", cov_kwds={"groups": cluster}, maxiter=500)
    p = 1 / (1 + np.exp(-(Xc @ m.params)))
    w = np.asarray(p * (1 - p))
    V = m.cov_params().values
    names = list(m.params.index)
    ame, ase = {}, {}
    for c in X.columns:
        if set(X[c].dropna().unique()) <= {0.0, 1.0}:
            X1, X0 = Xc.copy(), Xc.copy(); X1[c], X0[c] = 1.0, 0.0
            f = lambda b: float(np.mean(1 / (1 + np.exp(-(X1 @ b))) - 1 / (1 + np.exp(-(X0 @ b)))))
            ame[c] = f(m.params.values)
            eps = 1e-6; g = np.zeros(len(names))
            for j in range(len(names)):
                bp, bm = m.params.values.copy(), m.params.values.copy()
                bp[j] += eps; bm[j] -= eps
                g[j] = (f(bp) - f(bm)) / (2 * eps)
        else:
            bc = float(m.params[c]); ame[c] = float((w * bc).mean())
            g = (Xc.values * ((w * (1 - w)) * bc)[:, None]).mean(axis=0)
            g[names.index(c)] += float(w.mean())
        ase[c] = float(np.sqrt(g @ V @ g))
    return m, ame, ase

def pack(m, vars_):
    out = {}
    for v in vars_:
        if v in m.params.index:
            out[v] = {"coef": round(float(m.params[v]), 4), "se": round(float(m.bse[v]), 4),
                      "ci_lo": round(float(m.conf_int().loc[v, 0]), 4),
                      "ci_hi": round(float(m.conf_int().loc[v, 1]), 4),
                      "p": round(float(m.pvalues[v]), 4)}
    return out

def star(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

cur = frame[frame["tbd"] == 0].copy()
cur["age_days"] = (SNAPSHOT - cur["initial_posting_date"]).dt.days
cur["log_age"] = np.log1p(cur["age_days"])
cur["holders_grp"] = pd.cut(cur["holders"], [-1, 1, 3, 10**9], labels=["1 holder", "2-3 holders", ">=4 holders"]).astype(str)

cat_dm = dummies(cur, "cat_group", "cat")
form_dm = dummies(cur, "form_group", "form")
eyr_dm = dummies(cur, "entry_bin", "eyr")
cur = cur.join(cat_dm).join(form_dm).join(eyr_dm)
CAT_COLS, FORM_COLS, EYR_COLS = list(cat_dm.columns), list(form_dm.columns), list(eyr_dm.columns)
print(f"Current n={len(cur)} | market cells {cur.cell_id.nunique()} | category FE {len(CAT_COLS)} form FE {len(FORM_COLS)} posting-period FE {len(EYR_COLS)}")

# ================================================================ A. Clustering corrections
print("\n=== A. Market-cell clustering corrections for KM/logrank and OLS ===")

# A1 n_eff
neff_rows = []
for sv, mapping in [("single_holder", {0: "multi-source (>=2 holders)", 1: "single (1 holder)"}),
                    ("holders_grp", {"1 holder": "1 holder", "2-3 holders": "2-3 holders", ">=4 holders": ">=4 holders"})]:
    for lev, lab in mapping.items():
        s = cur[cur[sv].astype(str) == str(lev)] if sv == "holders_grp" \
            else cur[cur[sv].astype(float).astype(int).astype(str) == str(lev)]
        neff_rows.append({"Stratum": f"{sv}={lab}", "Records": len(s), "Distinct cells": s["cell_id"].nunique(),
                          "Records per cell": round(len(s) / max(s["cell_id"].nunique(), 1), 2)})
n_eff = pd.DataFrame(neff_rows)
gsize = cur.groupby("cell_id").size()
icc_resid = None
m0 = sm.OLS(cur["log_age"], sm.add_constant(with_miss_indicators(cur, ["log_holders", "round_R3"])
                                            .join(cur[CAT_COLS + FORM_COLS]))).fit()
resid = pd.DataFrame({"r": m0.resid, "g": cur["cell_id"].values})
gb = resid.groupby("g")["r"]
k = gb.size(); mu = resid["r"].mean(); s_between = gb.mean().var(ddof=1)
s_within = gb.var(ddof=0).groupby(k > 1).mean().loc[True]
icc_resid = float((s_between - s_within) / (s_between + (gsize.loc[gb.size() > 1].mean() - 1) * s_within))
deff = float(1 + (gsize.mean() - 1) * max(icc_resid, 0))
R["A1_neff"] = {"table": n_eff.to_dict("records"),
                "n_cells_total": int(cur.cell_id.nunique()),
                "share_records_in_ge4_cells": round(float((cur["holders"] >= 4).mean()), 4),
                "icc_residual_logage": round(icc_resid, 4), "mean_cluster_size": round(float(gsize.mean()), 2),
                "design_effect": round(deff, 2)}
print(n_eff.to_string(index=False))
print(f"Residual ICC(log_age|controls)={icc_resid:.3f}, mean cell size={gsize.mean():.2f}, design effect DEFF={deff:.2f}")

# A2 Cell collapse
cell_med = cur.groupby(["cell_id", "single_holder", "holders_grp", "holders"], observed=True)["age_days"].median().reset_index()
cell_mean = cur.groupby(["cell_id", "single_holder", "holders"], observed=True)["age_days"].mean().reset_index()
cell_rows = []
for nm, cd0 in [("median D per cell", cell_med), ("mean D per cell", cell_mean)]:
    cd = cd0.copy()
    cd["log_holders"] = np.log1p(cd["holders"])
    a = cd.loc[cd.single_holder == 1, "age_days"]; b = cd.loc[cd.single_holder == 0, "age_days"]
    mw = mannwhitneyu(b, a, alternative="greater")   # H1': multi-source cells older -> one-sided
    tt = ttest_ind(b, a, equal_var=False, alternative="greater")
    cg_cat = cur.groupby("cell_id")["cat_group"].first().reindex(cd["cell_id"]).to_frame("cat_group").reset_index(drop=True)
    cg_form = cur.groupby("cell_id")["form_group"].first().reindex(cd["cell_id"]).to_frame("form_group").reset_index(drop=True)
    cdm = pd.concat([dummies(cg_cat, "cat_group", "cat").reset_index(drop=True),
                     dummies(cg_form, "form_group", "form").reset_index(drop=True)], axis=1)
    cd["log_age"] = np.log1p(cd["age_days"])
    Xc = sm.add_constant(pd.concat([cd[["log_holders"]].reset_index(drop=True), cdm], axis=1).astype(float))
    mc = sm.OLS(cd["log_age"].values, Xc.values).fit()
    lh_i = list(Xc.columns).index("log_holders")
    cell_rows.append({"Collapse specification": nm, "n_cells": len(cd), "single_cells": int((cd.single_holder == 1).sum()),
                      "multi_cells": int((cd.single_holder == 0).sum()),
                      "median_D_single": round(float(cd.loc[cd.single_holder == 1, "age_days"].median()), 0),
                      "median_D_multi": round(float(cd.loc[cd.single_holder == 0, "age_days"].median()), 0),
                      "MW_one_sided_p": round(float(mw.pvalue), 4), "Welch_one_sided_p": round(float(tt.pvalue), 4),
                      "cell_OLS_log_holders": round(float(mc.params[lh_i]), 4),
                      "cell_OLS_HC1_se": round(float(mc.bse[lh_i]), 4),
                      "cell_OLS_p": round(float(mc.pvalues[lh_i]), 4)})
A2 = pd.DataFrame(cell_rows)
R["A2_cellcollapse"] = A2.to_dict("records")
print(A2.to_string(index=False))

# A3 Wild cluster bootstrap (percentile-t, Rademacher, cell level)
def wild_bootstrap_t(y, X, cluster, var, B=1999, seed=SEED):
    rng = np.random.default_rng(seed)
    Xc = sm.add_constant(X.astype(float))
    m = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": cluster})
    t_obs = float(m.tvalues[var])
    fitv, resid_, = m.fittedvalues, m.resid
    cells = pd.unique(pd.Series(cluster))
    gmap = {g: i for i, g in enumerate(cells)}
    cg = np.array([gmap[g] for g in cluster])
    cnt = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=len(cells))
        mb = sm.OLS(fitv + w[cg] * resid_, Xc).fit(cov_type="cluster", cov_kwds={"groups": cluster})
        if abs(float(mb.tvalues[var])) >= abs(t_obs):
            cnt += 1
    return t_obs, cnt / B, B

A3_rows = []
for spec_name, vs, var in [("M4 (log_holders, category+form FE+R3)", ["log_holders", "round_R3"], "log_holders"),
                           ("M2s_FE (single_holder, category+form FE)", ["single_holder"], "single_holder")]:
    X = with_miss_indicators(cur, vs).join(cur[CAT_COLS + FORM_COLS])
    t_obs, p_w, B = wild_bootstrap_t(cur["log_age"], X, cur["cell_id"].values, var)
    m = ols_cluster(cur["log_age"], X, cur["cell_id"])
    A3_rows.append({"spec": spec_name, "variable": var, "coef": round(float(m.params[var]), 4),
                    "clustered_SE_p": round(float(m.pvalues[var]), 4),
                    "wild_t": round(t_obs, 3), "wild_p(percentile-t)": round(p_w, 4), "B": B})
    print(f"  {spec_name}: coef={m.params[var]:.4f} clustered p={m.pvalues[var]:.4f} wild-p={p_w:.4f}")
A3 = pd.DataFrame(A3_rows)
R["A3_wildbootstrap"] = A3.to_dict("records")

# A4 Cell-level permutation logrank: permute single_holder labels across cells (holders constant within cell -> independent variation only at cell level)
def perm_logrank(B=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    cells = cur["cell_id"].unique()
    lab_single_cells = set(cur.loc[cur["single_holder"] == 1, "cell_id"])
    n_single = len(lab_single_cells)
    age = cur["age_days"].values
    cell_of = cur["cell_id"].values
    lr_obs = multivariate_logrank_test(age, cur["single_holder"].astype(str).values)
    stat_obs = float(lr_obs.test_statistic)
    cnt = 0
    for _ in range(B):
        perm_cells = rng.permutation(cells)
        newlab = pd.Series(np.zeros(len(cells), dtype=int), index=perm_cells)
        newlab.iloc[:n_single] = 1
        lab = newlab.reindex(cell_of).values.astype(str)
        st = float(multivariate_logrank_test(age, lab).test_statistic)
        if st >= stat_obs:
            cnt += 1
    return stat_obs, float(lr_obs.p_value), cnt / B, B

stat_obs, lr_naive_p, perm_p, B = perm_logrank()
R["A4_perm_logrank"] = {"logrank_chi2": round(stat_obs, 3), "naive_p_record_level": lr_naive_p,
                        "cell_permutation_p": round(perm_p, 5), "B": B}
print(f"  logrank chi2={stat_obs:.2f}, record-level naive p={lr_naive_p:.2e} -> cell-level permutation p={perm_p:.4f} (B={B})")
# Same treatment for the 3-stratum version
def perm_logrank_grp(B=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    cells = cur["cell_id"].unique()
    grp_of_cell = cur.groupby("cell_id")["holders"].first()
    age = cur["age_days"].values
    cell_of = cur["cell_id"].values
    lr_obs = multivariate_logrank_test(age, cur["holders_grp"].values)
    stat_obs = float(lr_obs.test_statistic)
    # Permutation: permute holder counts across cells (keeping the multiset of values), then re-stratify
    holder_vals_cell = grp_of_cell.values.astype(float).copy()
    cnt = 0
    for _ in range(B):
        pv = rng.permutation(holder_vals_cell)
        lab = pd.Series(pv, index=grp_of_cell.index).reindex(cell_of).values
        lab_grp = pd.cut(lab, [-1, 1, 3, 10**9], labels=["1 holder", "2-3 holders", ">=4 holders"]).astype(str)
        st = float(multivariate_logrank_test(age, lab_grp).test_statistic)
        if st >= stat_obs:
            cnt += 1
    return stat_obs, float(lr_obs.p_value), cnt / B, B

stat3, naive3, perm3, B3 = perm_logrank_grp()
R["A4_perm_logrank"]["holders_grp_chi2"] = round(stat3, 3)
R["A4_perm_logrank"]["holders_grp_naive_p"] = naive3
R["A4_perm_logrank"]["holders_grp_permutation_p"] = round(perm3, 5)
print(f"  3 strata chi2={stat3:.2f}, naive p={naive3:.2e} -> cell-level permutation p={perm3:.4f}")

# ================================================================ B. Placebos
print("\n=== B. Placebos ===")

def base_X(data, spec, eyr=False):
    X = with_miss_indicators(data, spec)
    X = X.join(data[CAT_COLS + FORM_COLS])
    if eyr:
        X = X.join(data[EYR_COLS])
    return X

# B1 Cell-level permutation test: permute log_holders / single_holder value vectors across cells, B=1000
def perm_ols(var, spec, B=1000, seed=SEED, eyr=False):
    rng = np.random.default_rng(seed)
    y = cur["log_age"]
    X = base_X(cur, spec, eyr)
    m = ols_cluster(y, X, cur["cell_id"])
    b_obs = float(m.params[var])
    cell_val = cur.groupby("cell_id")[var].first()
    cells = cell_val.index.values
    cnt, coefs = 0, []
    for _ in range(B):
        pv = pd.Series(rng.permutation(cell_val.values), index=cells)
        xv = pv.reindex(cur["cell_id"]).values
        Xp = X.copy(); Xp[var] = xv
        mp = sm.OLS(y, sm.add_constant(Xp.astype(float))).fit(cov_type="cluster", cov_kwds={"groups": cur["cell_id"]})
        bp = float(mp.params[var])
        coefs.append(bp)
        if abs(bp) >= abs(b_obs):
            cnt += 1
    coefs = np.array(coefs)
    return {"var": var, "b_obs": round(b_obs, 4), "perm_p_two_sided": round(cnt / B, 4), "B": B,
            "perm_q025": round(float(np.quantile(coefs, .025)), 4), "perm_q975": round(float(np.quantile(coefs, .975)), 4)}

B1 = [perm_ols("log_holders", ["log_holders", "round_R3"]),
      perm_ols("single_holder", ["single_holder"]),
      perm_ols("log_holders", ["log_holders", "round_R3"], eyr=True)]
R["B1_perm_ols"] = B1
for b in B1:
    print(f"  permutation {b['var']}: b={b['b_obs']}, perm-p={b['perm_p_two_sided']}, 95%perm=[{b['perm_q025']},{b['perm_q975']}]")

# B2 Fake snapshot dates +/-90/180 days + drop boundary records
B2_rows = []
for delta in [-180, -90, 0, 90, 180]:
    d_shift = cur.copy()
    d_shift["age_days"] = (SNAPSHOT + pd.Timedelta(days=delta) - d_shift["initial_posting_date"]).dt.days
    neg = int((d_shift["age_days"] < 0).sum())
    d_shift = d_shift[d_shift["age_days"] >= 0].copy()   # with negative shifts, not-yet-posted records are absent from the fake stock
    d_shift["log_age"] = np.log1p(d_shift["age_days"])
    for label, eyr in [("baseline FE", False), ("with posting-period FE", True)]:
        X = base_X(d_shift, ["log_holders", "round_R3"], eyr)
        m = ols_cluster(d_shift["log_age"], X, d_shift["cell_id"])
        B2_rows.append({"Snapshot shift (days)": delta, "spec": label, "log_holders_coef": round(float(m.params["log_holders"]), 4),
                        "se": round(float(m.bse["log_holders"]), 4), "p": round(float(m.pvalues["log_holders"]), 4),
                        "n": int(m.nobs), "dropped_not_yet_posted": neg})
# Drop records with D<=90 days
d_young = cur[cur["age_days"] > 90].copy()
X = base_X(d_young, ["log_holders", "round_R3"])
m = ols_cluster(d_young["log_age"], X, d_young["cell_id"])
B2_rows.append({"Snapshot shift (days)": "drop records with D<=90 days", "spec": "baseline FE", "log_holders_coef": round(float(m.params["log_holders"]), 4),
                "se": round(float(m.bse["log_holders"]), 4), "p": round(float(m.pvalues["log_holders"]), 4), "n": int(m.nobs)})
B2 = pd.DataFrame(B2_rows)
R["B2_fake_snapshot"] = B2.to_dict("records")
print(B2.to_string(index=False))

# ================================================================ C. Specification curve
print("\n=== C. Specification curve (64 specs) ===")
spec_rows = []
for outcome, cat, form, r3, eyr, var in product(
        ["log_age", "log_age30"], [0, 1], [0, 1], [0, 1], [0, 1], ["log_holders", "single_holder"]):
    d = cur.copy()
    if outcome == "log_age30":
        d["log_age"] = np.log1p(d["age_days"] + 30)
    spec = ([var] + (["round_R3"] if r3 else []))
    X = with_miss_indicators(d, spec)
    if cat: X = X.join(d[CAT_COLS])
    if form: X = X.join(d[FORM_COLS])
    if eyr: X = X.join(d[EYR_COLS])
    X = X.dropna(axis=1, how="any")
    m = ols_cluster(d["log_age"], X, d["cell_id"])
    spec_rows.append({"outcome": outcome, "category_FE": cat, "form_FE": form, "R3_control": r3, "posting_period_FE": eyr,
                      "variable": var, "coef": float(m.params[var]), "se": float(m.bse[var]),
                      "p": float(m.pvalues[var]), "n": int(m.nobs)})
C1 = pd.DataFrame(spec_rows)
share_sig = float((C1["p"] < 0.05).mean())
consistent = np.where(C1["variable"] == "log_holders", C1["coef"] > 0, C1["coef"] < 0)
share_pos = float((C1["coef"] > 0).mean())
share_consistent = float(consistent.mean())
R["C_speccurve"] = {"n_specs": len(C1), "share_p_lt_0.05": round(share_sig, 4), "share_coef_gt_0": round(share_pos, 4),
                    "share_H1_sign_consistent": round(share_consistent, 4),
                    "coef_median": round(float(C1["coef"].median()), 4), "coef_min": round(float(C1["coef"].min()), 4),
                    "coef_max": round(float(C1["coef"].max()), 4),
                    "by_var": {v: {"share_sig": round(float((C1.loc[C1["variable"] == v, "p"] < 0.05).mean()), 4),
                                   "coef_median": round(float(C1.loc[C1["variable"] == v, "coef"].median()), 4)}
                               for v in ["log_holders", "single_holder"]}}
print(f"  {len(C1)} specs: share significant={share_sig:.2%}, share coef>0={share_pos:.2%}, median={C1['coef'].median():.4f}, "
      f"range=[{C1['coef'].min():.4f},{C1['coef'].max():.4f}]")

# fig4 (distinguished by covariate: log_holders = blue circles / single_holder = orange squares; filled = p<0.05)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
Cc = C1.sort_values("coef").reset_index(drop=True)
for v, mk, col in [("log_holders", "o", "#1f6f8b"), ("single_holder", "s", "#b23a48")]:
    sub = Cc[Cc["variable"] == v]
    sig = sub["p"] < 0.05
    axes[0].errorbar(sub["coef"], range(len(Cc))[0:len(sub)] if False else sub.index,
                     xerr=1.96 * sub["se"], fmt="none", ecolor=col, lw=1, alpha=.6)
    axes[0].scatter(sub["coef"], sub.index, marker=mk, c=np.where(sig, col, "white"),
                    edgecolors=col, s=34, zorder=3, label=f"{v} (filled = p<0.05)")
axes[0].axvline(0, color="k", lw=.8, ls="--")
axes[0].set_xlabel("Coefficient (95% CI)")
axes[0].set_title(f"64 specs sorted by coefficient (blue = log_holders, orange = single_holder)\nshare significant {share_sig:.0%}, share consistent with H1' sign {share_consistent:.0%}", fontsize=10)
axes[0].set_ylabel("Specification index"); axes[0].legend(fontsize=8, loc="lower right")
Cp = C1.sort_values("p").reset_index(drop=True)
for v, mk, col in [("log_holders", "o", "#1f6f8b"), ("single_holder", "s", "#b23a48")]:
    sub = Cp[Cp["variable"] == v]
    axes[1].scatter(sub.index, sub["p"], marker=mk, s=22, c=col, alpha=.8, label=v)
axes[1].axhline(0.05, color="k", ls="--", lw=.8)
axes[1].text(len(Cp) * .02, 0.055, "p=0.05", fontsize=8)
axes[1].set_xlabel("Specification index (sorted by p)"); axes[1].set_ylabel("p-value"); axes[1].legend(fontsize=8)
axes[1].set_title("Distribution of p-values", fontsize=10)
fig.suptitle("fig4 | H1' specification curve: outcome (log1p D / log1p(D+30)) x category FE x form FE x R3 x posting-period FE x covariate = 64 specs", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig4_spec_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================================ D. Sample and definition sensitivities
print("\n=== D. Sample and definition sensitivities ===")
D_rows = []
# D1 Exclude R3
nR3 = int((cur["round_R3"] == 1).sum())
d_noR3 = cur[cur["round_R3"] == 0].copy()
X = with_miss_indicators(d_noR3, ["log_holders", "single_holder"]).join(d_noR3[CAT_COLS + FORM_COLS])
m = ols_cluster(d_noR3["log_age"], X, d_noR3["cell_id"])
for v in ["log_holders", "single_holder"]:
    D_rows.append({"Sensitivity": f"exclude R3 records (H1' OLS, {nR3} dropped)", "variable": v,
                   "estimate": round(float(m.params[v]), 4), "se": round(float(m.bse[v]), 4),
                   "p": round(float(m.pvalues[v]), 4), "n": int(m.nobs)})

# D2 Worst-case envelope: the 41 records with missing holders
cur_na = frame_dropped[frame_dropped["tbd"] == 0].copy()
cur_na["age_days"] = (SNAPSHOT - cur_na["initial_posting_date"]).dt.days
cur_na["log_age"] = np.log1p(cur_na["age_days"])
cur_na["cat_group"] = np.where(cur_na["first_cat"].isin(top_cats), cur_na["first_cat"], "other")
cur_na["form_group"] = np.where(cur_na["form_norm"].isin(FORM_KEEP), cur_na["form_norm"], "other")
print(f"  D2: {len(frame_dropped)} records with missing holders, of which Current {len(cur_na)}")
env = {}
for tag, hval in [("all imputed single holder (holders=1)", 1.0), (f"all imputed multi-source (holders=p95={cur['holders'].quantile(.95):.0f})", float(cur["holders"].quantile(.95)))]:
    d_all = pd.concat([cur.drop(columns=CAT_COLS + FORM_COLS + EYR_COLS, errors="ignore"), cur_na], ignore_index=True)
    d_all["holders_imp"] = d_all["holders"].fillna(hval)
    d_all["log_holders"] = np.log1p(d_all["holders_imp"])
    d_all["single_holder"] = (d_all["holders_imp"] <= 1).astype(float)
    cdm = dummies(d_all, "cat_group", "cat"); fdm = dummies(d_all, "form_group", "form")
    d_all = d_all.join(cdm).join(fdm)
    X = with_miss_indicators(d_all, ["log_holders", "round_R3"]).join(d_all[list(cdm.columns) + list(fdm.columns)])
    m = ols_cluster(d_all["log_age"], X, d_all["cell_id"])
    env[tag] = {"coef": round(float(m.params["log_holders"]), 4), "p": round(float(m.pvalues["log_holders"]), 4), "n": int(m.nobs)}
    D_rows.append({"Sensitivity": f"worst-case envelope: {tag}", "variable": "log_holders",
                   "estimate": env[tag]["coef"], "se": round(float(m.bse["log_holders"]), 4),
                   "p": env[tag]["p"], "n": int(m.nobs)})
# H2 version of worst-case: the 41 missing cells all imputed single vs all multi-source (status is observable for missing cells)
env_h2 = {}
all_units = pd.concat([frame, frame_dropped], ignore_index=True)
all_units["cat_group"] = np.where(all_units["first_cat"].isin(top_cats), all_units["first_cat"], "other")
all_units["form_group"] = np.where(all_units["form_norm"].isin(FORM_KEEP), all_units["form_norm"], "other")
all_units["entry_bin"] = pd.cut(all_units["entry_year"], [-np.inf, 2019, 2022, 2023, 2024, 2025, np.inf],
                                labels=["≤2019", "2020-22", "2023", "2024", "2025", "2026"])
cdm2 = dummies(all_units, "cat_group", "cat", prune_outcome="tbd")
fdm2 = dummies(all_units, "form_group", "form", prune_outcome="tbd")
edm2 = dummies(all_units, "entry_bin", "eyr", prune_outcome="tbd")
all_units = all_units.join(cdm2).join(fdm2).join(edm2)
for tag, hval in [("all imputed single (holders=1)", 1.0), (f"all imputed multi-source (holders=p95={frame['holders'].quantile(.95):.0f})", float(frame["holders"].quantile(.95)))]:
    all_units["log_holders"] = np.log1p(all_units["holders"].fillna(hval))
    all_units["single_holder"] = (all_units["holders"].fillna(hval) <= 1).astype(float)
    vs = ["single_holder", "round_R3"] + list(cdm2.columns) + list(fdm2.columns) + list(edm2.columns)
    X = with_miss_indicators(all_units, vs)
    try:
        mH, ameH, aseH = logit_ame_cluster(all_units["tbd"], X, all_units["cell_id"])
        env_h2[tag] = {"AME_single_holder": round(ameH["single_holder"], 4),
                       "p": round(float(mH.pvalues["single_holder"]), 4), "n": int(mH.nobs)}
        D_rows.append({"Sensitivity": f"H2 worst-case: {tag}", "variable": "single_holder",
                       "estimate": env_h2[tag]["AME_single_holder"], "se": round(aseH["single_holder"], 4),
                       "p": env_h2[tag]["p"], "n": int(mH.nobs)})
    except Exception as e:
        env_h2[tag] = {"error": str(e)}
R["D2_worstcase_h2"] = env_h2
print(f"  D2 (H2): {json.dumps(env_h2, ensure_ascii=False)}")
R["D2_worstcase"] = env

# D3 Alternative D definitions
D3_rows = []
for tag, col in [("log(1+D) baseline", "log_age"), ("log(1+D+30)", None), ("months log(1+D/30.44)", None), ("quarterly bins log(1+floor(D/91.31))", None)]:
    d = cur.copy()
    if tag == "log(1+D+30)":
        d["y_alt"] = np.log1p(d["age_days"] + 30)
    elif tag == "months log(1+D/30.44)":
        d["y_alt"] = np.log1p(d["age_days"] / 30.44)
    elif tag.startswith("quarterly"):
        d["y_alt"] = np.log1p(np.floor(d["age_days"] / 91.31))
    else:
        d["y_alt"] = d["log_age"]
    X = with_miss_indicators(d, ["log_holders", "round_R3"]).join(d[CAT_COLS + FORM_COLS])
    m = ols_cluster(d["y_alt"], X, d["cell_id"])
    D3_rows.append({"D definition": tag, "log_holders_coef": round(float(m.params["log_holders"]), 4),
                    "se": round(float(m.bse["log_holders"]), 4), "p": round(float(m.pvalues["log_holders"]), 4)})
    D_rows.append({"Sensitivity": f"D definition: {tag}", "variable": "log_holders", "estimate": D3_rows[-1]["log_holders_coef"],
                   "se": D3_rows[-1]["se"], "p": D3_rows[-1]["p"], "n": len(d)})
D3 = pd.DataFrame(D3_rows)
R["D3_altD"] = D3.to_dict("records")
print(D3.to_string(index=False))
D1df = pd.DataFrame(D_rows)

# ================================================================ E. Posting-period decomposition
print("\n=== E. Posting-period decomposition ===")
# Cohort-specific coefficients: M4 + log_holders x posting-period interactions (base period <=2019)
d = cur.copy()
d["log_h_x_eyr"] = d["log_holders"]
inter_cols = []
for c in EYR_COLS:
    d[f"lh_x_{c}"] = d["log_holders"] * d[c]
    inter_cols.append(f"lh_x_{c}")
X = with_miss_indicators(d, ["log_holders", "round_R3"]).join(d[CAT_COLS + FORM_COLS + EYR_COLS + inter_cols])
m_int = ols_cluster(d["log_age"], X, d["cell_id"])
cohort_rows = []
base_bin = "≤2019"
for c in [base_bin] + EYR_COLS:
    c_lab = c.replace("eyr_", "")   # strip dummy prefix
    if c == base_bin:
        b = float(m_int.params["log_holders"]); se = float(m_int.bse["log_holders"])
    else:
        cc = f"lh_x_{c}"
        b = float(m_int.params["log_holders"] + m_int.params[cc])
        se = float(np.sqrt(m_int.bse["log_holders"]**2 + m_int.bse[cc]**2 +
                           2 * m_int.cov_params().loc["log_holders", cc]))
    n_c = int((cur["entry_bin"].astype(str) == c_lab).sum())
    # Within-cohort simple regression (reported only when n>=40 and within-group variation > 0)
    sub = cur[cur["entry_bin"].astype(str) == c_lab]
    if len(sub) >= 40 and sub["log_holders"].std() > 0:
        ms_ = ols_cluster(sub["log_age"], with_miss_indicators(sub, ["log_holders"]), sub["cell_id"])
        simple_b, simple_p = float(ms_.params["log_holders"]), float(ms_.pvalues["log_holders"])
    else:
        simple_b, simple_p = np.nan, np.nan
    cohort_rows.append({"Entry cohort": c_lab, "n(Current)": n_c, "interaction-model coefficient": round(b, 4), "se": round(se, 4),
                        "p": round(float(2 * (1 - __import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(b / se)))), 4),
                        "within-cohort simple regression coefficient": None if np.isnan(simple_b) else round(simple_b, 4),
                        "within-cohort simple regression p": None if np.isnan(simple_p) else round(simple_p, 4)})
E1 = pd.DataFrame(cohort_rows)
R["E_cohort_decomp"] = E1.to_dict("records")
print(E1.to_string(index=False))

# Full-sample posting period x status composition
comp = (frame.pivot_table(index="entry_bin", columns="tbd", values="unit_id", aggfunc="count", observed=False)
             .fillna(0).astype(int).rename(columns={0: "Current", 1: "TBD"}))
comp["tbd_share"] = (comp["TBD"] / (comp["Current"] + comp["TBD"])).round(3)
R["E_entry_composition"] = comp.to_dict("index")

# fig5
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
ax = axes[0]
x = np.arange(len(comp))
ax.bar(x, comp["Current"], color="#1f6f8b", label="Current (temporary shortage)")
ax.bar(x, comp["TBD"], bottom=comp["Current"], color="#b23a48", label="To Be Discontinued (discontinuation)")
for i, (c_, t_) in enumerate(zip(comp["Current"], comp["TBD"])):
    if t_ > 0:
        ax.text(i, c_ + t_ + 1.5, f"{t_/(c_+t_):.0%}", ha="center", fontsize=8, color="#b23a48")
ax.set_xticks(x); ax.set_xticklabels(comp.index, fontsize=9)
ax.set_ylabel("Records"); ax.legend(fontsize=9)
ax.set_title("Posting period x list-status composition (n=459)", fontsize=11)
ax2 = axes[1]
Eb = E1.dropna(subset=["within-cohort simple regression coefficient"])
ax2.errorbar(Eb["within-cohort simple regression coefficient"], range(len(Eb)),
             xerr=1.96 * Eb["se"], fmt="o", color="#1f6f8b", capsize=3, lw=1.5)
for i, (_, r_) in enumerate(Eb.iterrows()):
    ax2.text(r_["within-cohort simple regression coefficient"] + 0.03, i, f"n={r_['n(Current)']}", va="center", fontsize=8)
ax2.set_yticks(range(len(Eb))); ax2.set_yticklabels(Eb["Entry cohort"])
ax2.axvline(0, color="grey", ls="--", lw=1)
ax2.set_xlabel("Coefficient on log(1+holder count) (95% CI, market-cell clustered)")
ax2.set_title("Entry-cohort decomposition of the H1' coefficient (within-cohort simple regression, n>=40)", fontsize=11)
fig.suptitle("fig5 | The 2025-26 discontinuation wave and the cohort composition of H1'", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig5_entry_cohort.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ================================================================ F. H2 supplementary checks
print("\n=== F. H2 supplements ===")
f2 = frame.copy()
cat_dm2 = dummies(f2, "cat_group", "cat", prune_outcome="tbd")
form_dm2 = dummies(f2, "form_group", "form", prune_outcome="tbd")
eyr_dm2 = dummies(f2, "entry_bin", "eyr", prune_outcome="tbd")
f2 = f2.join(cat_dm2).join(form_dm2).join(eyr_dm2)

def logit_ame_cluster(y, X, cluster):
    Xc = sm.add_constant(X.astype(float))
    m = sm.Logit(y, Xc).fit(disp=0, method="newton", cov_type="cluster", cov_kwds={"groups": cluster}, maxiter=500)
    p = 1 / (1 + np.exp(-(Xc @ m.params)))
    w = np.asarray(p * (1 - p))
    V = m.cov_params().values
    names = list(m.params.index)
    ame, ase = {}, {}
    for c in X.columns:
        if set(X[c].dropna().unique()) <= {0.0, 1.0}:
            X1, X0 = Xc.copy(), Xc.copy(); X1[c], X0[c] = 1.0, 0.0
            f = lambda b: float(np.mean(1 / (1 + np.exp(-(X1 @ b))) - 1 / (1 + np.exp(-(X0 @ b)))))
            ame[c] = f(m.params.values)
            eps = 1e-6; g = np.zeros(len(names))
            for j in range(len(names)):
                bp, bm = m.params.values.copy(), m.params.values.copy()
                bp[j] += eps; bm[j] -= eps
                g[j] = (f(bp) - f(bm)) / (2 * eps)
        else:
            bc = float(m.params[c]); ame[c] = float((w * bc).mean())
            g = (Xc.values * ((w * (1 - w)) * bc)[:, None]).mean(axis=0)
            g[names.index(c)] += float(w.mean())
        ase[c] = float(np.sqrt(g @ V @ g))
    return m, ame, ase

F_rows = []
cat2, form2, eyr2 = list(cat_dm2.columns), list(form_dm2.columns), list(eyr_dm2.columns)
for v in ["log_holders", "single_holder"]:   # consistent with the baseline: the two covariates enter mutually exclusive specifications
    vs = [v, "round_R3"] + cat2 + form2 + eyr2
    X = with_miss_indicators(f2, vs)
    m, ame, ase = logit_ame_cluster(f2["tbd"], X, f2["cell_id"])
    F_rows.append({"check": "H2 M4 full sample (baseline replication)", "variable": v, "AME": round(ame[v], 4),
                   "p": round(float(m.pvalues[v]), 4), "n": int(m.nobs)})
# Exclude R3 (also dropping the all-zero round_R3 column)
f3 = f2[f2["round_R3"] == 0].copy()
for v in ["log_holders", "single_holder"]:
    vs = [v] + cat2 + form2 + eyr2
    X = with_miss_indicators(f3, vs)
    try:
        m3, ame3, ase3 = logit_ame_cluster(f3["tbd"], X, f3["cell_id"])
        F_rows.append({"check": f"H2 excluding R3 records (n_R3={int((f2['round_R3']==1).sum())})", "variable": v,
                       "AME": round(ame3[v], 4), "p": round(float(m3.pvalues[v]), 4), "n": int(m3.nobs)})
    except Exception as e:
        F_rows.append({"check": "H2 excluding R3 records", "variable": v, "AME": "estimation failed", "p": str(e), "n": len(f3)})
# Fisher exact within the 2025/2026 cohorts
for yr in ["2025", "2026"]:
    sub = f2[f2["entry_bin"].astype(str) == yr]
    if sub["single_holder"].nunique() > 1:
        tab = pd.crosstab(sub["single_holder"], sub["tbd"])
        odds, pf = fisher_exact(tab.values)
        F_rows.append({"check": f"Fisher exact within the {yr} cohort", "variable": "single_holder",
                       "AME": f"TBD share single={sub.loc[sub.single_holder==1,'tbd'].mean():.3f} vs multi-source={sub.loc[sub.single_holder==0,'tbd'].mean():.3f}",
                       "p": round(float(pf), 4), "n": int(len(sub))})
F1 = pd.DataFrame(F_rows)
R["F_h2_checks"] = F1.to_dict("records")
print(F1.to_string(index=False))

# ================================================================ Write out
print("\n=== Write out ===")
notes = pd.DataFrame({"Notes": [
    "Table 5 robustness audit (Phase 4). All numbers computed directly from data/shortages_linked.parquet + the raw JSON; zero imputation.",
    "Sample and clustering same as baseline: Current stock n=285 (H1') / all units n=459 (H2); clustering = ingredient-form market cell.",
    "Panel A: the baseline logrank treats records as independent; in fact Current covers only 77 market cells, and single-holder accounts for only 16 cells.",
    "holders is constant within a cell (0 mixed cells) -> the independent variation of the concentration label exists only at the cell level; permutations and collapse are both implemented at the cell level.",
    "Wild bootstrap: Rademacher weights, cell level, percentile-t, B=1999.",
    "Panel B: permutation tests shuffle concentration labels across cells (preserving within-cell structure), B=1000; fake snapshot shifts of +/-90/180 days.",
    "Panel C: specification curve 2^5 = 64 specs (2 outcomes x category FE x form FE x R3 x posting-period FE x 2 covariates).",
    "Panel D: R3 exclusion / 41 OB-FDA-only worst-case envelope / alternative D definitions.",
    "Panel E: H1' cohort decomposition + posting period x status composition. Panel F: supplementary checks for the descriptive positioning of H2.",
    "Significance: * p<0.10, ** p<0.05, *** p<0.01.",
]})
with pd.ExcelWriter(RES / "table5_robustness.xlsx", engine="openpyxl") as w:
    n_eff.to_excel(w, sheet_name="A1_neff", index=False)
    A2.to_excel(w, sheet_name="A2_cell_collapse", index=False)
    A3.to_excel(w, sheet_name="A3_wild_bootstrap", index=False)
    pd.DataFrame([R["A4_perm_logrank"]]).to_excel(w, sheet_name="A4_perm_logrank", index=False)
    pd.DataFrame(B1).to_excel(w, sheet_name="B1_permutation", index=False)
    B2.to_excel(w, sheet_name="B2_fake_snapshot", index=False)
    C1.round(4).to_excel(w, sheet_name="C_spec_curve", index=False)
    D1df.to_excel(w, sheet_name="D_sample_sensitivity", index=False)
    D3.to_excel(w, sheet_name="D3_alt_D", index=False)
    E1.to_excel(w, sheet_name="E_cohort_decomp", index=False)
    comp.reset_index().rename(columns={"entry_bin": "Posting period"}).to_excel(w, sheet_name="E_entry_composition", index=False)
    F1.to_excel(w, sheet_name="F_h2_checks", index=False)
    notes.to_excel(w, sheet_name="Notes", index=False)

# Merge A3 results into R
R["D1_r3"] = D1df.to_dict("records")
(ART / "robustness_results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str))
print("Wrote:", RES / "table5_robustness.xlsx", "|", ART / "robustness_results.json", "| fig4 | fig5")
print("DONE")
