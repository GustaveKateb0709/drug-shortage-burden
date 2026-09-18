"""
paper18 build_pipeline.py -- Step 2/3: sample construction + joins + measured coverage
Run:  python3 src/build_pipeline.py

Analysis unit: product/ingredient level (shortage record x active ingredient x dosage form x holder).
Join strategy (cascading fallback by priority; coverage = share of shortage records attachable to market-structure dimensions):
  R1: openfda.product_ndc  <-> NDC directory product_ndc (exact)
  R2: openfda.application_number <-> Drugs@FDA / Orange Book (exact)
  R3: normalized generic_name ingredient extraction + normalized dosage form <-> NDC directory (ingredient, form) cells
Outputs:
  data/shortages_linked.parquet     -- record-level shortages + join results
  data/market_structure_ingredient_form.parquet -- (ingredient, form) market structure: holder count / application count / exclusivity
  data/market_structure_ingredient.parquet      -- ingredient-level summary
  artifacts/sample_construction.json -- sample-construction log + coverage
Memory constraint: peak <4GB; files loaded one at a time and released immediately after extraction.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DATA = ROOT / "data"
ART = ROOT / "artifacts"
if not DATA.exists():
    DATA.mkdir(parents=True)
if not ART.exists():
    ART.mkdir(parents=True)

log = []  # sample construction log


def rec(step: str, n: int, note: str = "") -> None:
    log.append({"step": step, "n_records": int(n), "note": note})
    print(f"  [{step}] n={n} {note}")


# ---------------------------------------------------------------- Normalization
FORM_WORDS = {
    "injection", "injectable", "inject", "solution", "tablet", "tablets",
    "capsule", "capsules", "suspension", "cream", "ointment", "gel",
    "spray", "drops", "syrup", "elixir", "granules", "powder", "kit",
    "inhalation", "inhaler", "suppository", "suppositories", "implant",
    "patch", "film", "pellets", "emulsion", "shampoo", "lotion", "aerosol",
    "intravenous", "intramuscular", "subcutaneous", "ophthalmic", "otic",
    "oral", "topical", "buccal", "sublingual", "rectal", "vaginal",
}
SALT_TOKENS = {
    "sodium", "potassium", "calcium", "hydrochloride", "hcl", "sulfate",
    "sulphate", "phosphate", "acetate", "maleate", "citrate", "tartrate",
    "besylate", "besilate", "fumarate", "succinate", "mesylate", "nitrate",
    "chloride", "bromide", "iodide", "oxalate", "pamoate", "stearate",
    "lactate", "gluconate", "carbonate", "bicarbonate", "bitartrate",
    "dihydrochloride", "monohydrate", "disodium", "monosodium",
    "hydrobromide", "propiotate", "sodiumphosphate",
}
SUFFIX_RE = re.compile(r"(extended[- ]release|delayed[- ]release|sustained[- ]release|"
                       r"controlled[- ]release|for[- ](injection|suspension|solution|oral use)|"
                       r"\ber\b|\bxr\b|\bcr\b|\bsr\b|\bla\b)", re.I)


def norm_text(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def extract_ingredient_form(generic_name: str) -> tuple[str, str]:
    """'Dexamethasone Sodium Phosphate Injection' -> ('dexamethasone', 'injection')"""
    s = norm_text(generic_name)
    s = SUFFIX_RE.sub(" ", s)
    toks = [t for t in s.split() if t]
    form = ""
    # Strip dosage-form words from the tail
    while toks and toks[-1] in FORM_WORDS:
        form = toks[-1] if not form else form
        toks.pop()
    form = form or "unspecified"
    # Strip salt bases: keep the main ingredient tokens
    keep = [t for t in toks if t not in SALT_TOKENS]
    ing = " ".join(keep) if keep else " ".join(toks)
    return norm_text(ing), form


def norm_ndc_form(form: str) -> str:
    """NDC directory dosage form -> vocabulary normalization (aligned with the form extracted on the shortage side)"""
    s = norm_text(form)
    if not s:
        return "unspecified"
    if "injection" in s or "injectable" in s:
        return "injection"
    if "tablet" in s:
        return "tablet"
    if "capsule" in s:
        return "capsule"
    if "solution" in s:
        return "solution"
    if "suspension" in s:
        return "suspension"
    if "cream" in s:
        return "cream"
    if "ointment" in s:
        return "ointment"
    if "gel" in s:
        return "gel"
    if "spray" in s:
        return "spray"
    if "drop" in s:
        return "drops"
    if "suppository" in s:
        return "suppository"
    if "patch" in s:
        return "patch"
    if "kit" in s:
        return "kit"
    return s


def appnum_norm(a: str) -> str | None:
    """'ANDA087702' -> ('ANDA','087702'); 'NDA002386'->('NDA','002386'); plain digits -> None"""
    m = re.match(r"(ANDA|NDA)(\d+)$", str(a).upper())
    return f"{m.group(1)}{int(m.group(2)):06d}" if m else None


def main() -> None:
    print("=== Step A: Parse shortages data ===")
    sh = json.load(open(RAW / "drug-shortages-0001-of-0001.json"))["results"]
    rec("0. raw shortage records", len(sh))
    rows = []
    for r in sh:
        of = r.get("openfda") or {}
        ip = r.get("initial_posting_date")
        ud = r.get("update_date")
        rows.append({
            "package_ndc": r.get("package_ndc"),
            "generic_name": r.get("generic_name"),
            "dosage_form": r.get("dosage_form"),
            "presentation": r.get("presentation"),
            "company_name": r.get("company_name"),
            "status": r.get("status"),
            "availability": r.get("availability"),
            "update_type": r.get("update_type"),
            "initial_posting_date": pd.to_datetime(ip, format="%m/%d/%Y", errors="coerce"),
            "update_date": pd.to_datetime(ud, format="%m/%d/%Y", errors="coerce"),
            "therapeutic_category": "; ".join(r.get("therapeutic_category") or []),
            "of_product_ndc": list(of.get("product_ndc") or []),
            "of_application_number": list(of.get("application_number") or []),
            "of_brand_name": list(of.get("brand_name") or []),
            "of_manufacturer_name": list(of.get("manufacturer_name") or []),
            "of_substance_name": list(of.get("substance_name") or []),
        })
    df = pd.DataFrame(rows)
    ing_form = df["generic_name"].map(extract_ingredient_form)
    df["ingredient"] = [t[0] for t in ing_form]
    df["form_norm"] = [t[1] for t in ing_form]
    rec("1. parsed (dates typed, openfda flattened)", len(df),
        f"openfda non-empty: {(df['of_product_ndc'].str.len() > 0).sum()}")
    n_bad_date = int(df["initial_posting_date"].isna().sum())
    rec("1a. unparseable initial_posting_date (kept, flagged)", len(df), f"n_bad={n_bad_date}")
    print("status:", dict(Counter(df["status"])))
    print("date range:", df["initial_posting_date"].min(), "->", df["initial_posting_date"].max())
    dup_pkg = int(df["package_ndc"].duplicated().sum())
    rec("1b. duplicate package_ndc (kept at record level, dedup on analysis unit later)",
        len(df), f"dup_pkg={dup_pkg}")

    # ------------------------------------------------ NDC directory (largest in memory; process first)
    print("=== Step B: Parse NDC directory -> product-level table ===")
    ndc = json.load(open(RAW / "drug-ndc-0001-of-0001.json"))["results"]
    print(f"  NDC products: {len(ndc)}")
    nrows = []
    for r in ndc:
        ings = sorted({norm_text(i.get("name", "")) for i in (r.get("active_ingredients") or []) if i.get("name")})
        ing_key = "; ".join(x for x in ings if x)
        nrows.append({
            "product_ndc": r.get("product_ndc"),
            "ingredient_key": ing_key,
            "dosage_form": norm_ndc_form(r.get("dosage_form") or ""),
            "labeler_name": (r.get("labeler_name") or "").upper(),
            "application_number": appnum_norm(r.get("application_number") or "") or (r.get("application_number") or ""),
            "marketing_category": r.get("marketing_category"),
            "marketing_start_date": pd.to_datetime(r.get("marketing_start_date"), format="%Y%m%d", errors="coerce"),
            "listing_expiration_date": pd.to_datetime(r.get("listing_expiration_date"), format="%Y%m%d", errors="coerce"),
        })
    ndc_df = pd.DataFrame(nrows)
    del ndc, nrows
    rec("B1. NDC products parsed", len(ndc_df), "human+otc all; filter Rx only below")
    ndc_rx = ndc_df[ndc_df["marketing_category"].fillna("").str.contains("PRESCRIPTION|NDA|ANDA|BLA", regex=True, na=False)].copy()
    rec("B2. NDC Rx products (marketing_category filter)", len(ndc_rx),
        f"OTC/others dropped: {len(ndc_df) - len(ndc_rx)}")
    ndc_rx = ndc_rx[ndc_rx["listing_expiration_date"].isna() | (ndc_rx["listing_expiration_date"] >= "2012-01-01")]
    rec("B3. NDC Rx products active-or-active-in-window", len(ndc_rx),
        "listing_expiration >= 2012-01-01 or missing")

    # Market-structure table @ (ingredient_key, dosage_form)
    ms_if = (ndc_rx.groupby(["ingredient_key", "dosage_form"])
             .agg(holders=("labeler_name", "nunique"),
                  ndc_products=("product_ndc", "nunique"),
                  applications=("application_number", lambda s: s[s != ""].nunique()))
             .reset_index())
    ms_if["single_holder"] = (ms_if["holders"] <= 1).astype(int)
    rec("B4. market-structure cells (ingredient_key, form) from NDC", len(ms_if))

    # ------------------------------------------------ Drugs@FDA
    print("=== Step C: Parse Drugs@FDA -> application-level table ===")
    dfs_ = json.load(open(RAW / "drug-drugsfda-0001-of-0001.json"))["results"]
    print(f"  Drugs@FDA applications: {len(dfs_)}")
    app_rows, prod_rows = [], []
    for r in dfs_:
        app = appnum_norm(r.get("application_number") or "")
        for p in r.get("products") or []:
            ings = sorted({norm_text(i.get("name", "")) for i in (p.get("active_ingredients") or []) if i.get("name")})
            prod_rows.append({
                "application_number": r.get("application_number") or "",
                "ingredient_key": "; ".join(x for x in ings if x),
                "dosage_form": norm_ndc_form(p.get("dosage_form") or ""),
                "brand_name": p.get("brand_name"),
                "reference_drug": p.get("reference_drug"),
            })
        app_rows.append({"application_number": r.get("application_number") or "",
                         "sponsor_name": (r.get("sponsor_name") or "").upper()})
    fda_prod = pd.DataFrame(prod_rows)
    fda_app = pd.DataFrame(app_rows).drop_duplicates("application_number")
    del dfs_, app_rows, prod_rows
    rec("C1. Drugs@FDA products parsed", len(fda_prod))
    fda_prod_key = fda_prod.assign(
        app_key=fda_prod["application_number"].map(appnum_norm))
    ms_app = (fda_prod_key.groupby(["ingredient_key", "dosage_form"])
              .agg(fda_applications=("app_key", "nunique"),
                   fda_sponsors=("app_key", lambda s: fda_app.set_index("application_number")
                                  .loc[[x for x in s.dropna() if x], "sponsor_name"].nunique()
                                  if len([x for x in s.dropna() if x]) else 0))
              .reset_index())
    rec("C2. market-structure cells (ingredient_key, form) from Drugs@FDA", len(ms_app))

    # ------------------------------------------------ Orange Book
    print("=== Step D: Parse Orange Book -> application-level + RLD/exclusivity ===")
    ob = json.load(open(RAW / "drug-orangebook-0001-of-0001.json"))["results"]
    print(f"  Orange Book application-product rows: {len(ob)}")
    ob_rows = []
    for r in ob:
        for p in r.get("products") or []:
            ings = sorted({norm_text(i.get("name", "")) for i in (p.get("active_ingredients") or []) if i.get("name")})
            ob_rows.append({
                "app_letter": p.get("application_type"),
                "app_number": p.get("application_number"),
                "ingredient_key": "; ".join(x for x in ings if x),
                "dosage_form": norm_ndc_form(p.get("dosage_form") or ""),
                "marketing_status": p.get("marketing_status"),
                "reference_listed_drug": bool(p.get("reference_listed_drug")),
                "applicant": (p.get("application_full_name") or p.get("application_name") or "").upper(),
            })
    ob_df = pd.DataFrame(ob_rows)
    del ob, ob_rows
    rec("D1. Orange Book product rows parsed", len(ob_df))
    ob_rx = ob_df[ob_df["marketing_status"].fillna("").str.contains("PRESCRIPTION", na=False)]
    rec("D2. Orange Book Rx rows", len(ob_rx))
    ms_ob = (ob_rx.groupby(["ingredient_key", "dosage_form"])
             .agg(ob_applications=("app_number", "nunique"),
                  ob_rld_present=("reference_listed_drug", "max"))
             .reset_index())
    # Exclusivity (no ANDA beyond the RLD = approximation of no generic competition; built at the ingredient-form level)
    ms_ob["no_generic"] = ((ob_rx.groupby(["ingredient_key", "dosage_form"])["app_letter"]
                            .apply(lambda s: (s == "A").sum() == 0)).astype(int).values)

    # ------------------------------------------------ Joins
    print("=== Step E: Joins (3-round strategy) ===")
    df["linked_round"] = ""
    df["linked_product_ndc"] = ""
    df["linked_application"] = ""
    df["linked_ingredient_key"] = ""
    ndc_index = ndc_rx.set_index("product_ndc")
    fda_index = fda_prod_key.set_index("application_number")
    ob_index_app = ob_df.set_index(ob_df["app_letter"].fillna("") + ob_df["app_number"].fillna(""))

    def to_appkeys(lst):
        return [appnum_norm(a) for a in lst if appnum_norm(a)]

    n_r1 = n_r2 = n_r3 = 0
    for i, row in df.iterrows():
        # R1: openfda.product_ndc -> NDC
        hit = None
        for pndc in row["of_product_ndc"]:
            if pndc in ndc_index.index:
                hit = pndc
                break
        if hit is not None:
            rec_ = ndc_index.loc[hit]
            rec_ = rec_.iloc[0] if isinstance(rec_, pd.DataFrame) else rec_
            df.at[i, "linked_round"] = "R1_product_ndc"
            df.at[i, "linked_product_ndc"] = hit
            df.at[i, "linked_ingredient_key"] = rec_["ingredient_key"]
            df.at[i, "linked_application"] = rec_["application_number"]
            n_r1 += 1
            continue
        # R2: openfda.application_number -> Drugs@FDA / Orange Book
        hit = None
        for a in to_appkeys(row["of_application_number"]):
            if a in fda_index.index:
                hit = ("fda", a)
                break
        if hit is None:
            for a in row["of_application_number"]:
                a2 = appnum_norm(a)
                if a2 and a2 in ob_index_app.index:
                    hit = ("ob", a2)
                    break
        if hit is not None:
            src, a = hit
            if src == "fda":
                rec_ = fda_index.loc[a]
                rec_ = rec_.iloc[0] if isinstance(rec_, pd.DataFrame) else rec_
                df.at[i, "linked_ingredient_key"] = rec_["ingredient_key"]
            else:
                rec_ = ob_index_app.loc[a]
                rec_ = rec_.iloc[0] if isinstance(rec_, pd.DataFrame) else rec_
                df.at[i, "linked_ingredient_key"] = rec_["ingredient_key"]
            df.at[i, "linked_round"] = "R2_application_number"
            df.at[i, "linked_application"] = a
            n_r2 += 1
            continue
        # R3: normalized (ingredient, form) -> NDC cell
        cand = ms_if[(ms_if["ingredient_key"] == row["ingredient"]) & (ms_if["dosage_form"] == row["form_norm"])]
        if cand.empty and row["ingredient"]:
            # Ingredient-string containment match (shortage names often carry salt/brand words)
            mask = ms_if["ingredient_key"].str.contains(re.escape(row["ingredient"]), na=False) \
                if len(row["ingredient"]) > 3 else pd.Series(False, index=ms_if.index)
            cand = ms_if[mask & (ms_if["dosage_form"] == row["form_norm"])]
        if not cand.empty:
            best = cand.sort_values("ndc_products", ascending=False).iloc[0]
            df.at[i, "linked_round"] = "R3_ingredient_form"
            df.at[i, "linked_ingredient_key"] = best["ingredient_key"]
            n_r3 += 1
    df["market_linked"] = (df["linked_round"] != "")
    n_cov = int(df["market_linked"].sum())
    print(f"  R1(product_ndc)={n_r1}  R2(application)={n_r2}  R3(norm ing+form)={n_r3}")
    print(f"  COVERAGE = {n_cov}/{len(df)} = {n_cov / len(df):.1%}")

    # Final (ingredient_key, form)-level market-structure table: merge NDC + Drugs@FDA + OB
    ms = ms_if.merge(ms_app, on=["ingredient_key", "dosage_form"], how="outer") \
              .merge(ms_ob[["ingredient_key", "dosage_form", "ob_applications", "ob_rld_present", "no_generic"]],
                     on=["ingredient_key", "dosage_form"], how="outer")
    for c in ["holders", "ndc_products", "applications"]:
        ms[c] = ms[c].fillna(0).astype(int)
    ms["exclusivity_flag"] = ((ms["applications"].fillna(0) <= 1) & (ms["ob_applications"].fillna(0) <= 1)).astype(int)

    # ------------------------------------------------ Persist
    df.to_parquet(DATA / "shortages_linked.parquet", index=False)
    ms.to_parquet(DATA / "market_structure_ingredient_form.parquet", index=False)
    ms.groupby("ingredient_key").agg(
        max_holders=("holders", "max"), max_ndc_products=("ndc_products", "max"),
        min_holders=("holders", "min"), any_single_holder=("single_holder", "max"),
        ob_applications=("ob_applications", "max"), no_generic=("no_generic", "max"),
        exclusivity_flag=("exclusivity_flag", "max")).reset_index() \
        .to_parquet(DATA / "market_structure_ingredient.parquet", index=False)

    # Coverage distribution: what uncovered records look like
    uncov = df[~df["market_linked"]]
    uncov_diag = {
        "n_uncovered": int(len(uncov)),
        "openfda_empty": int((uncov["of_product_ndc"].str.len() == 0).sum()),
        "status_dist": dict(Counter(uncov["status"].fillna("NA"))),
        "form_dist_top10": dict(Counter(uncov["form_norm"]).most_common(10)),
    }
    coverage = {
        "n_shortage_records": int(len(df)),
        "R1_product_ndc": n_r1,
        "R2_application_number": n_r2,
        "R3_ingredient_form": n_r3,
        "covered_total": n_cov,
        "coverage_rate": round(n_cov / len(df), 4),
        "uncovered_diagnostics": uncov_diag,
    }
    sample = {
        "project": "paper18-drug-shortages-inequity",
        "built_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "source_files": {
            "shortages": "drug-shortages-0001-of-0001.json",
            "ndc": "drug-ndc-0001-of-0001.json",
            "drugsfda": "drug-drugsfda-0001-of-0001.json",
            "orangebook": "drug-orangebook-0001-of-0001.json",
        },
        "sample_construction_log": log,
        "status_distribution": dict(Counter(df["status"].fillna("NA"))),
        "date_range": [str(df["initial_posting_date"].min().date()), str(df["initial_posting_date"].max().date())],
        "coverage": coverage,
        "join_strategy": [
            "R1: openfda.product_ndc -> NDC directory product_ndc (exact)",
            "R2: openfda.application_number -> Drugs@FDA / Orange Book application (normalized NDA/ANDA+6d)",
            "R3: normalized ingredient (salt-stripped, form words stripped) + normalized dosage form -> NDC (ingredient, form) cell",
        ],
    }
    (ART / "sample_construction.json").write_text(json.dumps(sample, indent=2, ensure_ascii=False, default=str))
    print("wrote:", DATA / "shortages_linked.parquet", "|", DATA / "market_structure_ingredient_form.parquet")
    print("wrote:", ART / "sample_construction.json")
    print(json.dumps(coverage, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
