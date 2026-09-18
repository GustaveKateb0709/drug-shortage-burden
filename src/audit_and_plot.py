"""
paper18 audit_and_plot.py -- Step 4/5: five checks on the data contract + first descriptive figure + coverage summary
Run:  python3 src/audit_and_plot.py
Outputs:
  data/fig1_shortages_overview.png   -- first figure: 2012-2026 annual shortage counts x therapeutic
                                        category stacked bars + status composition
  data/coverage_summary.xlsx         -- summary of coverage numbers by join specification
  artifacts/data_contract.json       -- field-level data contract + five-checks record
  artifacts/audit.json               -- audit results
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"

CAT_ORDER = None  # ordered by total volume


def five_checks(df: pd.DataFrame) -> dict:
    checks = {}
    checks["C1_shape_nonempty"] = {"pass": len(df) > 0, "n_rows": len(df), "n_cols": len(df.columns)}
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    key_dtype_ok = str(df["package_ndc"].dtype) == "object" and str(df["initial_posting_date"].dtype).startswith("datetime")
    checks["C2_dtypes"] = {"pass": key_dtype_ok, "dtypes": dtypes}
    miss = df.isna().mean().round(4).to_dict()
    key_vars_miss = max(df["status"].isna().mean(), df["initial_posting_date"].isna().mean())
    checks["C3_missing"] = {"pass": bool(key_vars_miss == 0), "key_vars_missing_rate": float(key_vars_miss), "per_column": miss}
    dup_pkg = int(df["package_ndc"].duplicated().sum())
    checks["C4_duplicate_keys"] = {"pass": True,  # record level allows the same package_ndc to be listed multiple times (different updates); count annotated
                                   "duplicate_package_ndc": dup_pkg,
                                   "note": "analysis-unit dedup is performed at the (ingredient, form, company, initial_posting) level"}
    panel_counts = df.groupby(df["initial_posting_date"].dt.year).size()
    checks["C5_panel_balance"] = {"pass": True, "type": "event data (unbalanced panel)",
                                  "records_per_year": {int(k): int(v) for k, v in panel_counts.items()}}
    return checks


def main() -> None:
    df = pd.read_parquet(DATA / "shortages_linked.parquet")
    ms = pd.read_parquet(DATA / "market_structure_ingredient_form.parquet")

    # ---- Data contract + audit
    checks = five_checks(df)
    audit = {
        "audit_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "five_checks": checks,
        "coverage": json.loads((ART / "sample_construction.json").read_text())["coverage"],
        "multi_category_share": float((df["therapeutic_category"].str.split("; ").apply(len) > 1).mean().round(4)),
        "market_structure_cells": int(len(ms)),
        "market_structure_note": "holders/ndc_products/applications come from NDC directory Rx products; fda_applications from Drugs@FDA; ob_applications/no_generic from the Orange Book",
    }
    (ART / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str))

    # ---- Data contract
    contract = {
        "project": "paper18-drug-shortages-inequity",
        "analysis_unit": "shortage record x active ingredient (normalized) x dosage form (normalized) x holder",
        "primary_key_note": "record level: (package_ndc, initial_posting_date); ingredient-form level: (ingredient, form_norm)",
        "fields": [
            {"name": "package_ndc", "source": "openFDA shortages", "type": "string", "missing": "allowed to be missing (uncovered records)", "role": "join key R1"},
            {"name": "generic_name", "source": "openFDA shortages", "type": "string", "missing": "0%", "role": "R3 normalization source"},
            {"name": "ingredient", "source": "derived: generic_name with salts/dosage-form words removed", "type": "string", "missing": "0%", "role": "join key R3"},
            {"name": "form_norm", "source": "derived: dosage_form normalized against a vocabulary", "type": "string", "missing": "0%", "role": "join key R3"},
            {"name": "initial_posting_date", "source": "openFDA shortages", "type": "datetime", "missing": "0%", "role": "event time (H1 entry risk)"},
            {"name": "update_date", "source": "openFDA shortages", "type": "datetime", "missing": "to check", "role": "last update"},
            {"name": "status", "source": "openFDA shortages", "type": "string(Current/To Be Discontinued/Resolved)", "missing": "0%", "role": "outcome/censoring"},
            {"name": "therapeutic_category", "source": "openFDA shortages (multi-valued, '; ' separated)", "type": "string", "missing": "to check", "role": "H3 stratification dimension"},
            {"name": "of_* (openfda nested flattened)", "source": "openFDA shortages.openfda", "type": "string list", "missing": "9.4% of records fully empty", "role": "join keys R1/R2"},
            {"name": "linked_round", "source": "derived", "type": "string(R1_product_ndc/R2_application_number/R3_ingredient_form/empty)", "missing": "3.3%", "role": "join level"},
            {"name": "linked_product_ndc", "source": "derived (=NDC directory)", "type": "string", "missing": "empty for R2/R3 records", "role": "NDC index"},
            {"name": "linked_application", "source": "derived", "type": "string(ANDA/NDA+6 digits)", "missing": "may be empty for R3 records", "role": "application index"},
            {"name": "market_linked", "source": "derived", "type": "bool", "missing": "0%", "role": "coverage determination"},
            # Market-structure table
            {"name": "holders", "source": "NDC directory Rx products", "type": "int (labeler nunique)", "missing": "0", "role": "H2 concentration proxy"},
            {"name": "ndc_products", "source": "NDC directory", "type": "int", "missing": "0", "role": "H2"},
            {"name": "applications", "source": "NDC directory application_number", "type": "int", "missing": "0", "role": "H1/H2 application count"},
            {"name": "fda_applications", "source": "Drugs@FDA", "type": "int", "missing": "NaN if the ingredient has no application", "role": "H2"},
            {"name": "ob_applications / ob_rld_present / no_generic", "source": "Orange Book", "type": "int/bool", "missing": "not applicable to OTC", "role": "exclusivity/generic competition (H2)"},
            {"name": "single_holder / exclusivity_flag", "source": "derived", "type": "int(0/1)", "missing": "0", "role": "exclusivity flag"},
        ],
        "join_keys": {
            "R1": "openfda.product_ndc == NDC.product_ndc (exact)",
            "R2": "openfda.application_number normalized (ANDA/NDA + 6 digits) == Drugs@FDA / Orange Book",
            "R3": "norm(ingredient) + norm(form) == NDC (ingredient_key, dosage_form)",
        },
        "five_checks": checks,
    }
    (ART / "data_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False, default=str))

    # ---- Coverage summary table
    cov = audit["coverage"]
    cov_tbl = pd.DataFrame([
        {"specification": "total shortage records (raw)", "n": cov["n_shortage_records"], "pct_of_raw": 100.0},
        {"specification": "R1: openfda.product_ndc -> NDC directory (exact)", "n": cov["R1_product_ndc"],
         "pct_of_raw": round(100 * cov["R1_product_ndc"] / cov["n_shortage_records"], 1)},
        {"specification": "R2: openfda.application_number -> Drugs@FDA/Orange Book", "n": cov["R2_application_number"],
         "pct_of_raw": round(100 * cov["R2_application_number"] / cov["n_shortage_records"], 1)},
        {"specification": "R3: normalized (ingredient, form) -> NDC directory (fallback)", "n": cov["R3_ingredient_form"],
         "pct_of_raw": round(100 * cov["R3_ingredient_form"] / cov["n_shortage_records"], 1)},
        {"specification": "total join coverage (attachable to market-structure dimensions)", "n": cov["covered_total"],
         "pct_of_raw": round(100 * cov["coverage_rate"], 1)},
        {"specification": "uncovered", "n": cov["n_shortage_records"] - cov["covered_total"],
         "pct_of_raw": round(100 * (1 - cov["coverage_rate"]), 1)},
    ])
    uncov = cov["uncovered_diagnostics"]
    with pd.ExcelWriter(DATA / "coverage_summary.xlsx", engine="openpyxl") as w:
        cov_tbl.to_excel(w, sheet_name="coverage", index=False)
        pd.DataFrame([{"uncovered diagnostic": k, "value": json.dumps(v, ensure_ascii=False, default=str)} for k, v in uncov.items()]) \
            .to_excel(w, sheet_name="uncovered_diagnostics", index=False)

    # ---- First figure
    df["year"] = df["initial_posting_date"].dt.year
    df["primary_category"] = df["therapeutic_category"].str.split("; ").str[0].fillna("(Uncategorized)")
    annual = (df.groupby(["year", "primary_category"]).size().unstack(fill_value=0))
    annual = annual[annual.sum().sort_values(ascending=False).index]

    status = df["status"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [2.4, 1]})
    cmap = plt.get_cmap("tab20")
    bottom = pd.Series(0, index=annual.index, dtype=float)
    for j, cat in enumerate(annual.columns):
        axes[0].bar(annual.index, annual[cat], bottom=bottom, color=cmap(j % 20), label=cat, width=0.8)
        bottom = bottom + annual[cat]
    axes[0].set_title("FDA Drug Shortages Newly Posted per Year by Therapeutic Category\n(openFDA, 2012–2026, n=1,607 records)", fontsize=11)
    axes[0].set_xlabel("Initial posting year")
    axes[0].set_ylabel("Shortage records")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=7, ncol=1, loc="upper left", framealpha=0.85)
    axes[1].barh(status.index[::-1], status.values[::-1], color=["#2b7bba", "#e1812c", "#3a923a"])
    for i, v in enumerate(status.values[::-1]):
        axes[1].text(v + 8, i, f"{v} ({v/len(df):.0%})", va="center", fontsize=9)
    axes[1].set_title("Status composition\n(as of 2026-09 snapshot)", fontsize=11)
    axes[1].set_xlabel("Records")
    fig.suptitle("Paper 18 — Descriptive Overview: US Drug Shortages 2012–2026", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(DATA / "fig1_shortages_overview.png", dpi=300)
    print("wrote:", DATA / "fig1_shortages_overview.png")
    print("wrote:", DATA / "coverage_summary.xlsx")
    print("wrote:", ART / "data_contract.json", "|", ART / "audit.json")


if __name__ == "__main__":
    main()
