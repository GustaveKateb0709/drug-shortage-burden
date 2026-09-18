"""
paper18 svi_state.py -- Phase 2.5 task 2: state-level SVI aggregation + county-state FIPS crosswalk
Run:  python3 src/svi_state.py

Aggregate data/raw/SVI_2022_US_county.csv (county-level SVI 2022, 3,144 counties) to states:
  - Population-weighted (E_TOTPOP): RPL_THEMES (overall) + RPL_THEME1..4 (four themes)
  - Simple means retained as a comparison
Outputs:
  data/state_svi.parquet        (state_fips, state_abbr, state_name, n_counties, pop_total,
                                 svi_overall_pw, svi_theme1..4_pw, svi_overall_mean, ...)
  data/fips_crosswalk.parquet   (county_fips=STCNTY, county_name, state_fips, state_abbr, state_name, e_totpop, area_sqmi, rpl_themes)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SVI_2022_US_county.csv"
THEMES = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES"]


def main() -> None:
    df = pd.read_csv(RAW, encoding="utf-8-sig", dtype={"FIPS": str, "STCNTY": str, "ST": str})
    df["FIPS"] = df["FIPS"].str.zfill(5)
    df["STCNTY"] = df["STCNTY"].str.zfill(5)
    df["ST"] = df["ST"].str.zfill(2)
    n_raw = len(df)
    df = df.dropna(subset=["E_TOTPOP"])
    df = df[df["E_TOTPOP"] > 0]

    def wavg(g: pd.DataFrame, col: str) -> float:
        m = g[col].notna() & (g[col] >= 0)  # RPL = -999 means missing / not computable
        gg = g[m]
        if gg.empty:
            return float("nan")
        return float((gg[col] * gg["E_TOTPOP"]).sum() / gg["E_TOTPOP"].sum())

    rows = []
    for st, g in df.groupby("ST"):
        rec = {
            "state_fips": st,
            "state_abbr": g["ST_ABBR"].iloc[0],
            "state_name": g["STATE"].iloc[0],
            "n_counties": int(g["FIPS"].nunique()),
            "pop_total": int(g["E_TOTPOP"].sum()),
        }
        for t in THEMES:
            rec[f"{t.lower()}_pw"] = wavg(g, t)
            rec[f"{t.lower()}_mean"] = float(g.loc[g[t].between(0, 1), t].mean())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("state_fips").reset_index(drop=True)
    out.to_parquet(ROOT / "data/state_svi.parquet", index=False)
    print(f"state_svi.parquet: {len(out)} states; "
          f"svi_overall_pw range [{out['rpl_themes_pw'].min():.3f}, {out['rpl_themes_pw'].max():.3f}]")

    cw = df[["STCNTY", "COUNTY", "ST", "ST_ABBR", "STATE", "E_TOTPOP", "AREA_SQMI", "RPL_THEMES"]].copy()
    cw.columns = ["county_fips", "county_name", "state_fips", "state_abbr", "state_name",
                  "e_totpop", "area_sqmi", "rpl_themes"]
    cw.to_parquet(ROOT / "data/fips_crosswalk.parquet", index=False)
    print(f"fips_crosswalk.parquet: {len(cw)} counties (raw input rows={n_raw})")


if __name__ == "__main__":
    main()
