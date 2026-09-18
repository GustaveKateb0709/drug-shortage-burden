"""
paper18 wonder_extract_ch_age.py -- Phase 2.5 addendum: chapter-level panel + Chapter A age cells
Run:  python3 src/wonder_extract_ch_age.py

Prompted by the MNAR question raised during the robustness review:
  1) Re-run D140 2003-2016 at chapter level (B_3=D140.V2-level1) -- chapter-level infectious-disease
     death counts are large, so near-zero suppression is expected; isomorphic to the D76 2017-2020
     chapter data -> a complete 2003-2020 chapter panel.
  2) Chapter A (A00-B99 as a filter, not a grouping) state x year x age-group cells (all 14 years in
     one query).
Outputs:
  data/raw/wonder/D140ch_20XX_20YY.tsv  /  D140ageA_2003_2016.tsv
  data/state_mortality_chapter.parquet  /  data/state_mortality_ageA.parquet
"""
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAWW = ROOT / "data/raw/wonder"
SRC = ROOT / "src"
AB = os.environ.get("BROWSER_AUTOMATION_CLI", "browser-automation-cli")  # browser-automation CLI driving the WONDER web form (see README)
SLEEP_Q = 130
WINDOWS = [(f"{y}", f"{y + 1}") for y in range(2003, 2017, 2)]


def ab_eval(js: str) -> str:
    r = subprocess.run([AB, "eval", js], capture_output=True, text=True, timeout=300)
    return (r.stdout or "").strip().strip('"')


def ab_wait() -> None:
    subprocess.run([AB, "wait", "--load", "networkidle"], capture_output=True, text=True, timeout=300)


def run_query(tag: str, tsv: Path, cfg_file: Path, years: list[str]) -> None:
    if tsv.exists() and tsv.stat().st_size > 30000:
        print(f"  [{tag}] cache hit", flush=True)
        return
    subprocess.run([AB, "open", "https://wonder.cdc.gov/cmf-icd10.html"], capture_output=True, timeout=300)
    ab_eval("(()=>{const b=[...document.querySelectorAll('input[type=submit]')].find(x=>x.name==='action-I Agree'); b&&b.click(); return 'ok'})()")
    ab_wait()
    cfg = cfg_file.read_text().replace(
        "window.PAPER18_YEARS || ['2003', '2004']", str(years))
    out = ab_eval(cfg)
    if "MISS" in out or "years" not in out:
        raise RuntimeError(f"[{tag}] configure failed: {out[:200]}")
    ab_eval("(()=>{const b=[...document.querySelectorAll('#wonderform input[type=submit]')].find(x=>x.name==='action-Send'); b.click(); return 'go'})()")
    for _ in range(50):
        time.sleep(5)
        title = ab_eval("document.title")
        if "Results" in title:
            break
        if "Message" in title:
            raise RuntimeError(f"[{tag}] query error: {ab_eval('document.body.innerText.slice(0,300)')[:200]}")
    else:
        raise RuntimeError(f"[{tag}] timeout waiting for results")
    time.sleep(3)
    action = ab_eval("document.querySelectorAll('form')[1].action")
    payload = ab_eval("(()=>{const f=document.querySelectorAll('form')[1]; const fd=new FormData(f); fd.set('O_export-format','tsv'); fd.append('action-Export','Download'); return new URLSearchParams(fd).toString()})()")
    cookies = ab_eval("document.cookie")
    open("/tmp/ch_export_post.txt", "w").write(payload)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "--max-time", "900",
         "--data", "@/tmp/ch_export_post.txt",
         "-H", "Content-Type: application/x-www-form-urlencoded",
         "-H", f"Cookie: {cookies}",
         "-H", "Referer: https://wonder.cdc.gov/controller/datarequest/D140",
         action.replace("&amp;", "&"), "-o", str(tsv), "-w", "%{http_code} %{size_download}"],
        capture_output=True, text=True, timeout=900,
    )
    print(f"  [{tag}] export {r.stdout.strip()} -> {tsv.name}", flush=True)
    if r.stdout.strip().split()[1] == "0":
        raise RuntimeError(f"[{tag}] export failed")


def parse_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df = df[df["State"].ne("")]
    code_col = [c for c in df.columns if ("Chapter" in c or "113" in c or "Age" in c) and c.endswith("Code")
                and "State" not in c and "Year" not in c][0]
    lab_col = [c for c in df.columns if ("Chapter" in c or "113" in c or "Age" in c) and not c.endswith("Code")
               and "State" not in c and "Year" not in c][0]
    aar_col = [c for c in df.columns if "adjust" in c.lower()]
    df = df.rename(columns={"State Code": "state_fips", code_col: "cause_code", lab_col: "cause_label",
                            **({aar_col[0]: "aar_raw"} if aar_col else {})})
    df["cause_label"] = df["cause_label"].str.lstrip("#")
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["suppressed"] = df["Deaths"].str.contains("Suppressed", case=False, na=False)
    df["aar_suppressed"] = (df["aar_raw"].str.contains("Suppressed", case=False, na=False)
                            if "aar_raw" in df else df["suppressed"])
    for c in ("Deaths", "Population"):
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False).str.extract(r"([\d.]+)")[0], errors="coerce")
    df["crude_rate"] = pd.to_numeric(df["Crude Rate"].str.extract(r"([\d.]+)")[0], errors="coerce")
    df["aar"] = (pd.to_numeric(df["aar_raw"].str.extract(r"([\d.]+)")[0], errors="coerce")
                 if "aar_raw" in df else pd.NA)
    out = df[["state_fips", "State", "year", "cause_code", "cause_label", "Deaths", "Population",
              "crude_rate", "aar", "suppressed", "aar_suppressed"]].copy()
    out.columns = ["state_fips", "state_name", "year", "cause_code", "cause_label", "deaths", "population",
                   "crude_rate", "aar", "suppressed", "aar_suppressed"]
    return out.dropna(subset=["state_fips", "year"])


def main() -> None:
    # ---- 1) Chapter level 2003-2016 (7 windows, with AAR)
    for i, (y1, y2) in enumerate(WINDOWS):
        if i:
            print(f"  throttle {SLEEP_Q}s ...", flush=True)
            time.sleep(SLEEP_Q)
        try:
            run_query(f"D140ch_{y1}_{y2}", RAWW / f"D140ch_{y1}_{y2}.tsv",
                      SRC / "browser-configure-chapter.js", [y1, y2])
        except Exception as e:  # noqa: BLE001
            print(f"  [D140ch_{y1}_{y2}] FAILED: {e}", flush=True)
    # ---- 2) Chapter A age cells (all 14 years in one query)
    print(f"  throttle {SLEEP_Q}s ...", flush=True)
    time.sleep(SLEEP_Q)
    try:
        run_query("D140ageA_2003_2016", RAWW / "D140ageA_2003_2016.tsv",
                  SRC / "browser-configure-ageA.js", [str(y) for y in range(2003, 2017)])
    except Exception as e:  # noqa: BLE001
        print(f"  [D140ageA] FAILED: {e}", flush=True)
    # ---- 3) Parse and persist
    frames = [parse_tsv(f) for f in sorted(RAWW.glob("D140ch_2*.tsv"))]
    if frames:
        ch = pd.concat(frames, ignore_index=True)
        ch["granularity"] = "chapter"
        ch["source"] = "CMF-ICD10-2016(D140)"
        ch.to_parquet(ROOT / "data/state_mortality_chapter.parquet", index=False)
        a_rows = ch[ch.cause_code == "A00-B99"]
        print(f"chapter panel rows={len(ch)} states={ch.state_fips.nunique()} "
              f"years={ch.year.min()}-{ch.year.max()} suppressed_share={ch.suppressed.mean():.4f}", flush=True)
        print(f"  A00-B99 rows={len(a_rows)} suppressed={int(a_rows.suppressed.sum())}", flush=True)
    age = RAWW / "D140ageA_2003_2016.tsv"
    if age.exists() and age.stat().st_size > 30000:
        ag = parse_tsv(age)
        ag["granularity"] = "age_group"
        ag["source"] = "CMF-ICD10-2016(D140)"
        ag.to_parquet(ROOT / "data/state_mortality_ageA.parquet", index=False)
        print(f"ageA rows={len(ag)} suppressed_share={ag.suppressed.mean():.4f} "
              f"ages={ag.cause_code.nunique()}", flush=True)


if __name__ == "__main__":
    main()
