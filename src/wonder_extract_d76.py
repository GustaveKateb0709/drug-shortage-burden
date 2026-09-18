"""
paper18 wonder_extract_d76.py -- WONDER UCD (D76) state-level extraction 2017-2020 (browser session flow)
Run:  python3 src/wonder_extract_d76.py
D76 = Underlying Cause of Death 1999-2020 (final data end at 2020; no final release for 2021-2022 on WONDER).
Differences from D140: the O_ucd single-select, years via the F_D76.V1 multi-select, and B_2=D76.V1-level1.
Outputs: data/raw/wonder/D76_20XX_20YY.tsv + data/state_mortality.parquet (merged over 2003-2020 with a source column)
"""
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAWW = ROOT / "data/raw/wonder"
AB = os.environ.get("BROWSER_AUTOMATION_CLI", "browser-automation-cli")  # browser-automation CLI driving the WONDER web form (see README)
CFG = (ROOT / "src/browser-configure-d76.js").read_text()
SLEEP_Q = 130
WINDOWS = [("2017", "2018"), ("2019", "2020")]


def ab_eval(js: str) -> str:
    r = subprocess.run([AB, "eval", js], capture_output=True, text=True, timeout=300)
    return (r.stdout or "").strip().strip('"')


def ab_wait() -> None:
    subprocess.run([AB, "wait", "--load", "networkidle"], capture_output=True, text=True, timeout=300)


def run_query(y1: str, y2: str) -> None:
    tag = f"{y1}_{y2}"
    tsv = RAWW / f"D76_{tag}.tsv"
    if tsv.exists() and tsv.stat().st_size > 100000:
        print(f"  [{tag}] cache hit", flush=True)
        return
    subprocess.run([AB, "open", "https://wonder.cdc.gov/ucd-icd10.html"], capture_output=True, timeout=300)
    ab_eval("(()=>{const b=[...document.querySelectorAll('input[type=submit]')].find(x=>x.name==='action-I Agree'); b&&b.click(); return 'ok'})()")
    ab_wait()
    cfg = CFG.replace("window.PAPER18_YEARS || ['2017', '2018']", f"['{y1}', '{y2}']")
    out = ab_eval(cfg)
    if "MISS" in out or "years" not in out:
        raise RuntimeError(f"[{tag}] configure failed: {out[:200]}")
    ab_eval("(()=>{const b=[...document.querySelectorAll('#wonderform input[type=submit]')].find(x=>x.name==='action-Send'); b.click(); return 'go'})()")
    for _ in range(40):
        time.sleep(5)
        title = ab_eval("document.title")
        if "Results" in title:
            break
        if "Message" in title:
            body = ab_eval("document.body.innerText.slice(0,400)")
            raise RuntimeError(f"[{tag}] query error: {body[:200]}")
    else:
        raise RuntimeError(f"[{tag}] timeout waiting for results")
    time.sleep(3)
    action = ab_eval("document.querySelectorAll('form')[1].action")
    payload = ab_eval("(()=>{const f=document.querySelectorAll('form')[1]; const fd=new FormData(f); fd.set('O_export-format','tsv'); fd.append('action-Export','Download'); return new URLSearchParams(fd).toString()})()")
    cookies = ab_eval("document.cookie")
    open("/tmp/d76_export_post.txt", "w").write(payload)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "--max-time", "900",
         "--data", "@/tmp/d76_export_post.txt",
         "-H", "Content-Type: application/x-www-form-urlencoded",
         "-H", f"Cookie: {cookies}",
         "-H", "Referer: https://wonder.cdc.gov/controller/datarequest/D76",
         action.replace("&amp;", "&"), "-o", str(tsv), "-w", "%{http_code} %{size_download}"],
        capture_output=True, text=True, timeout=900,
    )
    print(f"  [{tag}] export {r.stdout.strip()} -> {tsv.name}", flush=True)
    if r.stdout.strip().split()[1] == "0":
        raise RuntimeError(f"[{tag}] export failed")


def main() -> None:
    for i, (y1, y2) in enumerate(WINDOWS):
        if i:
            print(f"  throttle {SLEEP_Q}s ...", flush=True)
            time.sleep(SLEEP_Q)
        try:
            run_query(y1, y2)
        except Exception as e:  # noqa: BLE001
            print(f"  [{y1}_{y2}] FAILED: {e}", flush=True)
    # Parse the D76 TSVs and merge with D140
    frames = []
    for tsv in sorted(RAWW.glob("D76_*.tsv")):
        df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
        df = df[df["State"].ne("")]
        frames.append(df)
        print(f"  parse {tsv.name}: {len(df)} rows", flush=True)
    if frames:
        d76 = pd.concat(frames, ignore_index=True)
        code_col = [c for c in d76.columns if "113" in c and c.endswith("Code")][0]
        lab_col = [c for c in d76.columns if "113" in c and not c.endswith("Code")][0]
        d76 = d76.rename(columns={"State Code": "state_fips", code_col: "cause_group", lab_col: "cause_group_label"})
        d76["cause_group_label"] = d76["cause_group_label"].str.lstrip("#")
        d76["year"] = pd.to_numeric(d76["Year"], errors="coerce").astype("Int64")
        d76["suppressed"] = d76["Deaths"].str.contains("Suppressed", case=False, na=False)
        for c in ("Deaths", "Population"):
            d76[c] = pd.to_numeric(d76[c].str.replace(",", "", regex=False).str.extract(r"([\d.]+)")[0], errors="coerce")
        d76["crude_rate"] = pd.to_numeric(d76["Crude Rate"].str.extract(r"([\d.]+)")[0], errors="coerce")
        d76 = d76[["state_fips", "State", "year", "cause_group", "cause_group_label",
                   "Deaths", "Population", "crude_rate", "suppressed"]].copy()
        d76.columns = ["state_fips", "state_name", "year", "cause_group", "cause_group_label",
                       "deaths", "population", "crude_rate", "suppressed"]
        d76["source"] = "UCD-ICD10-2020(D76)"
        d76 = d76.dropna(subset=["state_fips", "year"])
        print(f"D76 rows={len(d76)} years={d76.year.min()}-{d76.year.max()}", flush=True)
        # Merge
        old = pd.read_parquet(ROOT / "data/state_mortality.parquet")
        old["source"] = "CMF-ICD10-2016(D140)"
        merged = pd.concat([old, d76], ignore_index=True)
        merged.to_parquet(ROOT / "data/state_mortality.parquet", index=False)
        print(f"merged state_mortality.parquet rows={len(merged)} states={merged.state_fips.nunique()} "
              f"years={merged.year.min()}-{merged.year.max()} suppressed={int(merged['suppressed'].sum())}", flush=True)


if __name__ == "__main__":
    main()
