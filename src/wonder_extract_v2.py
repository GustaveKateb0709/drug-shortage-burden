"""
paper18 wonder_extract_v2.py -- CDC WONDER state-level mortality extraction (browser session flow v2, final)
Run:  python3 src/wonder_extract_v2.py. Requires a browser-automation CLI (set via the BROWSER_AUTOMATION_CLI environment variable; see README) and curl

Run:  python3 src/wonder_extract_v2.py

Background (all verified empirically):
  - The WONDER XML API (D76/D140) is national level only for "web services" -> state-level data requires the UI flow
  - UI flow: browser form submission (State x Year x ICD-10 113 Groups), queried 2 years at a time (larger requests return 'too much data')
  - Export: the results page's form[1] (jsessionid session-bound) + action-Export=Download, O_export-format=tsv
  - The session cookie can be read from document.cookie (JSESSIONID is embedded in the URL path)
  - Official throttling: queries at most once every 2 minutes; this script waits 130s between queries
Outputs:
  data/raw/wonder/D140_20XX_20YY.tsv  raw TSV snapshots (7 windows)
  data/state_mortality.parquet
"""
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAWW = ROOT / "data/raw/wonder"
RAWW.mkdir(parents=True, exist_ok=True)
AB = os.environ.get("BROWSER_AUTOMATION_CLI", "browser-automation-cli")  # browser-automation CLI driving the WONDER web form (see README)
CFG = (ROOT / "src/browser-configure.js").read_text()
SLEEP_Q = 130  # interval between queries (official >=2 min throttling)

WINDOWS = [(f"{y}", f"{y + 1}") for y in range(2003, 2017, 2)]  # 2003-04 ... 2015-16


def ab_eval(js: str) -> str:
    r = subprocess.run([AB, "eval", js], capture_output=True, text=True, timeout=300)
    return (r.stdout or "").strip().strip('"')


def ab_wait() -> None:
    subprocess.run([AB, "wait", "--load", "networkidle"], capture_output=True, text=True, timeout=300)


def run_query(y1: str, y2: str) -> None:
    tag = f"{y1}_{y2}"
    tsv = RAWW / f"D140_{tag}.tsv"
    if tsv.exists() and tsv.stat().st_size > 100000:
        print(f"  [{tag}] cache hit", flush=True)
        return
    # 1. Open the form page (a new session first shows an I Agree step)
    subprocess.run([AB, "open", "https://wonder.cdc.gov/cmf-icd10.html"], capture_output=True, timeout=300)
    ab_eval("(()=>{const b=[...document.querySelectorAll('input[type=submit]')].find(x=>x.name==='action-I Agree'); b&&b.click(); return 'ok'})()")
    ab_wait()
    # 2. Configure
    cfg = CFG.replace("window.PAPER18_YEARS || ['2003', '2004']", f"['{y1}', '{y2}']")
    out = ab_eval(cfg)
    # the CLI output escapes quotes (\"years\":2), so match loosely
    if "MISS" in out or "years" not in out:
        raise RuntimeError(f"[{tag}] configure failed: {out[:200]}")
    # 3. Submit
    ab_eval("(()=>{const b=[...document.querySelectorAll('#wonderform input[type=submit]')].find(x=>x.name==='action-Send'); b.click(); return 'go'})()")
    for _ in range(40):  # wait up to ~200s
        time.sleep(5)
        title = ab_eval("document.title")
        if "Results" in title:
            break
        if "Message" in title:  # a WONDER Message page = error page
            body = ab_eval("document.body.innerText.slice(0,400)")
            raise RuntimeError(f"[{tag}] query error: {body[:200]}")
    else:
        raise RuntimeError(f"[{tag}] timeout waiting for results")
    time.sleep(3)
    # 4. Serialize the export form + session URL
    action = ab_eval("document.querySelectorAll('form')[1].action")
    payload = ab_eval("(()=>{const f=document.querySelectorAll('form')[1]; const fd=new FormData(f); fd.set('O_export-format','tsv'); fd.append('action-Export','Download'); return new URLSearchParams(fd).toString()})()")
    cookies = ab_eval("document.cookie")
    # 5. Download the TSV with curl
    open("/tmp/d140_export_post.txt", "w").write(payload)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "--max-time", "900",
         "--data", "@/tmp/d140_export_post.txt",
         "-H", "Content-Type: application/x-www-form-urlencoded",
         "-H", f"Cookie: {cookies}",
         "-H", "Referer: https://wonder.cdc.gov/controller/datarequest/D140",
         action.replace("&amp;", "&"), "-o", str(tsv), "-w", "%{http_code} %{size_download}"],
        capture_output=True, text=True, timeout=900,
    )
    print(f"  [{tag}] export {r.stdout.strip()} -> {tsv.name}", flush=True)
    if r.stdout.strip().split()[1] == "0" or not tsv.exists():
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
    # ---- Parse all TSVs
    frames = []
    for tsv in sorted(RAWW.glob("D140_*.tsv")):
        df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
        df = df[df["State"].ne("")]
        frames.append(df)
        print(f"  parse {tsv.name}: {len(df)} rows", flush=True)
    if not frames:
        raise SystemExit("no TSV parsed")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"State Code": "state_fips", "ICD 113 Groups Code": "cause_group"})
    out["year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["suppressed"] = out["Deaths"].str.contains("Suppressed", case=False, na=False)
    for c in ("Deaths", "Population"):
        out[c] = pd.to_numeric(out[c].str.replace(",", "", regex=False).str.extract(r"([\d.]+)")[0], errors="coerce")
    out["crude_rate"] = pd.to_numeric(out["Crude Rate"].str.extract(r"([\d.]+)")[0], errors="coerce")
    out["cause_group_label"] = out["ICD 113 Groups"]
    out = out[out["state_fips"].ne("")] if "state_fips" in out else out
    keep = out[["state_fips", "State", "year", "cause_group", "cause_group_label",
                "Deaths", "Population", "crude_rate", "suppressed"]].copy()
    keep.columns = ["state_fips", "state_name", "year", "cause_group", "cause_group_label",
                    "deaths", "population", "crude_rate", "suppressed"]
    keep = keep.dropna(subset=["state_fips", "year"]).reset_index(drop=True)
    keep.to_parquet(ROOT / "data/state_mortality.parquet", index=False)
    print(f"wrote state_mortality.parquet rows={len(keep)} states={keep.state_fips.nunique()} "
          f"years={keep.year.min()}-{keep.year.max()} causes={keep.cause_group.nunique()} "
          f"suppressed={int(keep['suppressed'].sum())}", flush=True)


if __name__ == "__main__":
    main()
