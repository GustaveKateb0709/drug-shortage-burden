"""
paper18 wonder_extract_aar.py -- Phase 2.5 revision: WONDER state-level extraction with age-adjusted rates (AAR)
Run:  python3 src/wonder_extract_aar.py

Same specification as state_mortality.parquet (51 states; D140 113 groups 2003-2016 in two-window
shards + D76 chapters 2017-2020); the only difference: the request enables O_aar_enable + M_9
(2000 US standard population), and the exported TSV gains an Age-Adjusted Rate column.
Outputs: data/raw/wonder/D140aar_*.tsv, D76aar_*.tsv + data/state_mortality_aar.parquet
      (schema identical to state_mortality.parquet plus the aar / aar_suppressed columns)
"""
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAWW = ROOT / "data/raw/wonder"
AB = os.environ.get("BROWSER_AUTOMATION_CLI", "browser-automation-cli")  # browser-automation CLI driving the WONDER web form (see README)
SLEEP_Q = 130
CFG_D140 = (ROOT / "src/browser-configure.js").read_text()       # already contains the AAR switch
CFG_D76 = (ROOT / "src/browser-configure-d76.js").read_text()    # AAR controls to be probed at runtime

WINDOWS_D140 = [(f"{y}", f"{y + 1}") for y in range(2003, 2017, 2)]
WINDOWS_D76 = [("2017", "2018"), ("2019", "2020")]


def ab_eval(js: str) -> str:
    r = subprocess.run([AB, "eval", js], capture_output=True, text=True, timeout=300)
    return (r.stdout or "").strip().strip('"')


def ab_wait() -> None:
    subprocess.run([AB, "wait", "--load", "networkidle"], capture_output=True, text=True, timeout=300)


def open_form(url: str) -> None:
    subprocess.run([AB, "open", url], capture_output=True, timeout=300)
    ab_eval("(()=>{const b=[...document.querySelectorAll('input[type=submit]')].find(x=>x.name==='action-I Agree'); b&&b.click(); return 'ok'})()")
    ab_wait()


def submit_and_export(tag: str, tsv: Path, cfg: str, years: list[str]) -> None:
    out = ab_eval(cfg.replace("window.PAPER18_YEARS || ['2003', '2004']", str(years))
                     .replace("window.PAPER18_YEARS || ['2017', '2018']", str(years)))
    if "MISS" in out or "years" not in out:
        raise RuntimeError(f"[{tag}] configure failed: {out[:200]}")
    ab_eval("(()=>{const b=[...document.querySelectorAll('#wonderform input[type=submit]')].find(x=>x.name==='action-Send'); b.click(); return 'go'})()")
    for _ in range(40):
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
    open("/tmp/aar_export_post.txt", "w").write(payload)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "--max-time", "900",
         "--data", "@/tmp/aar_export_post.txt",
         "-H", "Content-Type: application/x-www-form-urlencoded",
         "-H", f"Cookie: {cookies}",
         "-H", "Referer: https://wonder.cdc.gov/controller/datarequest",
         action.replace("&amp;", "&"), "-o", str(tsv), "-w", "%{http_code} %{size_download}"],
        capture_output=True, text=True, timeout=900,
    )
    print(f"  [{tag}] export {r.stdout.strip()} -> {tsv.name}", flush=True)
    if r.stdout.strip().split()[1] == "0":
        raise RuntimeError(f"[{tag}] export failed")


def run_db(prefix: str, form_url: str, windows: list[tuple[str, str]], cfg: str) -> None:
    for i, (y1, y2) in enumerate(windows):
        tag = f"{prefix}_{y1}_{y2}"
        tsv = RAWW / f"{tag}.tsv"
        if tsv.exists() and tsv.stat().st_size > 100000:
            print(f"  [{tag}] cache hit", flush=True)
            continue
        if i:
            print(f"  throttle {SLEEP_Q}s ...", flush=True)
            time.sleep(SLEEP_Q)
        try:
            open_form(form_url)
            submit_and_export(tag, tsv, cfg, [y1, y2])
        except Exception as e:  # noqa: BLE001
            print(f"  [{tag}] FAILED: {e}", flush=True)


def parse_and_merge() -> None:
    frames = []
    for prefix, source, gran in [("D140aar", "CMF-ICD10-2016(D140)", "113groups"),
                                 ("D76aar", "UCD-ICD10-2020(D76)", "chapter")]:
        for tsv in sorted(RAWW.glob(f"{prefix}_2*.tsv")):
            df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("")
            df = df[df["State"].ne("")]
            code_col = [c for c in df.columns if ("113" in c or "Chapter" in c) and c.endswith("Code")][0]
            lab_col = [c for c in df.columns if ("113" in c or "Chapter" in c) and not c.endswith("Code")][0]
            aar_col = [c for c in df.columns if "adjust" in c.lower() and not any(k in c.lower() for k in ("standard error", "confidence"))]
            if not aar_col:
                raise RuntimeError(f"{tsv.name}: no AAR column; columns={list(df.columns)}")
            df = df.rename(columns={"State Code": "state_fips", code_col: "cause_code", lab_col: "cause_label",
                                    aar_col[0]: "aar_raw", "Crude Rate": "crude_rate_raw"})
            df["cause_label"] = df["cause_label"].str.lstrip("#")
            df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
            df["suppressed"] = df["Deaths"].str.contains("Suppressed", case=False, na=False)
            df["aar_suppressed"] = df["aar_raw"].str.contains("Suppressed", case=False, na=False)
            for c in ("Deaths", "Population"):
                df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False).str.extract(r"([\d.]+)")[0], errors="coerce")
            df["crude_rate"] = pd.to_numeric(df["crude_rate_raw"].str.extract(r"([\d.]+)")[0], errors="coerce")
            df["aar"] = pd.to_numeric(df["aar_raw"].str.extract(r"([\d.]+)")[0], errors="coerce")
            out = df[["state_fips", "State", "year", "cause_code", "cause_label", "Deaths", "Population",
                      "crude_rate", "aar", "suppressed", "aar_suppressed"]].copy()
            out.columns = ["state_fips", "state_name", "year", "cause_code", "cause_label", "deaths", "population",
                           "crude_rate", "aar", "suppressed", "aar_suppressed"]
            out["granularity"] = gran
            out["source"] = source
            frames.append(out)
            print(f"  parse {tsv.name}: {len(out)} rows, aar non-suppressed={int((~out.aar_suppressed).sum())}", flush=True)
    if not frames:
        raise SystemExit("no TSV parsed")
    res = pd.concat(frames, ignore_index=True).dropna(subset=["state_fips", "year"])
    res.to_parquet(ROOT / "data/state_mortality_aar.parquet", index=False)
    n = len(res)
    cov = float((~res["aar_suppressed"]).mean())
    print(f"wrote state_mortality_aar.parquet rows={n} states={res.state_fips.nunique()} "
          f"years={res.year.min()}-{res.year.max()} aar_coverage={cov:.1%}", flush=True)
    # Magnitude sanity: an all-cause (death-weighted across states) AAR is not meaningful -- inspect the AAR distribution of chapter rows directly
    a = res.loc[~res.aar_suppressed, "aar"]
    print(f"aar range [{a.min():.1f}, {a.max():.1f}] per 100k, median={a.median():.1f}", flush=True)


def main() -> None:
    run_db("D140aar", "https://wonder.cdc.gov/cmf-icd10.html", WINDOWS_D140, CFG_D140)
    run_db("D76aar", "https://wonder.cdc.gov/ucd-icd10.html", WINDOWS_D76, CFG_D76)
    parse_and_merge()


if __name__ == "__main__":
    main()
