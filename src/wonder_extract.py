"""
paper18 wonder_extract.py -- Phase 2.5 task 1: CDC WONDER state x year x cause-group mortality extraction (v2)
Run:  python3 src/wonder_extract.py

Data source: CDC WONDER Compressed Mortality 1999-2016 (database code D140, XML API)
Key facts (all verified empirically, not guessed):
  - The D76 (Detailed Mortality 1999-2020) API is national level only (explicit in WONDER's own error messages) -> state-level data must come from D140
  - D140 and D76 use different parameter names: the cause-of-death single-select is O_icd (D76 calls it O_ucd)! Years are passed via the V_D140.V1 multi-select (CMF has no F_D140.V1)
  - Parameter values were captured from the real query-form DOM (serialized FormData from the rendered page; see README)
  - Posting the UI urlencoded payload directly returns the About/400 page; the XML API requires the request_xml form parameter
  - Network quirk: requests with a browser UA are blocked by Akamai fingerprinting (403); the default curl UA passes
  - Official throttling: automated queries at most once every 2 minutes (this script sleeps 130s)
Outputs:
  data/raw/wonder/D140_<span>.xml  raw response snapshots
  data/state_mortality.parquet     state_fips, state_name, year, cause_group, deaths, population, crude_rate, suppressed
"""
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAWW = ROOT / "data/raw/wonder"
RAWW.mkdir(parents=True, exist_ok=True)
SLEEP_S = 130

STATE_FIPS = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05", "California": "06",
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "District of Columbia": "11",
    "Florida": "12", "Georgia": "13", "Hawaii": "15", "Idaho": "16", "Illinois": "17",
    "Indiana": "18", "Iowa": "19", "Kansas": "20", "Kentucky": "21", "Louisiana": "22",
    "Maine": "23", "Maryland": "24", "Massachusetts": "25", "Michigan": "26",
    "Minnesota": "27", "Mississippi": "28", "Missouri": "29", "Montana": "30",
    "Nebraska": "31", "Nevada": "32", "New Hampshire": "33", "New Jersey": "34",
    "New Mexico": "35", "New York": "36", "North Carolina": "37", "North Dakota": "38",
    "Ohio": "39", "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42",
    "Rhode Island": "44", "South Carolina": "45", "South Dakota": "46", "Tennessee": "47",
    "Texas": "48", "Utah": "49", "Vermont": "50", "Virginia": "51", "Washington": "53",
    "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56",
}


def build_request(years: list[int]) -> str:
    P: list[tuple[str, list[str]]] = []

    def p(name: str, *values: str) -> None:
        P.append((name, list(values)))

    # -- Field-for-field identical to the UI-serialized payload (O_icd not O_ucd; years via the V_D140.V1 multi-select) --
    p("saved_id", "")
    p("dataset_code", "D140")
    p("dataset_label", "Compressed Mortality, 1999-2016")
    p("O_county", "D140.V100")
    p("dataset_vintage_latest", "CMF-ICD10")
    p("stage", "request")
    p("O_javascript", "on")
    p("M_1", "D140.M1")
    p("M_2", "D140.M2")
    p("M_3", "D140.M3")
    p("O_aar", "aar_none")
    p("B_1", "D140.V9-level1")       # State
    p("B_2", "D140.V1")              # Year
    p("B_3", "D140.V4-level1")       # ICD-10 113 Cause List
    p("B_4", "*None*")
    p("B_5", "*None*")
    p("O_title", "paper18 state mortality 113causes")
    p("O_oc-sect1-request", "close")
    p("O_rate_per", "100000")
    p("O_aar_pop", "0000")
    p("VM_D140.M6_D140.V1", "*All*")
    p("VM_D140.M6_D140.V7", "*All*")
    p("VM_D140.M6_D140.V17", "*All*")
    p("VM_D140.M6_D140.V8", "*All*")
    p("VM_D140.M6_D140.V10", "")
    p("O_location", "D140.V9")
    p("finder-stage-D140.V9", "codeset")
    p("O_V9_fmode", "freg")
    p("V_D140.V9", "")
    p("F_D140.V9", "*All*")
    p("I_D140.V9", "*All* (The United States)\n")
    p("finder-stage-D140.V10", "codeset")
    p("O_V10_fmode", "freg")
    p("V_D140.V10", "")
    p("F_D140.V10", "*All*")
    p("I_D140.V10", "*All* (The United States)\n")
    p("finder-stage-D140.V18", "codeset")
    p("O_V18_fmode", "freg")
    p("V_D140.V18", "")
    p("F_D140.V18", "*All*")
    p("I_D140.V18", "*All* (The United States)\n")
    p("O_cbsa", "D140.V20")
    p("V_D140.V20", "*All*")
    p("V_D140.V21", "*All*")
    p("O_urban", "D140.V22")
    p("V_D140.V22", "*All*")
    p("V_D140.V19", "*All*")
    p("V_D140.V23", "*All*")
    p("V_D140.V11", "*All*")
    p("O_age", "D140.V5")
    p("V_D140.V5", "*All*")
    p("V_D140.V6", "00")
    p("V_D140.V1", *[str(y) for y in years])
    p("V_D140.V7", "*All*")
    p("V_D140.V8", "*All*")
    p("V_D140.V17", "*All*")
    p("O_icd", "D140.V4")            # <- key CMF difference: the O_icd single-select switches to ICD-10 113 Groups
    p("finder-stage-D140.V2", "codeset")
    p("O_V2_fmode", "freg")
    p("V_D140.V2", "")
    p("F_D140.V2", "*All*")
    p("I_D140.V2", "*All* (All Causes of Death)\n")
    p("finder-stage-D140.V4", "codeset")
    p("O_V4_fmode", "freg")
    p("V_D140.V4", "")
    p("F_D140.V4", "*All*")
    p("I_D140.V4", "*All* (All Causes of Death)\n")
    p("V_D140.V12", "*All*")
    p("V_D140.V13", "*All*")
    p("O_export-format", "xls")
    p("O_show_zeros", "true")
    p("O_show_suppressed", "true")
    p("O_precision", "1")
    p("O_timeout", "600")
    p("action-Send", "Send")
    p("accept_datause_restrictions", "true")

    xml = '<?xml version="1.0" encoding="utf-8"?>\n<request-parameters>\n'
    for name, vals in P:
        xml += "\t<parameter>\n\t\t<name>%s</name>\n" % name
        for v in vals:
            xml += "\t\t<value>%s</value>\n" % v
        xml += "\t</parameter>\n"
    return xml + "</request-parameters>"


def post_query(xml: str, tag: str) -> str:
    req_file = RAWW / f"req_{tag}.xml"
    resp_file = RAWW / f"D140_{tag}.xml"
    req_file.write_text(xml)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-s", "--max-time", "900",
         "--data-urlencode", f"request_xml@{req_file}",
         "--data", "accept_datause_restrictions=true",
         "https://wonder.cdc.gov/controller/datarequest/D140",
         "-o", str(resp_file), "-w", "%{http_code}"],
        capture_output=True, text=True,
    )
    code = r.stdout.strip()
    body = resp_file.read_text(errors="replace")
    print(f"  [{tag}] HTTP {code}, {len(body)} bytes -> {resp_file.name}", flush=True)
    return body


def parse_response(body: str) -> pd.DataFrame | None:
    root = ET.fromstring(body)
    msgs = [m.text for m in root.iter("message")]
    if root.find(".//data-table") is None:
        print("  no data-table; messages:", msgs[:5], flush=True)
        return None
    rows = []
    for r in root.iter("r"):
        cells = [c.text if c.text is not None else "" for c in r.findall("c")]
        attrs = [c.attrib for c in r.findall("c")]
        if any(a.get("c") in ("1", "2") for a in attrs):
            continue
        if len(cells) >= 6:
            rows.append(cells[:6])
    return pd.DataFrame(rows, columns=["state_name", "year", "cause_group", "deaths", "population", "crude_rate"])


def main() -> None:
    year_splits = [list(range(2003, 2010)), list(range(2010, 2017))]
    frames = []
    for i, years in enumerate(year_splits):
        tag = f"{years[0]}_{years[-1]}"
        cache = RAWW / f"D140_{tag}.xml"
        if cache.exists() and cache.stat().st_size > 50000 and "<data-table" in cache.read_text(errors="replace")[:200000]:
            body = cache.read_text(errors="replace")
            print(f"  [{tag}] cache hit", flush=True)
        else:
            body = post_query(build_request(years), tag)
            if i < len(year_splits) - 1:
                print(f"  throttle {SLEEP_S}s ...", flush=True)
                time.sleep(SLEEP_S)
        df = parse_response(body)
        if df is not None and len(df):
            frames.append(df)
            print(f"  [{tag}] parsed rows: {len(df)}", flush=True)
    if not frames:
        print("NO DATA parsed -- raw responses kept for manual inspection", flush=True)
        sys.exit(1)
    out = pd.concat(frames, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for c in ("deaths", "population"):
        out[c] = pd.to_numeric(out[c].str.replace(",", "", regex=False), errors="coerce")
    out["crude_rate"] = pd.to_numeric(out["crude_rate"].str.replace(",", "", regex=False), errors="coerce")
    out["suppressed"] = out["deaths"].isna()
    out["state_fips"] = out["state_name"].map(STATE_FIPS)
    out = out.dropna(subset=["state_fips", "year"])
    out = out[["state_fips", "state_name", "year", "cause_group", "deaths", "population", "crude_rate", "suppressed"]]
    out.to_parquet(ROOT / "data/state_mortality.parquet", index=False)
    print(f"wrote {ROOT/'data/state_mortality.parquet'} rows={len(out)} "
          f"states={out.state_fips.nunique()} years={out.year.min()}-{out.year.max()} "
          f"causes={out.cause_group.nunique()} suppressed={int(out['suppressed'].sum())}", flush=True)


if __name__ == "__main__":
    main()
