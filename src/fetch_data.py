"""
paper18 fetch_data.py -- Step 0/1: persist all raw data snapshots
Run:  python3 src/fetch_data.py

All requests go through curl --noproxy '*' (subprocess calls), writing into data/raw/.
After each download, raw/manifest.json records the URL, download date (UTC), byte size, and MD5.
This script is idempotent: existing files whose MD5 matches the manifest are skipped (verification against the remote is best-effort).
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

OPENFDA_FILES = {
    # Full zipped JSON (the real links follow the download.open.fda.gov standard pattern; the download pages are JS SPAs without direct links)
    "drug-shortages-0001-of-0001.json.zip": "https://download.open.fda.gov/drug/shortages/drug-shortages-0001-of-0001.json.zip",
    "drug-orangebook-0001-of-0001.json.zip": "https://download.open.fda.gov/drug/orangebook/drug-orangebook-0001-of-0001.json.zip",
    "drug-drugsfda-0001-of-0001.json.zip": "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip",
    "drug-ndc-0001-of-0001.json.zip": "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip",
}
SVI_FILES = {
    # After the SVI site redesign, downloads are served by the webapi endpoint used by the JS app at https://svi2.cdc.gov/data-downloads
    "SVI_2022_US_county.csv": "https://svi2.cdc.gov/webapi/Documents/download?year=2022&type=csv&category=states_counties&name=SVI_2022_US_COUNTY",
    "SVI_2022_US_tracts.csv": "https://svi2.cdc.gov/webapi/Documents/download?year=2022&type=csv&category=states&name=SVI_2022_US",
}
# openFDA keyless rate limit is 240 req/min -- serial downloading is already far below the limit, no extra throttling needed


def curl(url: str, out: Path) -> bool:
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-sL", "--max-time", "900", "-o", str(out), url],
        capture_output=True,
    )
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for fname, url in {**OPENFDA_FILES, **SVI_FILES}.items():
        out = RAW / fname
        prev = manifest.get(fname, {})
        if prev.get("md5") and out.exists() and md5(out) == prev["md5"]:
            print(f"[skip] {fname}")
            continue
        ok = curl(url, out)
        if not ok:
            print(f"[FAIL] {fname} {url}")
            continue
        manifest[fname] = {
            "url": url,
            "download_date_utc": now,
            "bytes": out.stat().st_size,
            "md5": md5(out),
        }
        print(f"[ok]   {fname} {manifest[fname]['bytes']} bytes")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest -> {manifest_path}")
    # API check (validation only, not used in the build; the total shortage-record count should be on the order of 1,607)
    r = subprocess.run(
        ["curl", "--noproxy", "*", "-sL", "--max-time", "60",
         "https://api.fda.gov/drug/shortages.json?limit=1"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        try:
            meta = json.loads(r.stdout)["meta"]["results"]
            print(f"API check: shortages total = {meta.get('total')}")
        except Exception as e:  # noqa: BLE001
            print(f"API check failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
