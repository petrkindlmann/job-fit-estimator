"""Fetch MPSV ISPV open data and cache as parquet.

Source: https://data.mpsv.cz/od/soubory/ispv-zamestnani/ispv-zamestnani.json
Schema: CZ-ISCO × wage/pay sphere distributions (D1, Q1, median, Q3, D9, mean, count, period).
"""
from pathlib import Path
import json
import httpx
import pandas as pd

URL = "https://data.mpsv.cz/od/soubory/ispv-zamestnani/ispv-zamestnani.json"
OUT_RAW = Path("data/ispv/raw/ispv-zamestnani.json")
OUT_PARQUET = Path("data/ispv/ispv.parquet")


def fetch():
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.get(URL)
        r.raise_for_status()
        OUT_RAW.write_bytes(r.content)
    payload = json.loads(OUT_RAW.read_text())
    # MPSV uses "polozky" (items) as the array key.
    rows = payload.get("polozky") or payload.get("items") or payload
    if not isinstance(rows, list):
        raise SystemExit(f"Unexpected ISPV schema: {type(rows)}")
    df = pd.json_normalize(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(df)} rows → {OUT_PARQUET}")


if __name__ == "__main__":
    fetch()
