"""Public benchmark B: UCI Bike Sharing (daily) → one daily series of total rentals, for the SAME forecasting
pipeline as the retail data (no bike-specific application).

Source: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894
Licence: CC BY 4.0. File: bike_sharing_dataset.zip, 279,992 bytes, SHA-256
b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401 (downloaded 2026-09-28 from
https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip). The script refuses any other file.

What is used and what is not (recorded in the manifest):
- Target: `cnt`, total daily rentals (casual + registered), 2011-01-01 .. 2012-12-31, 731 days, no gaps.
- NOT used as predictors: `casual` and `registered` (their same-day sum IS the target), and the observed weather
  (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`): on the forecast day it is not known in advance. The default
  forecast uses the series' own past only. Calendar fields are derivable from the date and are not needed.
- available_at is an ASSUMPTION (the next day 06:00 UTC, i.e. a one-day data lag); the source has no such field.
- This is historical replay of 2011-2012, not a current forecast. There is no stock, no purchase and no action.

    python -m forecast_worker.benchmarks.uci_bike_sharing ZIP OUT_DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ZIP_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
BENCHMARK_VERSION = "uci-bike-sharing-daily-1"
SERIES = "bike-rentals|total"


def build(zip_path: Path, out: Path) -> dict:
    raw = zip_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ZIP_SHA256:
        raise SystemExit(f"refusing {zip_path.name}: sha256 {sha} is not the recorded UCI file")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("day.csv"), encoding="utf-8")))
    out.mkdir(parents=True, exist_ok=True)
    days, bad, sales = [], [], []
    for r in rows:
        try:
            d = date.fromisoformat(r["dteday"])
            cnt = int(r["cnt"])
            if cnt != int(r["casual"]) + int(r["registered"]):
                bad.append({"day": r["dteday"], "why": "cnt is not casual + registered"})
                continue
        except (KeyError, ValueError) as exc:
            bad.append({"row": r.get("instant"), "why": f"unreadable: {exc.__class__.__name__}"})
            continue
        avail = datetime.combine(d + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=6)
        sales.append({"series_id": SERIES, "period_start": d.isoformat(), "available_at": avail.isoformat(),
                      "rentals": cnt, "unit": "rentals"})
        days.append(d)
    expected = (max(days) - min(days)).days + 1
    path = out / "sales_daily.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sales[0]))
        w.writeheader()
        w.writerows(sales)
    manifest = {
        "benchmark": BENCHMARK_VERSION, "source": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894",
        "licence": "CC BY 4.0", "zip_sha256": sha, "file_used": "day.csv",
        "coverage": {"first_day": min(days).isoformat(), "last_day": max(days).isoformat(), "days": len(days),
                     "expected_days": expected, "gaps": expected - len(days)},
        "target": "cnt: total rentals per day (casual + registered)", "frequency": "daily",
        "not_used": {"casual, registered": "components of the same-day target",
                     "temp, atemp, hum, windspeed, weathersit": "observed on the day; not known when forecasting",
                     "season, yr, mnth, holiday, weekday, workingday": "derivable from the date; not used by the default models"},
        "excluded_rows": bad,
        "assumptions": {"available_at": "next day 06:00 UTC (one-day lag); the source has no arrival time"},
        "replay": "historical replay of 2011-2012; not a current forecast; no stock, purchasing or actions",
        "files": {"sales_daily.csv": {"rows": len(sales), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("out")
    a = ap.parse_args()
    m = build(Path(a.zip), Path(a.out))
    print(json.dumps({"coverage": m["coverage"], "excluded": len(m["excluded_rows"]), "files": m["files"]}))

