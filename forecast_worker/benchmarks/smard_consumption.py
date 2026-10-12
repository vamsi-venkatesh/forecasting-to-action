"""A maintained, recent public series: total German electricity consumption per day (grid load), from SMARD, the
Bundesnetzagentur's electricity market data platform.

Source: https://www.smard.de (chart data, filter 410 "Stromverbrauch: Gesamt (Netzlast)", region DE, daily).
Licence: CC BY 4.0. Attribution: "Bundesnetzagentur | SMARD.de" (verified on smard.de/home/datennutzung, 2026-09-28).

What the data is, and what is assumed (all recorded in the manifest, nothing filled in):
- One value per calendar day in Germany (Europe/Berlin), in MWh. A SMARD daily point is stamped at local midnight;
  it is mapped back to that local date. Days with a clock change have 23 or 25 hours and are kept as published.
- Days not yet published are empty in the source and are left out; the series ends at the latest published day.
- Recent values may still be revised by the source; the retrieval time and the checksum of what was read are kept,
  so a later retrieval that differs is a new version, never a silent change.
- The source gives no publication time per day. ASSUMPTION: a day's value is known at 12:00 UTC the next day. This is
  stated wherever the data is used; the forecast is only scored on what was known by then.
Nothing is shifted: the dates are the real dates, and the forecast is issued as of the retrieval time.

    python -m forecast_worker.benchmarks.smard_consumption OUT_DIR [--since 2019-01-01] [--regions DE]
    (--regions 50Hertz,Amprion,TenneT,TransnetBW: the four German control areas, one series each)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://www.smard.de/app/chart_data/410"
BERLIN = ZoneInfo("Europe/Berlin")
ATTRIBUTION = "Bundesnetzagentur | SMARD.de"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "VVDex-benchmark/1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _series(region: str, since: date, fetch, raw) -> dict[date, float]:
    index_raw = fetch(f"{BASE}/{region}/index_day.json")
    raw.update(index_raw)
    days: dict[date, float] = {}
    for t in json.loads(index_raw)["timestamps"]:
        if datetime.fromtimestamp(t / 1000, timezone.utc).date() < since - timedelta(days=400):
            continue                                   # each file is one year of days; older years are not read
        body = fetch(f"{BASE}/{region}/410_{region}_day_{t}.json")
        raw.update(body)
        for ts, v in json.loads(body)["series"]:
            if v is None:
                continue                               # not published yet: left out, never filled
            local = datetime.fromtimestamp(ts / 1000, timezone.utc).astimezone(BERLIN).date()
            if local >= since:
                days[local] = float(v)
    return days


def build(out: Path, since: date = date(2019, 1, 1), fetch=_get, regions: tuple[str, ...] = ("DE",)) -> dict:
    retrieved = datetime.now(timezone.utc).replace(microsecond=0)
    raw = hashlib.sha256()
    per = {r: _series(r, since, fetch, raw) for r in regions}
    if not all(per.values()):
        raise SystemExit("SMARD returned no published days for " + ", ".join(r for r, d in per.items() if not d))
    days = {d for v in per.values() for d in v}
    first, last = min(days), max(days)
    missing = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)
               if any(first + timedelta(days=i) not in v for v in per.values())]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sales_daily.csv").open("w", newline="") as fh:        # the pipeline's standard daily file
        w = csv.writer(fh)
        w.writerow(["series_id", "period_start", "available_at", "consumption_mwh", "unit"])
        for r in regions:
            for d in sorted(per[r]):
                w.writerow([f"{r}|electricity consumption", d.isoformat(), f"{(d + timedelta(days=1)).isoformat()}T12:00:00+00:00",
                            f"{per[r][d]:.2f}", "MWh"])
    manifest = {
        "benchmark": "smard-de-consumption-daily-1", "source": f"https://www.smard.de (chart data 410, {', '.join(regions)}, day)",
        "regions": list(regions),
        "licence": "CC BY 4.0", "attribution": ATTRIBUTION, "retrieved_at": retrieved.isoformat(),
        "raw_sha256": raw.hexdigest(), "target": "consumption_mwh: total electricity consumption (grid load) per day, MWh",
        "frequency": "daily", "calendar": "Europe/Berlin calendar days; clock-change days have 23 or 25 hours, kept as published",
        "coverage": {"first_day": first.isoformat(), "last_day": last.isoformat(), "days": len(days), "missing_days": missing},
        "assumptions": {"available_at": "a day's value is known at 12:00 UTC the next day (the source gives no publication time)",
                        "revisions": "recent days may be revised by the source; each retrieval is a version (raw_sha256)"},
        "lag_at_retrieval_days": (retrieved.astimezone(BERLIN).date() - last).days,
        "not_used": "no weather, prices or holidays: only the series' own history",
        "files": {"sales_daily.csv": hashlib.sha256((out / "sales_daily.csv").read_bytes()).hexdigest()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--regions", default="DE")
    a = ap.parse_args()
    m = build(Path(a.out), date.fromisoformat(a.since), regions=tuple(a.regions.split(",")))
    print(json.dumps({k: m[k] for k in ("retrieved_at", "coverage", "lag_at_retrieval_days", "raw_sha256")})[:600])

