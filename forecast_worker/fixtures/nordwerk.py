"""NordWerk Demo: a seeded, clearly FICTIONAL operational fixture (brief §5, "synthetic operational fixture").

It exercises workflow behaviour — ingestion, profiling, ambiguity questions, planning arithmetic — and is
NOT evidence of real-world forecast quality. Deliberate defects, each listed in the manifest:
  late-starting histories, zero-sales weeks, missing weeks, stockout weeks (sales censored by zero stock),
  returns, a mid-history schema change (qty -> quantity, a second file), duplicated context messages,
  unconfirmed and late receipts.

    python -m forecast_worker.fixtures.nordwerk OUT_DIR [--seed 20260928]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PRODUCTS = [("P-100", "Hydraulic filter"), ("P-200", "Seal kit"), ("P-300", "Pump housing"), ("P-400", "Valve block")]
REGIONS = ["DE-N", "DE-S", "AT"]
WEEKS = 156
START = date(2023, 10, 2)          # a Monday; weeks are ISO weeks starting Monday, Europe/Berlin calendar
FIXTURE_VERSION = "nordwerk-demo-1"


def _week(i: int) -> date:
    return START + timedelta(weeks=i)


def build(out: Path, seed: int = 20260928) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    series = [(p, r) for p, _ in PRODUCTS for r in REGIONS]
    defects = {"late_start": {}, "zero_weeks": {}, "missing_weeks": {}, "stockout_weeks": {}}
    rows_a, rows_b, inventory, receipts = [], [], [], []
    for k, (pid, region) in enumerate(series):
        sid = f"{pid}|{region}"
        level = rng.uniform(20, 160)
        trend = rng.uniform(-0.05, 0.12)
        amp = rng.uniform(0.05, 0.35) * level
        start = 0 if k % 5 else int(rng.integers(40, 90))          # every 5th series starts late
        defects["late_start"][sid] = start
        missing = set(rng.choice(range(start + 5, WEEKS), size=int(rng.integers(0, 4)), replace=False).tolist())
        stockouts = set(rng.choice(range(start + 10, WEEKS), size=int(rng.integers(0, 5)), replace=False).tolist())
        defects["missing_weeks"][sid] = sorted(missing)
        defects["stockout_weeks"][sid] = sorted(stockouts)
        zeros = []
        for w in range(start, WEEKS):
            season = amp * np.sin(2 * np.pi * (w % 52) / 52)
            mean = max(0.0, level + trend * w + season)
            qty = int(rng.poisson(mean))
            if rng.random() < 0.03:
                qty = 0
            if w in stockouts:
                qty = int(qty * rng.uniform(0.0, 0.3))            # censored by zero usable stock
            if qty == 0:
                zeros.append(w)
            returns = int(rng.binomial(qty, 0.02)) if qty else 0
            if w in missing:
                continue                                           # no row at all: missing, not zero
            wk = _week(w)
            event_time = datetime(wk.year, wk.month, wk.day, tzinfo=timezone.utc)
            available_at = event_time + timedelta(days=7 + int(rng.integers(0, 3)))
            rec = {"source_record_id": f"S-{sid}-{w:03d}", "series_id": sid, "product_id": pid, "region_id": region,
                   "week_start": wk.isoformat(), "available_at": available_at.isoformat(),
                   "unit": "pcs", "returned": returns, "source_revision": 1}
            if w < 104:
                rows_a.append({**rec, "qty": qty})                 # old export
            else:
                rows_b.append({**rec, "quantity": qty})            # new export: renamed column
            stock = 0 if w in stockouts else int(max(0, rng.normal(mean * 2.2, mean * 0.5)))
            inventory.append({"series_id": sid, "snapshot_time": (event_time + timedelta(days=6, hours=23)).isoformat(),
                              "available_at": (event_time + timedelta(days=7, hours=6)).isoformat(),
                              "usable_on_hand": stock, "committed": int(rng.integers(0, max(1, int(mean * 0.2)))),
                              "unit": "pcs", "source_version": 1})
        defects["zero_weeks"][sid] = zeros
        for j in range(3):                                         # future inbound receipts
            eta = _week(WEEKS + int(rng.integers(0, 5)))
            receipts.append({"receipt_id": f"R-{sid}-{j}", "series_id": sid, "quantity": int(rng.integers(20, 200)),
                             "expected_date": eta.isoformat(), "unit": "pcs",
                             "status": ["confirmed", "confirmed", "unconfirmed"][j],
                             "announced_at": _week(WEEKS - 1).isoformat()})
    rules = [{"rule_version": 1, "effective_from": _week(0).isoformat(), "series_id": sid, "horizon_weeks": 4,
              "protection_weeks": 4, "buffer_units": 20, "pack_size": 10, "min_order_units": 20,
              "unit_cost_eur": f"{rng.uniform(2, 40):.2f}", "currency": "EUR",
              "destination": "demo-planning-ledger", "approval": "operations lead"} for sid in (f"{p}|{r}" for p, r in series)]
    messages = []
    kinds = [("promotion", "A regional trade fair promotion for {p} runs in weeks {a}-{b}. Confirmed by marketing."),
             ("supply_delay", "Supplier notice: the inbound batch for {p} is delayed by about two weeks."),
             ("cancellation", "The planned {p} campaign in {r} is cancelled."),
             ("postponement", "The {p} customer rollout in {r} is postponed to next quarter."),
             ("rumour", "A distributor mentioned a possible large order for {p}, nothing confirmed.")]
    for m in range(40):
        kind, template = kinds[int(rng.integers(0, len(kinds)))]
        pid, pname = PRODUCTS[int(rng.integers(0, len(PRODUCTS)))]
        region = REGIONS[int(rng.integers(0, len(REGIONS)))]
        w = int(rng.integers(20, WEEKS))
        text = template.format(p=pname, r=region, a=w + 2, b=w + 4)
        pub = datetime.combine(_week(w), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=int(rng.integers(0, 6)))
        msg = {"source_event_id": f"M-{m:03d}", "source": "demo-inbox", "published_at": pub.isoformat(),
               "received_at": (pub + timedelta(hours=int(rng.integers(1, 30)))).isoformat(),
               "product_id": pid, "region_id": region, "text": text,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "fixture_label": kind}
        messages.append(msg)
        if m % 9 == 0:                                              # duplicated source message
            messages.append({**msg, "source_event_id": f"M-{m:03d}-dup",
                             "received_at": (pub + timedelta(hours=40)).isoformat()})
    files = {"sales_2023_2025.csv": rows_a, "sales_2025_2026.csv": rows_b, "inventory.csv": inventory,
             "receipts.csv": receipts, "rules.csv": rules, "context_messages.csv": messages}
    manifest = {"fixture": FIXTURE_VERSION, "fictional": True, "seed": seed, "weeks": WEEKS,
                "first_week": START.isoformat(), "series": len(series), "calendar": "ISO weeks, Monday start, Europe/Berlin",
                "note": "Synthetic. Tests workflow behaviour; says nothing about real forecast quality.",
                "defects": {"late_start": {k: v for k, v in defects["late_start"].items() if v},
                            "missing_weeks": {k: v for k, v in defects["missing_weeks"].items() if v},
                            "stockout_weeks": {k: v for k, v in defects["stockout_weeks"].items() if v},
                            "zero_week_count": {k: len(v) for k, v in defects["zero_weeks"].items()},
                            "schema_change": "sales_2025_2026.csv renames qty to quantity from week 104",
                            "duplicated_messages": [m["source_event_id"] for m in messages if m["source_event_id"].endswith("-dup")],
                            "unconfirmed_receipts": [r["receipt_id"] for r in receipts if r["status"] == "unconfirmed"]},
                "files": {}}
    for name, rows in files.items():
        path = out / name
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        manifest["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20260928)
    a = ap.parse_args()
    m = build(Path(a.out), a.seed)
    print(json.dumps({k: m[k] for k in ("fixture", "series", "weeks")} | {"files": {k: v["rows"] for k, v in m["files"].items()}}))

