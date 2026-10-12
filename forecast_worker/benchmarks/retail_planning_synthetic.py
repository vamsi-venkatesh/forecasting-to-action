"""SYNTHETIC planning inputs for the UCI Online Retail II sample, so the public retail demo can reach a held
demo action. Everything here is invented and labelled so; only the sales history is real (UCI, CC BY 4.0).

    python -m forecast_worker.benchmarks.retail_planning_synthetic PREPARED_UCI_DIR OUT_DIR [--series 8] [--seed 20260928]

- inventory.csv: one stock snapshot per series on the replay date (Sunday 2011-12-04), sized from the series'
  own past sales (2.5-6.5 weeks of its mean), so some series fall short and some do not.
- receipts.csv: 1-2 inbound receipts per series in December 2011, some confirmed, some not.
- rules.csv: buffer, pack size, minimum order and a PURCHASE cost in GBP. The purchase cost is synthetic: the UCI
  file has retail SELLING prices, which are not procurement costs, and nothing is converted to EUR.
Replay: the forecast is issued as of 2011-12-05 (after the last complete week), a historical replay, not 2026.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

DISCLOSURE = ("Historical sales: UCI public dataset. Inventory and purchasing rules: synthetic scenario. "
              "Action destination: isolated demo ledger.")
REPLAY_ISSUED_AT = "2011-12-05T08:00:00+00:00"
SNAPSHOT = date(2011, 12, 4)


def build(prepared: Path, out: Path, series: int = 8, seed: int = 20260928) -> dict:
    rng = random.Random(seed)
    src = next(prepared.glob("sales_*.csv"))
    with src.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = sorted({r["series_id"] for r in rows})[:series]
    rows = [r for r in rows if r["series_id"] in ids]
    out.mkdir(parents=True, exist_ok=True)
    with (out / src.name).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    inv, rec, rules = [], [], []
    for sid in ids:
        hist = [max(0, int(r["quantity"])) for r in rows if r["series_id"] == sid]
        mean = sum(hist[-13:]) / max(1, len(hist[-13:]))              # the last quarter, known on the replay date
        inv.append({"series_id": sid, "snapshot_time": f"{SNAPSHOT.isoformat()}T23:00:00+00:00",
                    "available_at": f"{(SNAPSHOT + timedelta(days=1)).isoformat()}T06:00:00+00:00",
                    "usable_on_hand": int(mean * rng.uniform(2.5, 6.5)), "committed": int(mean * rng.uniform(0, 0.2)),
                    "unit": "units", "source_version": 1})
        for j in range(rng.randint(1, 2)):
            rec.append({"receipt_id": f"SYN-R-{sid}-{j}", "series_id": sid, "quantity": int(mean * rng.uniform(0.5, 1.5)),
                        "expected_date": (SNAPSHOT + timedelta(days=rng.randint(3, 24))).isoformat(), "unit": "units",
                        "status": "confirmed" if j == 0 or rng.random() < 0.5 else "unconfirmed",
                        "announced_at": SNAPSHOT.isoformat()})
        pack = rng.choice([12, 24, 48])
        rules.append({"rule_version": 1, "effective_from": "2011-01-03", "series_id": sid, "horizon_weeks": 4,
                      "protection_weeks": 4, "buffer_units": int(mean * 0.5), "pack_size": pack, "min_order_units": pack * 2,
                      "unit_cost": f"{rng.uniform(0.3, 2.5):.2f}", "currency": "GBP",
                      "destination": "isolated demo ledger", "approval": "demo approver (your own session)"})
    # SYNTHETIC supplier messages about the confirmed receipts, for the sample's one AI-read step (labelled synthetic
    # in every row and in the manifest). Built after every random draw above, so the other files are unchanged.
    conf = [r for r in rec if r["status"] == "confirmed"]
    a, b, c, d = conf[0], conf[1], conf[2], conf[3]
    late = (date.fromisoformat(a["expected_date"]) + timedelta(days=14)).isoformat()
    msgs = [
        ("SYN-M-1", "2011-12-02T09:40:00+00:00", f"Supplier update about delivery {a['receipt_id']}: the carrier missed its loading "
         f"slot, so the goods now arrive on {late} instead of {a['expected_date']}. The new date is confirmed by the forwarder."),
        ("SYN-M-2", "2011-12-03T14:05:00+00:00", f"About {b['receipt_id']}: there may be a hold-up at the port this week. We will "
         "confirm the arrival date once customs has replied."),
        ("SYN-M-3", "2011-12-03T16:20:00+00:00", f"{c['receipt_id']} left our warehouse as planned and is on track for "
         f"{c['expected_date']}."),
        ("SYN-M-4", "2011-12-04T08:15:00+00:00", f"Please note our office is closed on 26 December. Nothing changes for "
         f"{d['receipt_id']}, which arrives as agreed."),
    ]
    files_msgs = [{"message_id": m, "received_at": t, "text": x, "label": "synthetic message (not a real supplier)"} for m, t, x in msgs]
    files = {"inventory.csv": inv, "receipts.csv": rec, "rules.csv": rules, "supplier_messages.csv": files_msgs}
    for name, data in files.items():
        with (out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    manifest = {"dataset": "uci-retail-sample", "disclosure": DISCLOSURE, "series": ids, "seed": seed,
                "replay_issued_at": REPLAY_ISSUED_AT, "currency": "GBP",
                "real": {src.name: "UCI Online Retail II weekly net units (CC BY 4.0, Chen 2012, doi:10.24432/C5CG6D)"},
                "synthetic": {"inventory.csv": "invented stock snapshot on 2011-12-04",
                              "receipts.csv": "invented inbound receipts, December 2011",
                              "rules.csv": "invented buffer, pack, minimum and PURCHASE cost in GBP (not the UCI selling price)",
                              "supplier_messages.csv": "invented supplier messages about the invented receipts (for the AI-read step)"},
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared")
    ap.add_argument("out")
    ap.add_argument("--series", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260928)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.prepared), Path(a.out), a.series, a.seed)["files"]))

