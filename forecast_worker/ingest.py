"""Recurring source, worker side (Release 3): merge newly delivered batch files into the next VERSION of a project's
prepared sales data, and compare earlier issued forecasts with the actuals that have now arrived.

Inputs (staged by the engine): the current prepared data `sales_feed.csv` (absent for the first batch), the new batch
files `batch_<n>_<name>.csv` in delivery order, and optionally `issued_forecast.csv` (the last issued forecast).
Params: {"target": column, "date": column, "frequency": "weekly"|"daily", "seen": [sha256 of earlier batches]}.

Rules (each outcome is reported, nothing is dropped silently):
- a batch whose bytes were delivered before is a DUPLICATE and changes nothing;
- a batch whose columns differ from the confirmed mapping (schema drift) is REJECTED with the difference;
- a batch with unparseable dates or quantities is REJECTED with the first bad row;
- rows for periods not seen before are ADDED; rows for a period already present with a different value are a
  CORRECTION (old and new value recorded; the later delivery wins);
- a batch that is entirely older than what is already held is accepted only as corrections (late records);
- dates are calendar dates of the source (its time zone is stated in the source configuration); no clock shifting.
Outputs: sales_feed.csv (the new version; the forecasting jobs read sales_*.csv), ingest_report.json, and forecast_vs_actual.json when an issued forecast and
new actuals overlap (error per series and week, forecast kept as it was issued).
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path


def _rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def run(data: Path, out: Path, params: dict) -> dict:
    target, date_col = params["target"], params["date"]
    seen = set(params.get("seen") or [])
    base_cols, base = _rows(data / "sales_feed.csv") if (data / "sales_feed.csv").exists() else ([], [])
    held = {(r["series_id"], r[date_col]): r for r in base}
    max_before = max((r[date_col] for r in base), default="")
    report = {"batches": [], "added": 0, "corrections": [], "max_period_before": max_before or None}
    cols_expected = base_cols or None
    for f in sorted(data.glob("batch_*.csv")):
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        entry = {"file": f.name.split("_", 2)[-1], "sha256": sha}
        if sha in seen:
            report["batches"].append({**entry, "status": "duplicate", "why": "the same bytes were delivered before"})
            continue
        cols, rows = _rows(f)
        need = {"series_id", date_col, target}
        if not need <= set(cols) or (cols_expected and set(cols) != set(cols_expected)):
            report["batches"].append({**entry, "status": "rejected",
                                      "why": f"columns differ from the confirmed mapping: missing {sorted(need - set(cols))}, "
                                             f"unexpected {sorted(set(cols) - set(cols_expected or cols))}"})
            continue
        bad = None
        for i, r in enumerate(rows, start=2):
            try:
                date.fromisoformat(r[date_col])
                float(r[target])
            except (ValueError, TypeError):
                bad = f"row {i}: {r.get(date_col)!r} / {r.get(target)!r}"
                break
        if bad:
            report["batches"].append({**entry, "status": "rejected", "why": f"unreadable value at {bad}"})
            continue
        added = corrected = 0
        for r in rows:
            k = (r["series_id"], r[date_col])
            if k in held:
                if held[k][target] != r[target]:
                    report["corrections"].append({"series_id": k[0], "period": k[1], "old": held[k][target], "new": r[target],
                                                  "batch": entry["file"]})
                    held[k] = r
                    corrected += 1
            else:
                held[k] = r
                added += 1
        cols_expected = cols_expected or cols
        seen.add(sha)
        report["added"] += added
        report["batches"].append({**entry, "status": "accepted", "rows": len(rows), "added": added, "corrected": corrected,
                                  "late_only": bool(rows) and max(r[date_col] for r in rows) <= max_before})
    merged = sorted(held.values(), key=lambda r: (r["series_id"], r[date_col]))
    if not merged:
        raise SystemExit("no accepted data: every batch was a duplicate or was rejected")
    with (out / "sales_feed.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols_expected or list(merged[0]))
        w.writeheader()
        w.writerows(merged)
    report["max_period_after"] = max(r[date_col] for r in merged)
    report["rows"] = len(merged)
    report["accepted_shas"] = sorted(b["sha256"] for b in report["batches"] if b["status"] == "accepted")
    (out / "ingest_report.json").write_text(json.dumps(report, indent=1) + "\n")
    outputs = ["sales_feed.csv", "ingest_report.json"]
    fc_path = data / "issued_forecast.csv"
    if fc_path.exists():                      # observe later actuals against the forecast AS IT WAS ISSUED
        _, fc = _rows(fc_path)
        rows = []
        for r in fc:
            a = held.get((r["series_id"], r["target_week"]))
            # a point is an honest forecast only when its actual arrived AFTER the issue (horizon 0, the data-lag week,
            # included); without an available_at column only horizon >= 1 is compared.
            if a is None:
                continue
            if a.get("available_at") and r.get("issued_at"):      # the arrival time decides: known before the issue = not a forecast
                if datetime.fromisoformat(a["available_at"]) <= datetime.fromisoformat(r["issued_at"]):
                    continue
            elif int(r.get("horizon", 1)) < 1:                    # no arrival time: only weeks after the data lag
                continue
            y, yhat = float(a[target]), float(r["yhat"])
            rows.append({"series_id": r["series_id"], "week": r["target_week"], "horizon": int(r["horizon"]),
                         "forecast": round(yhat, 3), "actual": y, "error": round(yhat - y, 3),
                         "inside_range": float(r["lo"]) <= y <= float(r["hi"])})
        if rows:
            (out / "forecast_vs_actual.json").write_text(json.dumps({
                "issued_at": fc[0]["issued_at"], "model": fc[0]["model"], "points": rows,
                "mean_absolute_error": round(sum(abs(x["error"]) for x in rows) / len(rows), 3),
                "note": "the issued forecast is compared unchanged, only for actuals that arrived after it was issued (horizon 0 = the data-lag week); ranges are not validated (interval study 1)"}, indent=1) + "\n")
            outputs.append("forecast_vs_actual.json")
    accepted = sum(b["status"] == "accepted" for b in report["batches"])
    return {"outputs": outputs, "summary": {"accepted": accepted, "duplicates": sum(b["status"] == "duplicate" for b in report["batches"]),
                                            "rejected": sum(b["status"] == "rejected" for b in report["batches"]),
                                            "added": report["added"], "corrections": len(report["corrections"]),
                                            "max_period": report["max_period_after"], "accepted_shas": report["accepted_shas"],
                                            "observed_actuals": "forecast_vs_actual.json" in outputs}}

