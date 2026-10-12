"""Forecast vs actual compares only actuals that arrived AFTER the forecast was issued (release review 2026-09-28:
points with horizon >= 1 were scored even when their actual was already known at issue time)."""
import csv
import json

from forecast_worker.ingest import run


def _w(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_only_actuals_that_arrived_after_the_issue_are_compared(tmp_path):
    data, out = tmp_path / "in", tmp_path / "out"
    data.mkdir(), out.mkdir()
    row = lambda wk, avail, q: {"series_id": "A", "week_start": wk, "available_at": avail, "quantity": q}
    _w(data / "sales_feed.csv", [row("2026-01-05", "2026-01-12T06:00:00+00:00", "10")])
    _w(data / "batch_0002_new.csv", [row("2026-01-12", "2026-01-19T06:00:00+00:00", "12"),     # known BEFORE the issue
                                     row("2026-01-19", "2026-01-26T06:00:00+00:00", "14")])    # arrived after it
    fc = lambda wk, h: {"issued_at": "2026-01-20T08:00:00+00:00", "model": "ets", "series_id": "A", "target_week": wk,
                        "horizon": h, "yhat": "11", "lo": "8", "hi": "15"}
    _w(data / "issued_forecast.csv", [fc("2026-01-12", 1), fc("2026-01-19", 1)])
    res = run(data, out, {"target": "quantity", "date": "week_start"})
    points = json.loads((out / "forecast_vs_actual.json").read_text())["points"]
    assert [p["week"] for p in points] == ["2026-01-19"] and res["summary"]["observed_actuals"] is True

