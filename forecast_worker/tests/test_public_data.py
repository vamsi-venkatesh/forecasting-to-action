"""Public data preparation and the fault pack: each seeded defect is noticed by the profile, the bike data never
carries same-day components or observed weather, and daily data runs through the same evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forecast_worker.benchmarks import fault_pack
from forecast_worker.profile import profile

ROOT = Path(__file__).resolve().parents[2]


def _daily(tmp_path: Path, days: int = 200) -> Path:
    import math
    d = tmp_path / "daily"
    d.mkdir()
    from datetime import date, timedelta
    with (d / "sales_daily.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series_id", "period_start", "available_at", "rentals", "unit"])
        for i in range(days):
            day = date(2011, 1, 1) + timedelta(days=i)
            y = 1000 + 300 * math.sin(i * 2 * math.pi / 7) + i * 2
            w.writerow(["bikes|total", day.isoformat(), f"{(day + timedelta(days=1)).isoformat()}T06:00:00+00:00", int(y), "rentals"])
    return d


def test_daily_data_is_profiled_as_daily_and_the_mapping_waits_for_a_person(tmp_path):
    r = profile(_daily(tmp_path))
    q = next(q for q in r["questions"] if q["id"] == "Q-mapping")
    assert r["frequency"] == "daily" and q["evidence"]["suggested"]["target"] == "rentals"
    assert "train" in q["blocks"] and "SUGGESTS" in q["question"]


def test_daily_evaluation_uses_a_seven_day_season(tmp_path):
    from forecast_worker.evaluate import evaluate
    d = _daily(tmp_path, 260)
    r = evaluate(d, tmp_path, {"frequency": "daily", "answers": {"quantity_columns": {"rentals": "t"}, "missing_weeks": "zero"},
                               "models": ["naive", "seasonal_naive_7"]})
    assert set(r["summary"]["cv_comparable"]) == {"naive", "seasonal_naive_7"}
    assert r["summary"]["selected"] in ("naive", "seasonal_naive_7")


def test_every_seeded_fault_is_noticed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    from datetime import date, timedelta
    with (src / "sales_uci_weekly.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series_id", "week_start", "available_at", "unit", "quantity"])
        for s in ("A", "B", "C"):
            for i in range(40):
                wk = date(2010, 1, 4) + timedelta(weeks=i)
                w.writerow([s, wk.isoformat(), f"{(wk + timedelta(days=7)).isoformat()}T00:00:00+00:00", "units", 10 + i])
    made = fault_pack.build(src, tmp_path / "faults")
    got = {case: profile(tmp_path / "faults" / case) for case in made}
    qid = lambda r: {q["id"] for q in r["questions"]}
    assert "Q-missing-weeks" in qid(got["missing_periods"])
    assert len(got["duplicate_events"]["duplicate_keys"]) >= 5
    assert got["ambiguous_dates"]["unparsed_rows"] >= 4
    pairs = next(q for q in got["changed_identifier"]["questions"] if q["id"] == "Q-series-continuity")["evidence"]["pairs"]
    assert {"ended": "A", "started": "A-NEW"}.items() <= {k: pairs[0][k] for k in ("ended", "started")}.items()
    assert "Q-units" in qid(got["invalid_units"])
    amb = next(q for q in got["ambiguous_target"]["questions"] if q["id"] == "Q-mapping")["evidence"]
    assert amb["suggested"]["target"] is None and amb["ambiguous"] and {"quantity", "quantity_shipped"} <= set(amb["target_candidates"])
    for case in made:
        assert json.loads((tmp_path / "faults" / case / "fault.json").read_text())["label"].startswith("FAULT TEST")


def test_bike_preparation_never_ships_same_day_components_or_observed_weather(tmp_path):
    z = Path.home() / "vvdex" / "data" / "uci-bike-sharing" / "bike_sharing_dataset.zip"
    if not z.exists():
        pytest.skip("the UCI Bike Sharing file is not cached on this machine")
    from forecast_worker.benchmarks.uci_bike_sharing import build
    m = build(z, tmp_path)
    with (tmp_path / "sales_daily.csv").open() as fh:
        cols = next(csv.reader(fh))
    assert cols == ["series_id", "period_start", "available_at", "rentals", "unit"]
    assert m["coverage"]["days"] == 731 and m["coverage"]["gaps"] == 0 and m["licence"] == "CC BY 4.0"


def test_identifier_columns_are_never_offered_as_the_quantity(tmp_path):
    d = tmp_path / "ids"
    d.mkdir()
    with (d / "sales_x.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series_id", "product_id", "week_start", "available_at", "unit", "quantity"])
        for i in range(10):
            w.writerow(["84077", "84077", f"2010-01-{4 + 7 * (i % 3):02d}", "2010-02-01T00:00:00+00:00", "units", 5 + i])
    q = next(q for q in profile(d)["questions"] if q["id"] == "Q-mapping")
    assert q["evidence"]["target_candidates"] == ["quantity"] and q["evidence"]["suggested"]["target"] == "quantity"

