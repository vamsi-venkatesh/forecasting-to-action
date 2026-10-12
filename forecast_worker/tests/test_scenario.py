"""Shipment-delay scenario: moving a confirmed receipt later changes supply only, never the demand forecast."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forecast_worker.operate import run_plan, run_scenario


def _w(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    weeks = ["2026-10-05", "2026-10-12", "2026-10-19", "2026-10-26"]
    _w(d / "issued_forecast.csv", [{"issued_at": "2026-09-30T08:00:00+00:00", "model": "ets", "origin_week": "2026-09-21",
                                    "series_id": "P-1|AT", "target_week": w, "horizon": i, "yhat": "250", "lo": "0", "hi": "0"}
                                   for i, w in enumerate(weeks)])
    _w(d / "inventory.csv", [{"series_id": "P-1|AT", "snapshot_time": "2026-09-27T23:00:00+00:00",
                              "available_at": "2026-09-28T06:00:00+00:00", "usable_on_hand": "600", "committed": "0"}])
    _w(d / "receipts.csv", [{"receipt_id": "R1", "series_id": "P-1|AT", "quantity": "200", "expected_date": "2026-10-13",
                             "status": "confirmed"},
                            {"receipt_id": "R2", "series_id": "P-1|AT", "quantity": "50", "expected_date": "2026-10-13",
                             "status": "unconfirmed"}])
    _w(d / "rules.csv", [{"series_id": "P-1|AT", "rule_version": "1", "buffer_units": "100", "pack_size": "50",
                          "min_order_units": "100", "unit_cost_eur": "5.00", "currency": "EUR"}])
    base = tmp_path / "base"
    base.mkdir()
    run_plan(d, base, {})
    (d / "plan.json").write_bytes((base / "plan.json").read_bytes())
    return d, tmp_path


def test_a_one_week_delay_moves_the_shortage_earlier_and_leaves_demand_alone(data):
    d, tmp = data
    out = tmp / "scn"
    out.mkdir()
    res = run_scenario(d, out, {"receipt_shifts": {"R1": 1}})
    base = json.loads((d / "plan.json").read_text())["plans"][0]
    scen = json.loads((out / "plan.json").read_text())["plans"][0]
    assert [t["requirement"] for t in base["trajectory"]] == [t["requirement"] for t in scen["trajectory"]]   # demand unchanged
    diff = json.loads((out / "scenario_diff.json").read_text())
    row = diff["changed_series"][0]
    assert diff["scenario"]["moved"][0] == {"receipt_id": "R1", "series_id": "P-1|AT", "from": "2026-10-13",
                                            "to": "2026-10-20", "quantity": "200"}
    # one week late, still inside the window: week 2 has 200 less (the whole buffer is used), the total is unchanged
    assert row["weeks_changed"] == ["2026-10-12"]
    assert row["availability"][0]["base_available"] == "800" and row["availability"][0]["scenario_available"] == "600"
    assert row["base_first_short_week"] == row["scenario_first_short_week"] == "2026-10-19"
    assert row["base_proposed_quantity"] == row["scenario_proposed_quantity"] == "300"          # the brief's fixture
    assert row["base_proposed_value"] == "1500.00" and res["summary"]["changed_series"] == 1
    assert "needs its own approval" in diff["label"]


def test_an_unconfirmed_or_unknown_receipt_cannot_be_the_scenario(data):
    d, tmp = data
    with pytest.raises(SystemExit, match="not a confirmed"):
        run_scenario(d, tmp / "x", {"receipt_shifts": {"R2": 1}})
    with pytest.raises(SystemExit, match="unknown receipt"):
        run_scenario(d, tmp / "y", {"receipt_shifts": {"R9": 1}})
    with pytest.raises(SystemExit, match="1 to 8 weeks"):
        run_scenario(d, tmp / "z", {"receipt_shifts": {"R1": 20}})


def test_a_delay_past_the_window_drops_the_receipt_and_raises_the_proposal(data):
    d, tmp = data
    out = tmp / "late"
    out.mkdir()
    run_scenario(d, out, {"receipt_shifts": {"R1": 3}})
    row = json.loads((out / "scenario_diff.json").read_text())["changed_series"][0]
    assert row["scenario_excluded_receipts"][0] == {"receipt_id": "R1", "why": "arrives after the protection window (late)"}
    assert (row["base_proposed_quantity"], row["scenario_proposed_quantity"]) == ("300", "500")
    assert (row["base_proposed_value"], row["scenario_proposed_value"]) == ("1500.00", "2500.00")
    # 600 on hand, need 350 / 600 / 850 / 1100: week 2 is exactly at the buffer (gap 0), short from week 3
    assert row["scenario_first_short_week"] == "2026-10-19" and row["availability"][-1]["scenario_gap"] == "500"

