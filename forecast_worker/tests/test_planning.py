"""Planning arithmetic (brief §8): the exact 300-unit / EUR 1,500 fixture, and the variants that must hold or
qualify the proposal. Pure code; runs in any Python environment."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from forecast_worker.planning import Receipt, Rule, plan

W = [date(2026, 10, 5) + timedelta(weeks=i) for i in range(4)]
RULE = Rule(buffer_units=D(100), pack_size=D(50), min_order_units=D(100), unit_cost=D("5"))


def test_the_brief_fixture_1000_600_200_100_gives_300_units_and_eur_1500():
    p = plan("S", W, [D(250)] * 4, usable_on_hand=D(600), commitments=D(0),
             receipts=[Receipt("R1", D(200), W[0], True)], rule=RULE)
    assert p.shortfall == D(300) and p.proposed_quantity == D(300)
    assert p.proposed_value == D("1500.00") and p.currency == "EUR" and not p.qualified


def test_a_receipt_after_the_shortage_does_not_hide_it():
    # same totals, but the 200 arrive in week 4: week 3 is already 250 short before they land
    p = plan("S", W, [D(250)] * 4, usable_on_hand=D(600), commitments=D(0),
             receipts=[Receipt("R1", D(200), W[3], True)], rule=RULE)
    gaps = [D(t["gap"]) for t in p.trajectory]
    assert gaps == [D(-250), D(0), D(250), D(300)]       # need 350/600/850/1100 vs available 600/600/600/800
    assert p.first_short_week == W[2].isoformat()
    assert p.shortfall == D(300)


@pytest.mark.parametrize("receipt,why", [(Receipt("R1", D(200), W[0], False), "unconfirmed"),
                                         (Receipt("R1", D(200), None, True), "no expected date"),
                                         (Receipt("R1", D(200), W[3] + timedelta(days=10), True), "late")])
def test_unconfirmed_undated_or_late_receipts_never_count_and_are_named(receipt, why):
    p = plan("S", W, [D(250)] * 4, usable_on_hand=D(600), commitments=D(0), receipts=[receipt], rule=RULE)
    assert p.shortfall == D(500) and p.proposed_quantity == D(500)
    assert any(why in q for q in p.qualified) and p.excluded_receipts[0]["receipt_id"] == "R1"


def test_a_missing_price_qualifies_the_proposal_and_is_never_guessed():
    p = plan("S", W, [D(250)] * 4, usable_on_hand=D(600), commitments=D(0),
             receipts=[Receipt("R1", D(200), W[0], True)], rule=Rule(D(100), D(50), D(100), None))
    assert p.proposed_quantity == D(300) and p.proposed_value is None
    assert any("price" in q for q in p.qualified)


def test_packs_and_minimums_and_commitments():
    p = plan("S", W, [D(10)] * 4, usable_on_hand=D(60), commitments=D(20), receipts=[], rule=RULE)
    assert p.shortfall == D(100)                          # 40 + 100 buffer - (60 - 20)
    assert p.proposed_quantity == D(100)
    small = plan("S", W, [D(10)] * 4, usable_on_hand=D(125), commitments=D(0), receipts=[], rule=RULE)
    assert small.shortfall == D(15) and small.proposed_quantity == D(100)     # minimum order
    none = plan("S", W, [D(10)] * 4, usable_on_hand=D(500), commitments=D(0), receipts=[], rule=RULE)
    assert none.shortfall == D(0) and none.proposed_quantity == D(0) and none.proposed_value is None


def test_money_is_exact_decimal():
    p = plan("S", W, [D(1)] * 4, usable_on_hand=D(0), commitments=D(0), receipts=[],
             rule=Rule(D(0), D(1), D(1), D("0.10")))
    assert p.proposed_value == D("0.40")                 # 4 x 0.10, not 0.4000000001


def test_inputs_that_cannot_be_a_plan_are_refused():
    with pytest.raises(ValueError):
        plan("S", W, [D(1)] * 3, usable_on_hand=D(0), commitments=D(0), receipts=[], rule=RULE)
    with pytest.raises(ValueError):
        plan("S", W, [D(-1)] * 4, usable_on_hand=D(0), commitments=D(0), receipts=[], rule=RULE)


def test_an_issued_forecast_uses_only_rows_available_at_issue_time(tmp_path):
    pytest.importorskip("pandas")
    import json
    from forecast_worker.fixtures.nordwerk import build
    from forecast_worker.operate import issue
    d = tmp_path / "nw"
    build(d)
    answers = {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}
    out = tmp_path / "o"
    out.mkdir()
    issue(d, out, {"model": "naive", "answers": answers, "issued_at": "2026-09-28T09:00:00+00:00"})
    m = json.loads((out / "issue_manifest.json").read_text())
    import csv as _csv
    sales = [r for f in ("sales_2023_2025.csv", "sales_2025_2026.csv") for r in _csv.DictReader((d / f).open())]
    for sid, week in m["last_known_week"].items():
        rows = [r for r in sales if r["series_id"] == sid]
        known = [r["week_start"] for r in rows if r["available_at"] <= "2026-09-28T09:00:00+00:00"]
        assert week == max(known)                     # each series stops at ITS last available week
    assert len(set(m["last_known_week"].values())) > 1          # in this fixture availability differs by series
    fc = list(_csv.DictReader((out / "issued_forecast.csv").open()))
    targets = {r["target_week"] for r in fc}
    assert len(targets) == 5 and all(sum(1 for r in fc if r["target_week"] == t) == 12 for t in targets)
    assert sorted({int(r["horizon"]) for r in fc}) == [0, 1, 2, 3, 4]          # the lag week is horizon 0
    with pytest.raises(SystemExit, match="approved model"):
        issue(d, out, {"answers": answers})


def test_the_plan_counts_the_current_week_between_the_stock_snapshot_and_the_first_scored_week(tmp_path):
    pytest.importorskip("pandas")
    import json
    from forecast_worker.fixtures.nordwerk import build
    from forecast_worker.operate import issue, run_plan
    d = tmp_path / "nw"
    build(d)
    answers = {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}
    (tmp_path / "i").mkdir()
    issue(d, tmp_path / "i", {"model": "naive", "answers": answers, "issued_at": "2026-09-28T09:00:00+00:00"})
    plan_in = tmp_path / "p_in"
    plan_in.mkdir()
    import shutil
    shutil.copy(tmp_path / "i" / "issued_forecast.csv", plan_in / "issued_forecast.csv")
    for f in ("inventory.csv", "receipts.csv", "rules.csv"):
        shutil.copy(d / f, plan_in / f)
    (tmp_path / "p").mkdir()
    run_plan(plan_in, tmp_path / "p", {})
    doc = json.loads((tmp_path / "p" / "plan.json").read_text())
    for p in doc["plans"]:
        assert p["weeks"][0] == "2026-09-28"          # snapshot ends week 21 Sep; week 28 Sep is consumed from it


def test_a_stale_stock_snapshot_is_held_not_planned_on(tmp_path):
    """Brief fault: stale snapshots. A count five weeks old does not contain five weeks of sales; planning on it would
    order too little. That series is held with the snapshot's age; the others plan as before."""
    pytest.importorskip("pandas")
    import csv
    import json
    import shutil
    from forecast_worker.fixtures.nordwerk import build
    from forecast_worker.operate import issue, run_plan
    d = tmp_path / "nw"
    build(d)
    answers = {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}
    (tmp_path / "i").mkdir()
    issue(d, tmp_path / "i", {"model": "naive", "answers": answers, "issued_at": "2026-09-28T09:00:00+00:00"})
    plan_in = tmp_path / "p_in"
    plan_in.mkdir()
    shutil.copy(tmp_path / "i" / "issued_forecast.csv", plan_in / "issued_forecast.csv")
    for f in ("receipts.csv", "rules.csv"):
        shutil.copy(d / f, plan_in / f)
    with (d / "inventory.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    old = rows[0]["series_id"]
    for r in rows:
        if r["series_id"] == old:
            r["snapshot_time"], r["available_at"] = "2026-08-23T23:00:00+00:00", "2026-08-24T06:00:00+00:00"
    with (plan_in / "inventory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "p").mkdir()
    run_plan(plan_in, tmp_path / "p", {})
    doc = json.loads((tmp_path / "p" / "plan.json").read_text())
    held = {h["series_id"]: h["why"] for h in doc["held"]}
    assert old in held and "35 days old" in held[old] and old not in {p["series_id"] for p in doc["plans"]}
    assert len(doc["plans"]) == len({r["series_id"] for r in rows}) - 1 and doc["max_snapshot_age_days"] == 7

