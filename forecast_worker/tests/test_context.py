"""Context features (brief 7A): code does the dates and the aggregation; nothing from the future enters;
an unclassified message is unknown, never "no event"; a supply delay is never a demand signal."""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from forecast_worker.context import experiment
from forecast_worker.context.features import issue_time, items, training_frame, value
from forecast_worker.context.fixture import build, week_start
from forecast_worker.context.parse import read, rule_classify, spans

UTC = timezone.utc


def _msg(mid, text, at):
    return {"message_id": mid, "text": text, "available_at": at.isoformat()}


def _its(msgs, classes):
    return items([read(m) for m in msgs], classes)


PROMO = "Marketing has confirmed the Seal kit promotion for Austria in KW 10 and KW 11 (2025)."
WEEK10 = date.fromisocalendar(2025, 10, 1)


def test_the_three_written_week_forms_are_read_by_code():
    assert spans("in KW 10 and KW 11 (2025)") == [(WEEK10, WEEK10 + timedelta(weeks=1))]
    assert spans("in weeks 10-11 of 2025") == [(WEEK10, WEEK10 + timedelta(weeks=1))]
    assert spans("for two weeks from Monday 3 March 2025") == [(WEEK10, WEEK10 + timedelta(weeks=1))]


def test_a_message_counts_only_once_it_was_available():
    at = datetime(2025, 2, 20, 9, tzinfo=UTC)
    its = _its([_msg("m1", PROMO, at)], {"m1": {"event_type": "promotion", "status": "confirmed", "abstained": False}})
    assert value(its, "P-200|AT", WEEK10, at - timedelta(seconds=1))["promo_confirmed"] == 0
    assert value(its, "P-200|AT", WEEK10, at)["promo_confirmed"] == 1
    assert value(its, "P-200|DE-N", WEEK10, at)["promo_confirmed"] == 0          # other series untouched


def test_training_rows_never_see_a_message_that_arrived_after_their_issue_time():
    t = WEEK10
    cut = issue_time(t - timedelta(weeks=1))
    cls = {"m1": {"event_type": "promotion", "status": "confirmed", "abstained": False}}
    late = training_frame(_its([_msg("m1", PROMO, cut + timedelta(seconds=1))], cls), ["P-200|AT"], [t])
    on_time = training_frame(_its([_msg("m1", PROMO, cut)], cls), ["P-200|AT"], [t])
    assert late["promo_confirmed"].tolist() == [0] and on_time["promo_confirmed"].tolist() == [1]


def test_the_latest_message_wins_and_a_postponement_moves_the_slot():
    a, b = datetime(2025, 1, 10, tzinfo=UTC), datetime(2025, 2, 1, tzinfo=UTC)
    msgs = [_msg("m1", PROMO, a),
            _msg("m2", "The Seal kit promotion for Austria planned in KW 10 and KW 11 (2025) is postponed to KW 20 and KW 21 (2025).", b)]
    cls = {"m1": {"event_type": "promotion", "status": "confirmed", "abstained": False},
           "m2": {"event_type": "promotion", "status": "postponed", "abstained": False}}
    its = _its(msgs, cls)
    assert value(its, "P-200|AT", WEEK10, a) == {"promo_confirmed": 1, "promo_tentative": 0, "supply_delay": 0, "ctx_missing": 0}
    assert value(its, "P-200|AT", WEEK10, b)["promo_confirmed"] == 0
    assert value(its, "P-200|AT", date.fromisocalendar(2025, 20, 1), b)["promo_tentative"] == 1


def test_an_unclassified_message_is_unknown_not_no_event():
    its = _its([_msg("m1", PROMO, datetime(2025, 1, 10, tzinfo=UTC))], {"m1": {"abstained": True, "event_type": None}})
    v = value(its, "P-200|AT", WEEK10, datetime(2025, 3, 1, tzinfo=UTC))
    assert v["ctx_missing"] == 1 and v["promo_confirmed"] == 0


def test_a_supply_delay_is_a_supply_feature_only():
    text = "Supplier notice: the inbound Seal kit batch for Austria is delayed, now expected in KW 10 and KW 11 (2025)."
    its = _its([_msg("m1", text, datetime(2025, 1, 10, tzinfo=UTC))],
               {"m1": {"event_type": "supply_constraint", "status": "not_a_promotion", "abstained": False}})
    v = value(its, "P-200|AT", WEEK10, datetime(2025, 3, 1, tzinfo=UTC))
    assert v == {"promo_confirmed": 0, "promo_tentative": 0, "supply_delay": 1, "ctx_missing": 0}


def test_every_fixture_message_is_written_before_the_weeks_it_announces(tmp_path):
    build(tmp_path)
    with (tmp_path / "context_messages.csv").open() as fh:
        for m in csv.DictReader(fh):
            if m["label_event_type"] == "promotion":
                first = min(a for a, _ in spans(m["text"]))
                assert datetime.fromisoformat(m["available_at"]).date() < first, m["message_id"]


def test_an_arm_with_unclassified_messages_is_not_run_with_zeros(tmp_path, monkeypatch):
    build(tmp_path)
    with (tmp_path / "context_messages.csv").open() as fh:
        ids = [m["message_id"] for m in csv.DictReader(fh)]
    cls = {i: {"event_type": "other", "status": "not_a_promotion", "abstained": False} for i in ids[1:]}   # one missing
    (tmp_path / "classes-jev.json").write_text(json.dumps({"classes": cls}))
    monkeypatch.setattr(experiment, "CV_ORIGINS", [60])
    monkeypatch.setattr(experiment, "HOLDOUT_ORIGINS", [140])
    res = experiment.run(tmp_path, tmp_path)
    assert res["unavailable_arms"]["jev"] == "1 messages unclassified"
    assert "jev" not in res["arms"] and "control_shuffled" not in res["arms"]


def test_the_keyword_rules_are_fooled_by_negation_as_documented():
    assert rule_classify("The Seal kit promo for Austria is going ahead, not cancelled after all.")["status"] == "cancelled"


def test_week_start_is_a_monday():
    assert all(week_start(i).weekday() == 0 for i in range(0, 156, 13))


def test_region_codes_are_read_only_as_whole_upper_case_tokens():
    from forecast_worker.context.parse import region
    assert region("The planned Seal kit campaign in DE-S is cancelled.") == "DE-S"
    assert region("Delivery for AT is late.") == "AT"
    assert region("We met at the fair and looked at prices.") is None
    assert region("Orders from DE-SOUTH or ATX are not ours.") is None

