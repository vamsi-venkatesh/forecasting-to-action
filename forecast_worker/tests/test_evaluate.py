"""Phase 2 acceptance: the selection is frozen before the holdout, a baseline win stays visible, candidates
(LightGBM included) cannot see the future. Runs in the forecasting environment."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("mlforecast")

from forecast_worker.evaluate import evaluate, select     # noqa: E402
from forecast_worker.fixtures.nordwerk import build       # noqa: E402

ANSWERS = {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("nw")
    build(d)
    return d


def _eval(d: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    return evaluate(d, out, {"answers": ANSWERS})["summary"]


def _bump(d: Path, target: Path, after: str) -> None:
    shutil.copytree(d, target)
    for name, col in (("sales_2023_2025.csv", "qty"), ("sales_2025_2026.csv", "quantity")):
        rows = list(csv.DictReader((target / name).open()))
        for r in rows:
            if r["week_start"] > after:
                r[col] = str(int(r[col]) * 7 + 11)
        with (target / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def test_the_holdout_cannot_change_the_frozen_selection(data, tmp_path):
    a = _eval(data, tmp_path / "a")
    report = json.loads((tmp_path / "a" / "evaluation_report.json").read_text())
    last_cv_week = max(r["ds"] for r in csv.DictReader((tmp_path / "a" / "evaluation_forecasts.csv").open()) if r["split"] == "cv")
    _bump(data, tmp_path / "changed", after=last_cv_week)             # rewrite every holdout week
    b = _eval(tmp_path / "changed", tmp_path / "b")
    assert (tmp_path / "a" / "selection_frozen.json").read_bytes() == (tmp_path / "b" / "selection_frozen.json").read_bytes()
    assert a["selected"] == b["selected"] and a["frozen_sha256"] == b["frozen_sha256"]
    assert a["holdout_comparable"] != b["holdout_comparable"]        # the holdout itself did change
    assert report["population"]["holdout_from"] > last_cv_week


def test_no_candidate_sees_the_future(data, tmp_path):
    _eval(data, tmp_path / "a")
    fa = pd.read_csv(tmp_path / "a" / "evaluation_forecasts.csv")
    cv = fa[fa["split"] == "cv"]
    last_origin = cv["origin"].max()
    _bump(data, tmp_path / "changed", after=last_origin)
    _eval(tmp_path / "changed", tmp_path / "b")
    fb = pd.read_csv(tmp_path / "b" / "evaluation_forecasts.csv")
    cols = ["model", "origin", "unique_id", "ds", "horizon", "yhat", "lo", "hi"]
    assert set(cv["model"]) >= {"lgbm_lags", "ets", "naive"}
    pd.testing.assert_frame_equal(cv[cols].reset_index(drop=True), fb[fb["split"] == "cv"][cols].reset_index(drop=True))


@pytest.mark.parametrize("candidate,expected", [(0.74, "seasonal_naive_52"), (0.70, "lgbm_lags")])
def test_a_candidate_must_clear_the_margin_or_the_baseline_stays(candidate, expected):
    m = lambda v: {"n": 10, "mase": v, "mae": 1, "bias": 0, "coverage": 0.8, "mean_width": 1}
    decision = select({"naive": m(0.95), "seasonal_naive_52": m(0.76), "lgbm_lags": m(candidate), "ets": m(0.81)})
    assert decision["selected"] == expected
    if expected != "lgbm_lags":
        assert "baseline stays" in decision["reason"]


def test_a_gap_below_the_datas_measured_lag_is_refused_never_trusted():
    import pandas as pd
    import pytest
    from forecast_worker.baseline import check_gap, required_gap
    week = pd.Timestamp("2026-01-05")
    lag9 = pd.DataFrame({"ds": [week], "available_at": [pd.Timestamp("2026-01-14T09:00", tz="UTC")]})   # 9 days + 9 h
    assert required_gap(lag9) == (1, 2.38)
    with pytest.raises(SystemExit, match="gap_weeks=0 is shorter than the data's own lag"):
        check_gap(lag9, 0)
    assert check_gap(lag9, 1)["measured_gap_weeks"] == 1
    on_time = pd.DataFrame({"ds": [week], "available_at": [pd.Timestamp("2026-01-12T00:00", tz="UTC")]})  # the Monday after
    assert required_gap(on_time) == (0, 0.0)
    slow = pd.DataFrame({"ds": [week], "available_at": [pd.Timestamp("2026-01-27T00:00", tz="UTC")]})    # 15 days after the end
    assert required_gap(slow)[0] == 3


def test_baseline_and_evaluation_themselves_refuse_a_gap_below_the_lag(tmp_path):
    from forecast_worker.baseline import baseline
    d = tmp_path / "nw"
    build(d)
    for run in (baseline, evaluate):
        with pytest.raises(SystemExit, match="shorter than the data's own lag"):
            run(d, tmp_path, {"answers": ANSWERS, "gap_weeks": 0})


def test_intervals_are_reported_as_not_validated_until_a_study_says_otherwise(tmp_path):
    from forecast_worker.evaluate import INTERVALS
    d = tmp_path / "nw"
    build(d)
    evaluate(d, tmp_path, {"answers": ANSWERS})
    report = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert report["intervals"] == INTERVALS
    assert INTERVALS["lgbm_lags"]["status"] == "not_validated"          # interval studies 1 and 2 -> not validated
    for ev in INTERVALS["lgbm_lags"]["evidence"].split(", "):
        study = json.loads((Path(__file__).resolve().parents[2] / ev).read_text())
        assert study["verdict"] == "NOT validated"                         # the label follows every recorded verdict


def test_insufficient_history_is_stated_with_its_numbers_at_the_exact_boundary():
    """Brief: 'insufficient history' is an honest outcome. With too little data the first origin fell before the data
    and the lag check then said, wrongly, that no row had an available_at."""
    import pandas as pd
    from forecast_worker.baseline import enough_history
    from forecast_worker.calendar import WEEKLY

    def panel(weeks):
        ds = pd.date_range("2026-01-05", periods=weeks, freq="W-MON")
        return pd.DataFrame({"unique_id": "A", "ds": ds, "y": 1.0})

    def first_origin(p, h=4, gap=1, holdout=8, n=12):
        return p["ds"].max() - WEEKLY.periods(holdout) - WEEKLY.periods(h + gap + n - 1)
    ok = panel(32)
    enough_history(ok, first_origin(ok), WEEKLY, h=4, gap=1, holdout=8, n_origins=12)        # exactly enough
    short = panel(31)
    with pytest.raises(SystemExit, match=r"insufficient history: the longest series has 31 weeks; this evaluation needs at least 32"):
        enough_history(short, first_origin(short), WEEKLY, h=4, gap=1, holdout=8, n_origins=12)

