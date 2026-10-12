"""Forecast worker acceptance (brief §12 items 1, 2, 4, 5). Runs in the forecasting environment
(requirements-forecast.txt); skipped where pandas is not installed (the engine's own environment)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("statsforecast")

from forecast_worker.baseline import baseline          # noqa: E402
from forecast_worker.fixtures.nordwerk import build    # noqa: E402
from forecast_worker.profile import profile            # noqa: E402

ANSWERS = {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("nw")
    build(d)
    return d


def _run(d: Path, out: Path, **params) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    baseline(d, out, {"answers": ANSWERS, **params})
    return json.loads((out / "baseline_report.json").read_text())


def test_the_profile_is_reproducible_and_row_order_does_not_change_it(data, tmp_path):
    shuffled = tmp_path / "shuffled"
    shutil.copytree(data, shuffled)
    for name in ("sales_2023_2025.csv", "sales_2025_2026.csv"):
        rows = list(csv.DictReader((shuffled / name).open()))
        rows.reverse()
        with (shuffled / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    a, b = profile(data), profile(shuffled)
    assert a == profile(data)
    strip = lambda p: {k: v for k, v in p.items() if k not in ("inputs", "inputs_sha256")}   # bytes differ, meaning not
    assert strip(a) == strip(b)


def test_missing_weeks_are_reported_as_missing_and_never_scored(data, tmp_path):
    prof = profile(data)
    missing = {(s["series_id"], w) for s in prof["series"] for w in s["missing_weeks"]}
    assert missing and all(s["zero_weeks"] >= 0 for s in prof["series"])
    _run(data, tmp_path / "o")
    fc = pd.read_csv(tmp_path / "o" / "backtest_forecasts.csv")
    scored_keys = set(zip(fc.loc[~fc["imputed"], "unique_id"], fc.loc[~fc["imputed"], "ds"]))
    assert not (scored_keys & missing)
    assert fc.loc[fc["imputed"]].shape[0] > 0          # they exist as model input, flagged


def test_the_job_refuses_without_the_owners_answers(data, tmp_path):
    with pytest.raises(SystemExit, match="answers"):
        baseline(data, tmp_path, {"answers": {"missing_weeks": "missing"}})


def test_future_values_cannot_change_a_past_forecast(data, tmp_path):
    """Leakage: change every sales value after the last evaluation origin; forecasts made at earlier origins
    must be byte-identical (the holdout and later weeks are never read by them)."""
    a = _run(data, tmp_path / "a")
    last_origin = max(a["population"]["origins"])
    changed = tmp_path / "changed"
    shutil.copytree(data, changed)
    for name, col in (("sales_2023_2025.csv", "qty"), ("sales_2025_2026.csv", "quantity")):
        rows = list(csv.DictReader((changed / name).open()))
        for r in rows:
            if r["week_start"] > last_origin:
                r[col] = str(int(r[col]) * 10 + 7)
        with (changed / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    _run(changed, tmp_path / "b")
    fa = pd.read_csv(tmp_path / "a" / "backtest_forecasts.csv")
    fb = pd.read_csv(tmp_path / "b" / "backtest_forecasts.csv")
    cols = ["model", "origin", "unique_id", "ds", "horizon", "yhat", "lo", "hi"]
    assert fa[cols].equals(fb[cols])


def test_models_are_compared_on_one_identical_population(data, tmp_path):
    r = _run(data, tmp_path / "c")
    comp = r["comparable"]
    ns = {m: v["n"] for m, v in comp["models"].items()}
    assert len(set(ns.values())) == 1 and set(ns) == {"naive", "seasonal_naive_52"}
    assert r["models"]["naive"]["overall"]["n"] > comp["population"]["points"]   # naive alone covers more
    assert len(r["skipped"]) > 0                                                   # short series said so


def test_an_undefined_metric_is_null_never_zero(data, tmp_path):
    r = _run(data, tmp_path / "d")
    for model in r["models"].values():
        for m in model["by_series"].values():
            assert m["mase"] is None or m["mase"] >= 0
    empty = {"n": 0, "mae": None, "bias": None, "mase": None, "coverage": None, "mean_width": None}
    for model in r["models"].values():
        so = model["stockout_weeks"]
        assert so["n"] > 0 or so == empty

