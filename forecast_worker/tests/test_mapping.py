"""An unfamiliar file goes through the same pipeline once a person confirms what its columns mean (brief 2026-09-28:
the decisive generalisation test is an unseen upload, not another pre-shaped sample)."""
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from forecast_worker import run as worker
from forecast_worker.mapping import canonicalize, propose


def _shop_log(path: Path, days: int = 420) -> None:
    """A shop's transaction log: day-first dates, several sales per day, a price column, no arrival time."""
    rng = random.Random(7)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Order date", "Store", "SKU", "Units sold", "Unit price"])
        for i in range(days):
            d = date(2025, 1, 1) + timedelta(days=i)
            for store in ("Berlin", "Hamburg"):
                for sku in ("A-100", "B-200"):
                    for _ in range(rng.randint(1, 3)):
                        base = 5 + (3 if d.weekday() >= 5 else 0) + (2 if store == "Berlin" else 0)
                        w.writerow([d.strftime("%d/%m/%Y"), store, sku, rng.randint(0, base), f"{rng.uniform(2, 9):.2f}"])


def _job(kind, files, answers=None, **params):
    return {"kind": kind, "inputs": {p.name: str(p) for p in files}, "params": {**({"answers": answers} if answers else {}), **params}}


def test_the_profile_suggests_a_mapping_and_asks_before_reading_further(tmp_path):
    _shop_log(tmp_path / "shop_export.csv")
    prop = propose(tmp_path)
    s = prop["suggested"]
    assert s["date"] == "Order date" and s["date_order"] == "dmy"               # 13/01 etc. settle the order
    assert s["series"] == ["Store", "SKU"] and s["quantity"] == "Units sold"     # not the price
    assert s["available_at"] is None and s["frequency"] == "daily"
    out = tmp_path / "o"; out.mkdir()
    res = worker.run_profile(_job("profile", [tmp_path / "shop_export.csv"]), out)
    assert [q["id"] for q in res["questions"]] == ["Q-columns"] and res["summary"]["stage"] == "mapping"


def test_a_confirmed_mapping_runs_the_same_evaluation_as_any_standard_file(tmp_path):
    src = tmp_path / "shop_export.csv"
    _shop_log(src)
    answers = {"columns": {"date": "Order date", "date_order": "dmy", "series": ["Store", "SKU"], "available_at": None, "lag_days": 2},
               "quantity_columns": {"Units sold": "target"}, "frequency": "weekly", "missing_weeks": "zero"}
    out = tmp_path / "p"; out.mkdir()
    prof = worker.run_profile(_job("profile", [src], json.loads(json.dumps(answers))), out)
    assert prof["summary"]["series"] == 4 and "Q-mapping" not in prof["summary"]["questions"]
    doc = json.loads((out / "profile.json").read_text())
    assert doc["mapping"]["rule"]["same period rows"] == "added up" and doc["mapping"]["files"][0]["unread_dates"] == 0
    ev = tmp_path / "e"; ev.mkdir()
    res = worker.run_evaluate(_job("evaluate", [src], json.loads(json.dumps(answers)), frequency="weekly"), ev)
    assert res["summary"]["selected"] and res["summary"]["origins"] == 12


def test_ambiguous_dates_are_never_guessed(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("day,units\n03/04/2026,5\n04/05/2026,6\n05/06/2026,7\n")
    assert propose(tmp_path)["ambiguous"]["date_order"] is True
    with pytest.raises(SystemExit, match="day/month or month/day"):
        canonicalize(tmp_path, {"columns": {"date": "day", "date_order": "ambiguous", "series": [], "lag_days": 1},
                                "quantity_columns": {"units": "target"}, "frequency": "daily"})

