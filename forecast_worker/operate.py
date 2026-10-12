"""Workflow B in the worker: issue a forecast with the APPROVED model, then compute plans (brief §8).

issue  params {model, answers, horizon_weeks, gap_weeks}: fit the approved model on every week available at
       issue time; forecast the next `horizon_weeks` after the data lag; write issued_forecast.csv with the issue
       time, target week, model, horizon, point and interval, and the input set hash. An issued forecast is
       immutable: later actuals are compared against it, never the reverse.
plan   inputs: issued_forecast.csv + inventory.csv + receipts.csv + rules.csv. For each series: the latest stock
       snapshot available at issue time, future receipts, the versioned rule; planning.plan() does the arithmetic.
       The requirement per week is the point forecast, floored at 0 (a negative forecast is no demand), stated in
       the output. Exceptions are the series with a positive proposed quantity.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from .baseline import _load
from .calendar import horizon, of
from .evaluate import INTERVAL_METHOD, INTERVALS, _forecast_at
from .planning import Receipt, Rule, plan

MODELS = {"naive", "seasonal_naive_52", "seasonal_naive_7", "ets", "lgbm_lags"}


def issue(data: Path, out: Path, params: dict) -> dict:
    model = params.get("model")
    if model not in MODELS:
        raise SystemExit("issue needs the approved model (a promotion) - none was given")
    cal = of(params)
    panel, info = _load(data, params.get("answers") or {}, cal)
    h, gap, _, _ = horizon(params, cal)
    issued_at = params.get("issued_at") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    # only what was known at issue time, PER SERIES: a week's row counts once its own available_at has passed
    known = panel[panel["available_at"].notna() & (panel["available_at"] <= pd.Timestamp(issued_at))]
    if known.empty:
        raise SystemExit("no sales data was available at the issue time")
    last_known = known.groupby("unique_id")["ds"].max()
    panel = panel[panel["ds"] <= panel["unique_id"].map(last_known)]
    origin = last_known.max()
    # every week after the last known one: the data-lag weeks (horizon 0) are not scored as forecasts, but
    # planning needs them, because their sales consume stock before the first scored week
    targets = [origin + cal.periods(k) for k in range(1, gap + h + 1)]
    lag = int((origin - last_known.min()) / cal.step)
    fc, skipped = _forecast_at(panel, origin, h + gap + lag, 0, {model}, cal,
                               params.get("interval_method") or INTERVAL_METHOD)
    fc = fc[fc["ds"].isin(targets)].copy()
    fc["horizon"] = fc["ds"].map({t: max(0, i + 1 - gap) for i, t in enumerate(targets)})
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.csv"))}
    fc = fc.sort_values(["unique_id", "ds"])
    rows = [{"issued_at": issued_at, "model": model, "origin_week": str(origin.date()), "series_id": r["unique_id"],
             "target_week": str(pd.Timestamp(r["ds"]).date()), "horizon": int(r["horizon"]),
             "yhat": f"{r['yhat']:.6f}", "lo": f"{r['lo']:.6f}", "hi": f"{r['hi']:.6f}"} for r in fc.to_dict("records")]
    with (out / "issued_forecast.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest = {"issued_at": issued_at, "replay": bool(params.get("replay")), "model": model, "origin_week": str(origin.date()), "horizon_weeks": h,
                "frequency": cal.name,
                "intervals": INTERVALS["lgbm_lags"] if model == "lgbm_lags" else INTERVALS["baselines"],
                "rule": "each series fit on its own rows whose available_at <= issued_at; common target weeks",
                "last_known_week": {k: str(v.date()) for k, v in last_known.items()},
                "gap_weeks": gap, "inputs": inputs, "series": sorted({r["series_id"] for r in rows}), "skipped": skipped,
                "forecast_sha256": hashlib.sha256((out / "issued_forecast.csv").read_bytes()).hexdigest(), **info}
    (out / "issue_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return {"outputs": ["issued_forecast.csv", "issue_manifest.json"],
            "summary": {"model": model, "issued_at": issued_at, "replay": bool(params.get("replay")), "origin_week": manifest["origin_week"],
                        "series": len(manifest["series"]), "forecast_sha256": manifest["forecast_sha256"]}}


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_plan(data: Path, out: Path, params: dict) -> dict:
    need = ["issued_forecast.csv", "inventory.csv", "receipts.csv", "rules.csv"]
    missing = [n for n in need if not (data / n).exists()]
    if missing:
        raise SystemExit(f"plan needs {', '.join(missing)}")
    fc = _csv(data / "issued_forecast.csv")
    issued_at = datetime.fromisoformat(fc[0]["issued_at"])
    # a stock count older than this at issue time is not planned on: the sales since then are not in it, so stock
    # would look too high and the plan would order too little. The series is held with the snapshot's age.
    max_age = timedelta(days=int(params.get("max_snapshot_age_days") or 7))
    stock: dict[str, dict] = {}
    for r in _csv(data / "inventory.csv"):
        avail = datetime.fromisoformat(r["available_at"])
        if avail <= issued_at and (r["series_id"] not in stock or r["snapshot_time"] > stock[r["series_id"]]["snapshot_time"]):
            stock[r["series_id"]] = r
    stale = {sid: issued_at - datetime.fromisoformat(r["snapshot_time"]) for sid, r in stock.items()
             if issued_at - datetime.fromisoformat(r["snapshot_time"]) > max_age}
    receipts: dict[str, list[Receipt]] = {}
    for r in _csv(data / "receipts.csv"):
        receipts.setdefault(r["series_id"], []).append(Receipt(
            r["receipt_id"], Decimal(r["quantity"]), date.fromisoformat(r["expected_date"]) if r.get("expected_date") else None,
            r.get("status") == "confirmed"))
    rules = {}
    for r in _csv(data / "rules.csv"):
        if r["series_id"] not in rules or int(r["rule_version"]) > int(rules[r["series_id"]]["rule_version"]):
            rules[r["series_id"]] = r
    plans, held = [], []
    by_series: dict[str, list[dict]] = {}
    for r in fc:
        by_series.setdefault(r["series_id"], []).append(r)
    for sid, rows in sorted(by_series.items()):
        rows.sort(key=lambda r: r["target_week"])
        if sid in stock:
            # plan every forecast week after the stock snapshot's week: its sales come out of that stock
            snap = datetime.fromisoformat(stock[sid]["snapshot_time"]).date()
            snap_week = snap.toordinal() - snap.weekday()
            rows = [r for r in rows if date.fromisoformat(r["target_week"]).toordinal() > snap_week]
        if sid in stale:
            held.append({"series_id": sid, "why": f"the stock snapshot is {stale[sid].days} days old at issue time "
                                                  f"(limit {max_age.days}): refresh the stock count before planning"})
            continue
        if sid not in stock or sid not in rules:
            held.append({"series_id": sid, "why": "no stock snapshot available at issue time" if sid not in stock else "no rule"})
            continue
        rule = rules[sid]
        cost = rule.get("unit_cost") if rule.get("unit_cost") not in (None, "") else rule.get("unit_cost_eur")
        price = Decimal(cost) if cost not in (None, "") else None          # a purchase cost in rule["currency"]
        p = plan(sid, [date.fromisoformat(r["target_week"]) for r in rows],
                 [max(Decimal(0), Decimal(r["yhat"])) for r in rows],
                 usable_on_hand=Decimal(stock[sid]["usable_on_hand"]), commitments=Decimal(stock[sid].get("committed") or 0),
                 receipts=receipts.get(sid, []),
                 rule=Rule(Decimal(rule["buffer_units"]), Decimal(rule["pack_size"]), Decimal(rule["min_order_units"]),
                           price, rule.get("currency") or "EUR"))
        d = p.as_dict()
        d.update(stock_snapshot=stock[sid]["snapshot_time"], rule_version=int(rule["rule_version"]),
                 destination=rule.get("destination"), approval=rule.get("approval"))
        plans.append(d)
    exceptions = [p for p in plans if Decimal(p["proposed_quantity"]) > 0]
    result = {"issued_at": fc[0]["issued_at"], "model": fc[0]["model"], "max_snapshot_age_days": max_age.days,
              "policy": "requirement = point forecast floored at 0; test policy, not an optimal inventory rule",
              "plans": plans, "held": held, "exceptions": [p["series_id"] for p in exceptions]}
    (out / "plan.json").write_text(json.dumps(result, indent=1) + "\n")
    return {"outputs": ["plan.json"],
            "summary": {"series_planned": len(plans), "exceptions": len(exceptions), "held": len(held),
                        "proposed_units": str(sum((Decimal(p["proposed_quantity"]) for p in exceptions), Decimal(0))),
                        "plan_sha256": hashlib.sha256((out / "plan.json").read_bytes()).hexdigest()}}


def run_scenario(data: Path, out: Path, params: dict) -> dict:
    """A planning scenario: the same issued forecast, the same stock and rules, with named confirmed receipts moved
    later (params.receipt_shifts {receipt_id: weeks, 1..8}). The demand forecast is NOT changed: a supplier delay
    moves supply, it does not reduce demand. Writes plan.json (the scenario's plan) and scenario_diff.json (per
    series: base vs scenario shortfall, first short week, proposed quantity and value)."""
    import shutil
    import tempfile
    shifts = {str(k): int(v) for k, v in (params.get("receipt_shifts") or {}).items()}
    if not shifts or any(not 1 <= v <= 8 for v in shifts.values()):
        raise SystemExit("a scenario moves at least one receipt, by 1 to 8 weeks")
    base = json.loads((data / "plan.json").read_text())
    stage = Path(tempfile.mkdtemp(prefix="fw-scn-"))
    for f in data.iterdir():
        if f.name != "receipts.csv":
            shutil.copyfile(f, stage / f.name)
    receipts = _csv(data / "receipts.csv")
    moved, found = [], set()
    for r in receipts:
        if r["receipt_id"] in shifts:
            found.add(r["receipt_id"])
            if r.get("status") != "confirmed" or not r.get("expected_date"):
                raise SystemExit(f"{r['receipt_id']} is not a confirmed, dated receipt: moving it changes nothing")
            old = date.fromisoformat(r["expected_date"])
            r["expected_date"] = (old + timedelta(weeks=shifts[r["receipt_id"]])).isoformat()
            moved.append({"receipt_id": r["receipt_id"], "series_id": r["series_id"], "from": old.isoformat(),
                          "to": r["expected_date"], "quantity": r["quantity"]})
    if set(shifts) - found:
        raise SystemExit(f"unknown receipt(s): {', '.join(sorted(set(shifts) - found))}")
    with (stage / "receipts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(receipts[0]))
        w.writeheader()
        w.writerows(receipts)
    res = run_plan(stage, out, {**params, "max_snapshot_age_days": base.get("max_snapshot_age_days") or 7})   # the base plan's limit
    scen = json.loads((out / "plan.json").read_text())
    b, s = {p["series_id"]: p for p in base["plans"]}, {p["series_id"]: p for p in scen["plans"]}
    rows = []
    for sid in sorted(set(b) | set(s)):
        x, y = b.get(sid) or {}, s.get(sid) or {}
        keys = ("shortfall", "first_short_week", "proposed_quantity", "proposed_value")
        weeks = [{"week": t["week"], "base_available": t["available"], "scenario_available": u["available"],
                  "base_gap": t["gap"], "scenario_gap": u["gap"]}
                 for t, u in zip(x.get("trajectory", []), y.get("trajectory", [])) if t["available"] != u["available"]]
        if weeks or any(x.get(k) != y.get(k) for k in keys):   # timing matters even when the total does not change
            rows.append({"series_id": sid, **{f"base_{k}": x.get(k) for k in keys}, **{f"scenario_{k}": y.get(k) for k in keys},
                         "weeks_changed": [w["week"] for w in weeks], "availability": weeks,
                         "scenario_excluded_receipts": y.get("excluded_receipts")})
    diff = {"scenario": {"receipt_shifts": shifts, "moved": moved},
            "unchanged": "the issued demand forecast, stock snapshots, rules and prices",
            "base_plan_sha256": hashlib.sha256((data / "plan.json").read_bytes()).hexdigest(),
            "changed_series": rows, "label": "scenario: nothing here is sent or approved; a proposal from it is a new "
                                              "proposal that needs its own approval"}
    (out / "scenario_diff.json").write_text(json.dumps(diff, indent=1) + "\n")
    shutil.rmtree(stage, ignore_errors=True)
    return {"outputs": ["plan.json", "scenario_diff.json"],
            "summary": {**res["summary"], "changed_series": len(rows), "moved": [m["receipt_id"] for m in moved]}}

