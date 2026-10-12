"""Baseline models and rolling-origin evaluation. See docs/limitations.md for known preparation issues.

Decisions, each recorded in the output manifest:
- Target: observed weekly unit sales (not unconstrained demand). The quantity column mapping and the meaning
  of missing weeks come from the owner's answers (params.answers); without them the job refuses to run.
- Availability: a week's row becomes available `available_at` (7-9 days later in the NordWerk fixture). A
  forecast issued on the Monday of week t+2 knows data through week t only, so its four targets are steps
  2..5 from the last known week (params.gap_weeks, default 1).
- Missing weeks: imputed ONLY as model input (linear within the series, flagged); an imputed week is never an
  actual that gets scored. Stockout weeks (usable stock 0) are scored separately, not mixed into the main score.
- Holdout: the last `holdout_weeks` (default 8) are left untouched here for model selection later.
- Models: Naive (last value) always; SeasonalNaive(52) only for series with >= 104 observed weeks, otherwise
  skipped and said so. Intervals: the models' own 80% intervals from StatsForecast; coverage is measured.
- Metrics per series and horizon step: MAE, bias (mean of forecast minus actual; positive = over-forecast),
  MASE scaled by the series' in-sample one-step naive MAE. An undefined value (zero scale, no scored weeks)
  is reported as null, never as 0.
"""
from __future__ import annotations

import math

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import WEEKLY, Calendar, horizon, of

LEVEL = 80


def enough_history(panel: pd.DataFrame, first_origin, cal, *, h: int, gap: int, holdout: int, n_origins: int) -> None:
    """Insufficient history is an honest outcome, stated with its numbers: never a misleading error further on (with
    too little data the first origin falls before the data starts, and the lag check then finds no rows)."""
    sizes = panel[panel["ds"] <= first_origin].groupby("unique_id").size()
    have = int(sizes.max()) if len(sizes) else 0
    if have >= cal.min_any:
        return
    longest = int(panel.groupby("unique_id").size().max()) if len(panel) else 0
    need = cal.min_any + holdout + h + gap + n_origins - 1
    raise SystemExit(f"insufficient history: the longest series has {longest} {cal.unit}s; this evaluation needs at least {need} "
                     f"({cal.min_any} to learn from before the first origin, {n_origins} origins, a {h}-{cal.unit} horizon after "
                     f"a {gap}-{cal.unit} lag, {holdout} held out). Fewer origins or a shorter holdout lower the bar; "
                     f"a forecast cannot be judged on less.")


def required_gap(panel: pd.DataFrame, cal: Calendar = WEEKLY, before=None) -> tuple[int, float]:
    """The data lag, MEASURED from the rows themselves, as (periods, max days).

    A forecast at origin o (the last period used) can only be issued once period o's row has arrived, at its
    available_at. Every period that has already started by then is the lag: it is not scored as a forecast.
    periods = ceil(max(available_at - end of its period) / one period), at least 0. Only rows up to `before`
    (the first evaluation origin) are measured: the lag used to set up an evaluation must itself be knowable at
    that time, never read from later rows. A caller's gap below this is refused, never trusted."""
    known = panel.dropna(subset=["available_at"])
    if before is not None:
        known = known[known["ds"] <= before]
    if known.empty:
        raise SystemExit("no row says when it became available (available_at): the data lag cannot be measured")
    period_end = pd.to_datetime(known["ds"]).dt.tz_localize("UTC") + cal.step
    days = float(((known["available_at"] - period_end).dt.total_seconds() / 86400).max())
    return max(0, math.ceil(days / (cal.step.total_seconds() / 86400))), round(days, 2)


def check_gap(panel: pd.DataFrame, gap: int, cal: Calendar = WEEKLY, before=None) -> dict:
    need, days = required_gap(panel, cal, before)
    if gap < need:
        raise SystemExit(f"{'gap_weeks' if cal.name == 'weekly' else 'gap'}={gap} is shorter than the data's own lag: rows arrive up to {days} days after "
                         f"their {cal.unit} ends, so at least {need} {cal.unit}(s) after each origin cannot be scored")
    return {"measured_lag_days": days, "measured_gap_weeks" if cal.name == "weekly" else "measured_gap": need}


def _load(data: Path, answers: dict, cal: Calendar = WEEKLY) -> tuple[pd.DataFrame, dict]:
    qmap = answers.get("quantity_columns")            # e.g. {"qty": "net_units", "quantity": "net_units"}
    missing_policy = answers.get("missing_weeks")     # "missing" or "zero"
    if not qmap or missing_policy not in {"missing", "zero"}:
        raise SystemExit("baseline needs the owner's answers: quantity_columns and missing_weeks")
    frames = []
    for p in sorted(data.glob("sales_*.csv")):
        df = pd.read_csv(p, dtype=str)
        col = next((c for c in df.columns if c in qmap), None)
        if col is None:
            raise SystemExit(f"{p.name}: no quantity column the owner mapped")
        date_col = "week_start" if "week_start" in df.columns else "period_start"
        if date_col not in df.columns:
            raise SystemExit(f"{p.name}: no week_start or period_start column")
        frames.append(pd.DataFrame({"unique_id": df["series_id"], "ds": pd.to_datetime(df[date_col]),
                                    "y": pd.to_numeric(df[col]), "available_at": pd.to_datetime(df["available_at"], utc=True)}))
    sales = pd.concat(frames).sort_values(["unique_id", "ds"]).drop_duplicates(["unique_id", "ds"], keep="last")
    stock = pd.read_csv(data / "inventory.csv", dtype=str) if (data / "inventory.csv").exists() else None
    stockout = set()
    if stock is not None:
        snap = pd.to_datetime(stock["snapshot_time"].str[:10])
        monday = snap - pd.to_timedelta(snap.dt.weekday, unit="D")
        for sid, wk, q in zip(stock["series_id"], monday, pd.to_numeric(stock["usable_on_hand"])):
            if q == 0:
                stockout.add((sid, wk))
    rows = []
    for sid, g in sales.groupby("unique_id"):
        full = pd.date_range(g["ds"].min(), g["ds"].max(), freq=cal.freq)
        s = g.set_index("ds").reindex(full)
        imputed = s["y"].isna()
        if missing_policy == "zero":
            s["y"] = s["y"].fillna(0)
            imputed[:] = False
        else:
            s["y"] = s["y"].interpolate(limit_direction="both")
        for ds, y, imp, avail in zip(full, s["y"], imputed, s["available_at"]):
            rows.append({"unique_id": sid, "ds": ds, "y": float(y), "imputed": bool(imp),
                         "stockout": (sid, ds) in stockout, "available_at": avail})
    panel = pd.DataFrame(rows)
    info = {"missing_policy": missing_policy, "quantity_columns": qmap,
            "imputed_weeks": int(panel["imputed"].sum()), "stockout_weeks": int(panel["stockout"].sum())}
    return panel, info


def _scale(y: np.ndarray) -> float | None:
    d = np.abs(np.diff(y))
    return float(d.mean()) if len(d) and d.mean() > 0 else None


def baseline(data: Path, out: Path, params: dict) -> dict:
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, SeasonalNaive
    cal = of(params)
    panel, info = _load(data, params.get("answers") or {}, cal)
    h, gap, holdout, n_origins = horizon(params, cal)
    last = panel["ds"].max()
    cut_end = last - cal.periods(holdout)                            # evaluation never sees the holdout
    origins = [cut_end - cal.periods(h + gap + k) for k in range(n_origins)][::-1]
    enough_history(panel, origins[0], cal, h=h, gap=gap, holdout=holdout, n_origins=n_origins)
    info = {**info, **check_gap(panel, gap, cal, before=origins[0]), "frequency": cal.name}
    long_enough = {sid for sid, g in panel.groupby("unique_id") if (~g["imputed"]).sum() >= cal.min_seasonal}
    records, skipped = [], []
    for origin in origins:
        train = panel[panel["ds"] <= origin]
        train = train.groupby("unique_id").filter(lambda g: len(g) >= cal.min_any)
        fits = [("naive", [Naive()], sorted(train["unique_id"].unique()))]
        seas_ids = sorted(set(train["unique_id"]) & long_enough &
                          {sid for sid, g in train.groupby("unique_id") if len(g) >= cal.min_seasonal})
        if seas_ids:
            fits.append((cal.seasonal_model, [SeasonalNaive(season_length=cal.season)], seas_ids))
        for name, models, ids in fits:
            sf = StatsForecast(models=models, freq=cal.freq, n_jobs=1)
            fc = sf.forecast(h=h + gap, df=train[train["unique_id"].isin(ids)][["unique_id", "ds", "y"]], level=[LEVEL])
            fc = fc.reset_index() if "unique_id" not in fc.columns else fc
            col = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
            fc["step"] = fc.groupby("unique_id").cumcount() + 1
            fc = fc[fc["step"] > gap]                                  # steps inside the data lag are not a forecast
            for d in fc.to_dict("records"):
                records.append({"model": name, "origin": origin, "unique_id": d["unique_id"], "ds": d["ds"],
                                "horizon": int(d["step"] - gap), "yhat": float(d[col]),
                                "lo": float(d[f"{col}-lo-{LEVEL}"]), "hi": float(d[f"{col}-hi-{LEVEL}"])})
        for sid in sorted(set(train["unique_id"]) - set(seas_ids)):
            skipped.append({"unique_id": sid, "origin": str(origin.date()), "model": cal.seasonal_model,
                            "reason": f"fewer than {cal.min_seasonal} observed {cal.unit}s" if sid not in long_enough
                            else f"fewer than {cal.min_seasonal} {cal.unit}s of history at this origin"})
    fc = pd.DataFrame(records).merge(panel, on=["unique_id", "ds"], how="left")
    fc = fc[fc["y"].notna()]
    scored = fc[~fc["imputed"]]
    scales = {sid: _scale(g[g["ds"] <= cut_end]["y"].to_numpy()) for sid, g in panel.groupby("unique_id")}

    def metrics(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0, "mae": None, "bias": None, "mase": None, "coverage": None, "mean_width": None}
        err = df["yhat"] - df["y"]
        sc = df["unique_id"].map(scales)
        mase = (err.abs() / sc).where(sc.notna())
        return {"n": int(len(df)), "mae": round(float(err.abs().mean()), 4), "bias": round(float(err.mean()), 4),
                "mase": round(float(mase.mean()), 4) if mase.notna().any() else None,
                "coverage": round(float(((df["y"] >= df["lo"]) & (df["y"] <= df["hi"])).mean()), 4),
                "mean_width": round(float((df["hi"] - df["lo"]).mean()), 4)}

    report = {"population": {"origins": [str(o.date()) for o in origins], "holdout_from": str((cut_end + cal.step).date()),
                             "series": sorted(panel["unique_id"].unique().tolist()), "horizon_weeks": h, "gap_weeks": gap,
                             "interval_level": LEVEL, **info},
              "models": {}, "skipped": skipped,
              "note": "Scored on observed, non-imputed weeks; stockout weeks reported separately. Holdout untouched."}
    for model, g in scored.groupby("model"):
        normal, out_of_stock = g[~g["stockout"]], g[g["stockout"]]
        report["models"][model] = {
            "overall": metrics(normal), "stockout_weeks": metrics(out_of_stock),
            "by_horizon": {int(k): metrics(v) for k, v in normal.groupby("horizon")},
            "by_series": {k: metrics(v) for k, v in normal.groupby("unique_id")}}
    # like for like: only the (series, origin, week) every model forecast, so a model that skips hard series
    # cannot look better by being scored on easier ones
    keys = ["unique_id", "origin", "ds"]
    normal_all = scored[~scored["stockout"]]
    common = None
    for _, g in normal_all.groupby("model"):
        k = set(map(tuple, g[keys].astype(str).to_numpy()))
        common = k if common is None else common & k
    if common:
        mask = normal_all[keys].astype(str).apply(tuple, axis=1).isin(common)
        report["comparable"] = {"population": {"points": len(common),
                                               "series": sorted({c[0] for c in common})},
                                "models": {m: metrics(g) for m, g in normal_all[mask].groupby("model")}}
    fc_out = fc[["model", "origin", "unique_id", "ds", "horizon", "yhat", "lo", "hi", "y", "imputed", "stockout"]].copy()
    fc_out["origin"], fc_out["ds"] = fc_out["origin"].dt.date.astype(str), fc_out["ds"].dt.date.astype(str)
    fc_out.to_csv(out / "backtest_forecasts.csv", index=False)
    (out / "baseline_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    digest = hashlib.sha256((out / "backtest_forecasts.csv").read_bytes()).hexdigest()
    summary = {m: v["overall"] for m, v in report["models"].items()}
    comparable = (report.get("comparable") or {}).get("models") or {}
    return {"outputs": ["baseline_report.json", "backtest_forecasts.csv"],
            "summary": {"models": summary, "comparable": comparable, "origins": len(origins), "forecasts_sha256": digest,
                        "skipped_seasonal": len({s["unique_id"] for s in skipped})}}

