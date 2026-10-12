"""Candidate models, a frozen selection rule, and a single holdout evaluation (brief A3/A4, Phase 2).

Candidates (each skipped where unsuitable, and the skip is recorded):
  naive                 last observed value
  seasonal_naive_52     same week last year; only series with >= 104 weeks of history at the origin
  ets                   AutoETS, non-seasonal (a 52-period seasonal ETS is not appropriate for weekly data of
                        this length); StatsForecast's own intervals
  lgbm_lags             LightGBM on lags 1-4 (+ lag 52 where every series has >= 104 weeks), a 4-week rolling
                        mean and week-of-year; fixed, documented hyper-parameters (no tuning, so no inner
                        search to leak); conformal intervals built from the training data only
Protocol:
  1. Rolling-origin evaluation on data before the holdout, same data lag as the baselines (gap weeks).
  2. The selection RULE is written and hashed before any holdout forecast exists.
  3. The rule picks on the like-for-like population (every model forecast the same series, origins, weeks).
  4. The holdout (last `holdout_weeks`) is forecast once, for the selected model and the baselines only.
  Promotion into use is not done here: a person approves it in the engine.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import LEVEL, _load, _scale, check_gap, enough_history
from .calendar import WEEKLY, Calendar, horizon, of

LGBM_PARAMS = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10,
               "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 0, "n_jobs": 1, "verbose": -1}
INTERVAL_METHOD = "pooled_w10"        # chosen by the frozen interval study (forecast_worker/interval_study.py)
INTERVALS = {"lgbm_lags": {
    "method": "pooled_w10", "status": "not_validated",
    "summary": "Two frozen interval studies: for a nominal 80%, coverage was 80-84% on four weekly sets (three NordWerk "
               "realisations, UCI retail) but 58.0% on UCI Bike Sharing daily and 88.0% on four real daily German "
               "grid-load series (SMARD). The frozen rule needed 75-85% everywhere, so these ranges are NOT validated: "
               "read them as a rough indication. Nothing in planning uses them.",
    "evidence": "docs/forecast/evidence/intervals/test.json, docs/forecast/evidence/intervals-2/test.json"},
    "baselines": {"method": "StatsForecast model intervals", "status": "not_validated",
                  "summary": "The baselines' own intervals over-cover (91-99% in evaluation) and are wide."}}
RULE = {"metric": "mase", "population": "like_for_like",
        "select": "lowest MASE among candidates that beat seasonal_naive_52 (or naive where it cannot run) by >= 5%",
        "otherwise": "keep the best baseline", "min_improvement": 0.05, "holdout": "evaluated once, after the rule"}


def _stats(train: pd.DataFrame, h: int, ids: list[str], name: str, model, cal: Calendar = WEEKLY) -> pd.DataFrame:
    from statsforecast import StatsForecast
    sf = StatsForecast(models=[model], freq=cal.freq, n_jobs=1)
    fc = sf.forecast(h=h, df=train[train["unique_id"].isin(ids)][["unique_id", "ds", "y"]], level=[LEVEL])
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    col = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
    return pd.DataFrame({"model": name, "unique_id": fc["unique_id"], "ds": fc["ds"], "yhat": fc[col],
                         "lo": fc[f"{col}-lo-{LEVEL}"], "hi": fc[f"{col}-hi-{LEVEL}"]})


def _mlf(long: bool, cal: Calendar):
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    lags = list(cal.lags) + ([cal.long_lag] if long and cal.long_lag else [])
    return MLForecast(models={"lgbm": lgb.LGBMRegressor(**LGBM_PARAMS)}, freq=cal.freq, lags=lags,
                      lag_transforms={1: [RollingMean(window_size=cal.roll)]}, date_features=list(cal.date_features),
                      num_threads=1)


def _lgbm(train: pd.DataFrame, h: int, ids: list[str], long: bool, cal: Calendar = WEEKLY,
          interval: str = "conformal_w3") -> pd.DataFrame:
    from mlforecast.utils import PredictionIntervals
    df = train[train["unique_id"].isin(ids)][["unique_id", "ds", "y"]]
    mf = _mlf(long, cal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if interval == "conformal_w3":
            mf.fit(df, prediction_intervals=PredictionIntervals(n_windows=3, h=h))
            fc = mf.predict(h, level=[LEVEL])
            lo, hi = fc[f"lgbm-lo-{LEVEL}"], fc[f"lgbm-hi-{LEVEL}"]
        else:
            from .intervals import pooled_bounds
            mf.fit(df)
            fc = mf.predict(h)
            lo, hi = pooled_bounds(df, fc, lambda: _mlf(long, cal), h, cal, interval, LEVEL)
    return pd.DataFrame({"model": "lgbm_lags", "unique_id": fc["unique_id"], "ds": fc["ds"], "yhat": fc["lgbm"],
                         "lo": lo, "hi": hi})


def _forecast_at(panel: pd.DataFrame, origin, h: int, gap: int, models: set[str], cal: Calendar = WEEKLY,
                 interval: str = "conformal_w3") -> tuple[pd.DataFrame, list[dict]]:
    from statsforecast.models import AutoETS, Naive, SeasonalNaive
    train = panel[panel["ds"] <= origin]
    lengths = train.groupby("unique_id").size()
    ids = sorted(lengths[lengths >= cal.min_any].index)
    seas = sorted(lengths[lengths >= cal.min_seasonal].index)
    ml_ids = sorted(lengths[lengths >= cal.min_ml].index)
    out, skipped = [], []
    steps = h + gap
    sm = cal.seasonal_model
    if "naive" in models:
        out.append(_stats(train, steps, ids, "naive", Naive(), cal))
    if sm in models and seas:
        out.append(_stats(train, steps, seas, sm, SeasonalNaive(season_length=cal.season), cal))
    if "ets" in models:
        out.append(_stats(train, steps, ids, "ets", AutoETS(season_length=1), cal))
    if "lgbm_lags" in models and ml_ids:
        long = bool(cal.long_lag) and len(ml_ids) > 0 and all(lengths[i] >= cal.long_lag * 2 + cal.min_ml for i in ml_ids)
        out.append(_lgbm(train, steps, ml_ids, long, cal, interval))
    for sid in ids:
        if sid not in seas and sm in models:
            skipped.append({"unique_id": sid, "origin": str(origin.date()), "model": sm,
                            "reason": f"fewer than {cal.min_seasonal} {cal.unit}s of history at this origin"})
        if sid not in ml_ids and "lgbm_lags" in models:
            skipped.append({"unique_id": sid, "origin": str(origin.date()), "model": "lgbm_lags",
                            "reason": f"fewer than {cal.min_ml} {cal.unit}s of history at this origin"})
    fc = pd.concat(out, ignore_index=True)
    fc["step"] = fc.groupby(["model", "unique_id"]).cumcount() + 1
    fc = fc[fc["step"] > gap].copy()
    fc["horizon"] = fc["step"] - gap
    fc["origin"] = origin
    return fc.drop(columns="step"), skipped


def _metrics(df: pd.DataFrame, scales: dict) -> dict:
    if df.empty:
        return {"n": 0, "mae": None, "bias": None, "mase": None, "coverage": None, "mean_width": None}
    err = df["yhat"] - df["y"]
    sc = df["unique_id"].map(scales)
    mase = (err.abs() / sc).where(sc.notna())
    return {"n": int(len(df)), "mae": round(float(err.abs().mean()), 4), "bias": round(float(err.mean()), 4),
            "mase": round(float(mase.mean()), 4) if mase.notna().any() else None,
            "coverage": round(float(((df["y"] >= df["lo"]) & (df["y"] <= df["hi"])).mean()), 4),
            "mean_width": round(float((df["hi"] - df["lo"]).mean()), 4)}


def _like_for_like(scored: pd.DataFrame, scales: dict) -> tuple[dict, dict]:
    keys = ["unique_id", "origin", "ds"]
    sets = [set(map(tuple, g[keys].astype(str).to_numpy())) for _, g in scored.groupby("model")]
    common = set.intersection(*sets) if sets else set()
    mask = scored[keys].astype(str).apply(tuple, axis=1).isin(common)
    pop = {"points": len(common), "series": sorted({c[0] for c in common})}
    return pop, {m: _metrics(g, scales) for m, g in scored[mask].groupby("model")}


def select(comparable: dict) -> dict:
    """The frozen rule, applied to cross-validation results only."""
    seasonal = next((k for k in comparable if k.startswith("seasonal_naive_")), None)
    base = seasonal or "naive"
    base_mase = comparable[base]["mase"]
    baselines = {k: v for k, v in comparable.items() if (k == "naive" or k.startswith("seasonal_naive_")) and v["mase"] is not None}
    best_base = min(baselines, key=lambda k: baselines[k]["mase"])
    winners = {k: v for k, v in comparable.items() if k not in baselines and v["mase"] is not None and base_mase
               and v["mase"] <= base_mase * (1 - RULE["min_improvement"])}
    if winners:
        chosen = min(winners, key=lambda k: winners[k]["mase"])
        return {"selected": chosen, "reason": f"MASE {winners[chosen]['mase']} vs {base} {base_mase} "
                                              f"({(1 - winners[chosen]['mase'] / base_mase) * 100:.1f}% better)"}
    return {"selected": best_base, "reason": f"no candidate beat {base} (MASE {base_mase}) by "
                                             f"{RULE['min_improvement'] * 100:.0f}%: the baseline stays"}


def evaluate(data: Path, out: Path, params: dict) -> dict:
    cal = of(params)
    panel, info = _load(data, params.get("answers") or {}, cal)
    h, gap, holdout, n_origins = horizon(params, cal)
    models = set(params.get("models") or ["naive", cal.seasonal_model, "ets", "lgbm_lags"])
    interval = params.get("interval_method") or INTERVAL_METHOD
    last = panel["ds"].max()
    cut_end = last - cal.periods(holdout)
    origins = [cut_end - cal.periods(h + gap + k) for k in range(n_origins)][::-1]
    # a gap below the data lag measured on rows up to the first origin is refused
    enough_history(panel, origins[0], cal, h=h, gap=gap, holdout=holdout, n_origins=n_origins)
    info = {**info, **check_gap(panel, gap, cal, before=origins[0]), "frequency": cal.name, "interval_method": interval}
    scales = {sid: _scale(g[g["ds"] <= cut_end]["y"].to_numpy()) for sid, g in panel.groupby("unique_id")}
    cv, skipped = [], []
    for origin in origins:
        fc, sk = _forecast_at(panel, origin, h, gap, models, cal, interval)
        cv.append(fc)
        skipped += sk
    cv = pd.concat(cv).merge(panel, on=["unique_id", "ds"], how="left")
    cv = cv[cv["y"].notna() & ~cv["imputed"]]
    normal = cv[~cv["stockout"]]
    pop, comparable = _like_for_like(normal, scales)
    # ---- freeze the rule and the choice BEFORE any holdout forecast exists ----
    decision = select(comparable)
    frozen = {"rule": RULE, "cv_comparable": comparable, "decision": decision, "lgbm_params": LGBM_PARAMS}
    frozen_sha = hashlib.sha256(json.dumps(frozen, sort_keys=True, default=str).encode()).hexdigest()
    (out / "selection_frozen.json").write_text(json.dumps({**frozen, "sha256": frozen_sha}, indent=1, default=str) + "\n")
    # ---- holdout: once, for the selected model and the baselines ----
    ho_models = {decision["selected"], "naive", cal.seasonal_model} & models
    ho_origins = [cut_end - cal.periods(gap) + cal.periods(h * k) for k in range(holdout // h)]
    ho = pd.concat([_forecast_at(panel, o, h, gap, ho_models, cal, interval)[0] for o in ho_origins]).merge(panel, on=["unique_id", "ds"], how="left")
    ho = ho[ho["y"].notna() & ~ho["imputed"] & (ho["ds"] > cut_end)]
    ho_pop, ho_comp = _like_for_like(ho[~ho["stockout"]], scales)
    report = {
        "population": {"origins": [str(o.date()) for o in origins], "holdout_from": str((cut_end + cal.step).date()),
                       "holdout_origins": [str(o.date()) for o in ho_origins], "series": sorted(panel["unique_id"].unique()),
                       "horizon_weeks": h, "gap_weeks": gap, "interval_level": LEVEL, **info},
        "cross_validation": {"comparable_population": pop, "comparable": comparable,
                             "per_model": {m: {"overall": _metrics(g, scales),
                                               "by_horizon": {int(k): _metrics(v, scales) for k, v in g.groupby("horizon")},
                                               "by_series": {k: _metrics(v, scales) for k, v in g.groupby("unique_id")},
                                               "stockout_weeks": _metrics(cv[(cv["model"] == m) & cv["stockout"]], scales)}
                                           for m, g in normal.groupby("model")}},
        "selection": {**decision, "frozen_sha256": frozen_sha, "rule": RULE},
        "intervals": INTERVALS,
        "holdout": {"population": ho_pop, "comparable": ho_comp},
        "skipped": skipped,
        "note": "Intervals: statistical models' own; LightGBM conformal from training windows. Negative interval "
                "bounds are reported as produced (not clipped). Promotion requires a person's approval."}
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    fc_out = pd.concat([cv.assign(split="cv"), ho.assign(split="holdout")])
    fc_out = fc_out[["split", "model", "origin", "unique_id", "ds", "horizon", "yhat", "lo", "hi", "y", "imputed", "stockout"]]
    fc_out["origin"], fc_out["ds"] = pd.to_datetime(fc_out["origin"]).dt.date.astype(str), pd.to_datetime(fc_out["ds"]).dt.date.astype(str)
    fc_out.sort_values(["split", "model", "unique_id", "origin", "ds"]).to_csv(out / "evaluation_forecasts.csv", index=False, float_format="%.6f")
    return {"outputs": ["evaluation_report.json", "selection_frozen.json", "evaluation_forecasts.csv"],
            "summary": {"selected": decision["selected"], "reason": decision["reason"], "frozen_sha256": frozen_sha,
                        "cv_comparable": comparable, "holdout_comparable": ho_comp, "origins": len(origins)}}

