"""Interval methods for the LightGBM model (the models from StatsForecast keep their own).

conformal_w3   (the original) MLForecast's conformal intervals from 3 calibration windows PER SERIES. With 3 scores,
               an 80% quantile is effectively the 2nd-3rd largest of three: reproduced 2026-09-28 at 74.5% CV and
               64.2% holdout coverage on NordWerk. Kept only to reproduce that result.
pooled_w10     scale-free split conformal: refit the same model at 10 rolling calibration origins INSIDE the
               training data, take |error| / series scale per step ahead, POOL across series, and use the
               finite-sample conformal quantile ceil((n+1)(level))/n. Interval = forecast +- q_step * series scale.
pooled_w20     the same with 20 calibration origins.
spread_y1      the same, but the calibration origins are spread over the last year of training data instead of its
               last few periods: weekly 26 origins two weeks apart, daily 52 origins a week apart (interval study 2,
               2026-09-28: pooled_w10's 10 consecutive daily origins gave one series only 10 scores per step, all
               from the same fortnight).
The scale is the series' in-sample one-step naive MAE on the training data only (as MASE). Nothing after the
training data's last period is used.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

WINDOWS = {"pooled_w10": 10, "pooled_w20": 20}
SPREAD = {"spread_y1": {"weekly": (26, 2), "daily": (52, 7)}}      # (origins, periods between origins)


def windows_of(method: str, cal) -> tuple[int, int]:
    return SPREAD[method][cal.name] if method in SPREAD else (WINDOWS[method], 1)


def _scales(df: pd.DataFrame) -> pd.Series:
    def one(g):
        d = np.abs(np.diff(g.sort_values("ds")["y"].to_numpy()))
        return float(d.mean()) if len(d) and d.mean() > 0 else np.nan
    return df.groupby("unique_id").apply(one, include_groups=False)


def calibration_quantiles(df: pd.DataFrame, make, h: int, windows: int, level: int, step_size: int = 1) -> tuple[dict[int, float], dict]:
    """q[step] from rolling refits inside df (the training data); also the population it came from."""
    mf = make()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = mf.cross_validation(df=df, n_windows=windows, h=h, step_size=step_size, refit=True)
    cv = cv.rename(columns={"lgbm": "yhat"})
    cv["step"] = cv.groupby(["unique_id", "cutoff"]).cumcount() + 1
    scale = _scales(df)
    cv["score"] = (cv["y"] - cv["yhat"]).abs() / cv["unique_id"].map(scale)
    cv = cv.dropna(subset=["score"])
    alpha = level / 100
    q, pop = {}, {}
    for step, g in cv.groupby("step"):
        s = np.sort(g["score"].to_numpy())
        n = len(s)
        k = math.ceil((n + 1) * alpha)
        q[int(step)] = float(s[min(k, n) - 1]) if n else float("nan")
        pop[int(step)] = n
    return q, {"scores_per_step": pop, "windows": windows, "step_size": step_size}


def pooled_bounds(df: pd.DataFrame, fc: pd.DataFrame, make, h: int, cal, method: str, level: int):
    windows, step_size = windows_of(method, cal)
    q, _ = calibration_quantiles(df, make, h, windows, level, step_size)
    scale = _scales(df)
    step = fc.groupby("unique_id").cumcount() + 1
    half = step.map(q) * fc["unique_id"].map(scale)
    return fc["lgbm"] - half, fc["lgbm"] + half

