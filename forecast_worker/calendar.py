"""The time grid a forecast runs on: one definition used by profiling, evaluation, issuing and planning.

Weekly is the original grid (ISO weeks, Monday start) and keeps its exact settings, so weekly results are
unchanged by this module. Daily was added for the public Bike Sharing benchmark: a 7-day season, recent-day lags,
and a day-of-week feature. Settings are fixed here, not tuned per dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Calendar:
    name: str                  # "weekly" | "daily"
    freq: str                  # pandas / Nixtla frequency
    step: pd.Timedelta         # one period
    season: int                # seasonal period of the seasonal-naive baseline
    min_seasonal: int          # periods of history needed before the seasonal baseline runs (two full seasons)
    lags: tuple[int, ...]      # LightGBM lags
    long_lag: int | None       # an extra seasonal lag when every series is long enough
    roll: int                  # rolling-mean window on lag 1
    date_features: tuple[str, ...]
    min_ml: int                # periods of history before LightGBM runs for a series
    min_any: int               # periods of history before any model runs for a series
    unit: str                  # "week" | "day", for messages

    @property
    def seasonal_model(self) -> str:
        return f"seasonal_naive_{self.season}"

    def periods(self, n: int) -> pd.Timedelta:
        return self.step * n


WEEKLY = Calendar("weekly", "W-MON", pd.Timedelta(weeks=1), 52, 104, (1, 2, 3, 4), 52, 4, ("week",), 30, 8, "week")
DAILY = Calendar("daily", "D", pd.Timedelta(days=1), 7, 14, (1, 2, 3, 7, 14), None, 7, ("dayofweek",), 60, 14, "day")
CALENDARS = {c.name: c for c in (WEEKLY, DAILY)}


def of(params: dict) -> Calendar:
    name = (params or {}).get("frequency", "weekly")
    if name not in CALENDARS:
        raise SystemExit(f"unsupported frequency {name!r}: supported are {', '.join(CALENDARS)}")
    return CALENDARS[name]


def horizon(params: dict, cal: Calendar) -> tuple[int, int, int, int]:
    """(horizon, gap, holdout, origins) in periods. Weekly accepts its original *_weeks names."""
    p = params or {}
    def get(generic: str, weekly: str, default: int) -> int:
        return int(p.get(generic, p.get(weekly, default) if cal.name == "weekly" else default))
    d = {"weekly": (4, 1, 8, 12), "daily": (7, 1, 28, 12)}[cal.name]
    return (get("horizon", "horizon_weeks", d[0]), get("gap", "gap_weeks", d[1]),
            get("holdout", "holdout_weeks", d[2]), int(p.get("origins", d[3])))

