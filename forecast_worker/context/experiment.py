"""Matched context-feature comparison (brief 7A). Every arm uses the SAME learner, parameters, lags, series,
origins, target weeks and holdout; only the feature columns differ.

  numeric          LightGBM on lags only (the numerical-only model)
  rules            + features from the keyword classifier (forecast_worker.context.parse.rule_classify)
  llm              + features from a conventional LLM classifier (same questions, JSON answers)
  jev              + features from Jev's answers (abstentions become ctx_missing, never "no event")
  oracle_labels    + features from the construction labels: the ceiling of what these messages can give;
                   a reference line, never a candidate
  control_shuffled + Jev features moved to another series (fixed derangement): negative control
  control_shifted  + Jev features moved 26 weeks later: negative control (right series, wrong time)

Protocol: rolling origins on the last-sales week (CV), then a holdout evaluated once for every arm. There is
no arm selection inside the run, so nothing is tuned on the holdout. The evaluated sample is identical for
every arm (asserted). Point forecasts only: interval quality is covered by the main workbench evaluation.

    python -m forecast_worker.context.experiment DATA_DIR CLASSES_DIR OUT.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import LGBM_PARAMS
from .features import FEATURES, items, future_frame, training_frame
from .fixture import START
from .parse import read, rule_classify

H, GAP = 4, 1
CV_ORIGINS = list(range(60, 137, 4))          # last-sales week index
HOLDOUT_ORIGINS = [140, 144, 148]
SHIFT_WEEKS = 26
EXPERIMENT_VERSION = "context-exp-1"


def _week(i: int) -> date:
    return START + timedelta(weeks=i)


def _fit_predict(train: pd.DataFrame, future: pd.DataFrame | None, cols: list[str], steps: int) -> pd.DataFrame:
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    mf = MLForecast(models={"lgbm": lgb.LGBMRegressor(**LGBM_PARAMS)}, freq="W-MON", lags=[1, 2, 3, 4],
                    lag_transforms={1: [RollingMean(window_size=4)]}, date_features=["week"], num_threads=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf.fit(train[["unique_id", "ds", "y", *cols]], static_features=[])
        fc = mf.predict(steps, X_df=future[["unique_id", "ds", *cols]] if cols else None)
    return fc.rename(columns={"lgbm": "yhat"})


def load(data: Path) -> tuple[pd.DataFrame, list[dict]]:
    sales = pd.read_csv(data / "sales.csv")
    sales["ds"] = pd.to_datetime(sales["week_start"])
    panel = sales.rename(columns={"series_id": "unique_id", "quantity": "y"})[["unique_id", "ds", "y", "promo_ran"]]
    with (data / "context_messages.csv").open(newline="") as fh:
        messages = list(csv.DictReader(fh))
    return panel, messages


def classes_from_labels(messages: list[dict]) -> dict[str, dict]:
    return {m["message_id"]: {"event_type": m["label_event_type"], "status": m["label_status"],
                              "product": m["label_product"], "abstained": False} for m in messages}


def feature_frames(messages: list[dict], classes: dict[str, dict], series: list[str]) -> tuple[list, pd.DataFrame]:
    its = items([read(m) for m in messages], classes)
    return its, training_frame(its, series, [_week(i) for i in range(157)])


def _derange(series: list[str]) -> dict[str, str]:
    return {s: series[(i + 5) % len(series)] for i, s in enumerate(series)}   # fixed, no series maps to itself


def run(data: Path, classes_dir: Path) -> dict:
    panel, messages = load(data)
    series = sorted(panel["unique_id"].unique())
    sources = {"rules": {m["message_id"]: rule_classify(m["text"]) for m in messages},
               "oracle_labels": classes_from_labels(messages)}
    unavailable = {}
    for name in ("llm", "jev"):
        p = classes_dir / f"classes-{name}.json"
        if not p.exists():
            unavailable[name] = "no classification file"
            continue
        c = json.loads(p.read_text())
        missing = [m["message_id"] for m in messages if m["message_id"] not in c["classes"]]
        if missing:                                      # never fill gaps with zeros: the arm is not run
            unavailable[name] = f"{len(missing)} messages unclassified"
            continue
        sources[name] = c["classes"]
    frames = {}
    for name, cls in sources.items():
        its, tf = feature_frames(messages, cls, series)
        frames[name] = (its, tf, None)
    if "jev" in frames:
        its, tf, _ = frames["jev"]
        d = _derange(series)
        frames["control_shuffled"] = (its, tf.assign(unique_id=tf["unique_id"].map(d)), ("shuffled", d))
        frames["control_shifted"] = (its, tf.assign(ds=tf["ds"] + pd.Timedelta(weeks=SHIFT_WEEKS)), ("shifted", SHIFT_WEEKS))
    arms = ["numeric", *frames]
    scale = {s: float(np.abs(np.diff(g.sort_values("ds")["y"].to_numpy()[:61])).mean()) for s, g in panel.groupby("unique_id")}
    results: dict[str, list[pd.DataFrame]] = {a: [] for a in arms}
    for phase, origins in (("cv", CV_ORIGINS), ("holdout", HOLDOUT_ORIGINS)):
        for o in origins:
            last = pd.Timestamp(_week(o))
            train = panel[panel["ds"] <= last]
            targets = [_week(o + k) for k in range(1, GAP + H + 1)]
            for arm in arms:
                if arm == "numeric":
                    fc = _fit_predict(train, None, [], GAP + H)
                else:
                    its, tf, control = frames[arm]
                    tr = train.merge(tf, on=["unique_id", "ds"], how="left")
                    tr[FEATURES] = tr[FEATURES].fillna(0).astype(int)   # weeks before any message: genuinely no event
                    if control is None:
                        fut = future_frame(its, series, targets, _week(o))
                    elif control[0] == "shuffled":                       # another series' values, as of this issue time
                        fut = future_frame(its, series, targets, _week(o)).assign(
                            unique_id=lambda f: f["unique_id"].map(control[1]))
                    else:                                                # values of 26 weeks earlier, as of THAT issue time
                        back = timedelta(weeks=control[1])
                        fut = future_frame(its, series, [t - back for t in targets], _week(o) - back)
                        fut = fut.assign(ds=fut["ds"] + pd.Timedelta(weeks=control[1]))
                    fc = _fit_predict(tr, fut, FEATURES, GAP + H)
                fc["step"] = fc.groupby("unique_id").cumcount() + 1
                fc = fc[fc["step"] > GAP]
                fc = fc.merge(panel, on=["unique_id", "ds"], how="left")
                fc["horizon"], fc["origin"], fc["phase"] = fc["step"] - GAP, o, phase
                results[arm].append(fc)
    out = {"experiment": EXPERIMENT_VERSION, "learner": {"model": "LightGBM", "params": LGBM_PARAMS, "lags": [1, 2, 3, 4],
           "rolling_mean": 4, "date_features": ["week"]}, "horizon": H, "gap": GAP,
           "cv_origins": CV_ORIGINS, "holdout_origins": HOLDOUT_ORIGINS, "unavailable_arms": unavailable, "arms": {}}
    keys = None
    for arm in arms:
        df = pd.concat(results[arm], ignore_index=True)
        k = set(map(tuple, df[["unique_id", "origin", "ds"]].astype(str).to_numpy()))
        assert keys is None or k == keys, f"arm {arm} evaluated a different sample"
        keys = k
        df["ase"] = (df["yhat"] - df["y"]).abs() / df["unique_id"].map(scale)
        out["arms"][arm] = {ph: _metrics(g) for ph, g in df.groupby("phase")}
        results[arm] = df
    base = results["numeric"]
    for arm in arms[1:]:
        out["arms"][arm]["paired_vs_numeric"] = {ph: _paired(base[base["phase"] == ph], results[arm][results[arm]["phase"] == ph])
                                                 for ph in ("cv", "holdout")}
    out["sample"] = {"points_per_arm": len(keys), "series": len(series)}
    return out


def _metrics(g: pd.DataFrame) -> dict:
    promo, rest = g[g["promo_ran"] == 1], g[g["promo_ran"] == 0]
    return {"n": int(len(g)), "mase": round(float(g["ase"].mean()), 4),
            "mase_promo_weeks": round(float(promo["ase"].mean()), 4) if len(promo) else None,
            "mase_other_weeks": round(float(rest["ase"].mean()), 4) if len(rest) else None,
            "bias_promo_weeks": round(float((promo["yhat"] - promo["y"]).mean()), 3) if len(promo) else None,
            "n_promo_weeks": int(len(promo))}


def _paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Difference in MASE (arm minus numeric; negative = arm better), with a bootstrap 95% interval that
    resamples whole origins (errors within one origin are not independent)."""
    m = a.merge(b, on=["unique_id", "origin", "ds"], suffixes=("_a", "_b"))
    per = m.groupby("origin").apply(lambda g: (g["ase_b"] - g["ase_a"]).mean(), include_groups=False).to_numpy()
    rng = np.random.default_rng(0)
    boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return {"diff": round(float(per.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], "origins": int(len(per))}


if __name__ == "__main__":
    data, classes_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = run(data, classes_dir)
    res["inputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.glob("*.csv"))}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({a: {ph: v.get(ph, {}).get("mase") for ph in ("cv", "holdout")} for a, v in res["arms"].items()}))

