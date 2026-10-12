"""Which interval method may the LightGBM model use? A frozen study (rules hashed before any result).

    python -m forecast_worker.interval_study freeze OUT_DIR            # writes study_frozen.json (rules + sha256)
    python -m forecast_worker.interval_study run OUT_DIR NW_A UCI NW_B BIKE [--phase dev|test]

Why: the original conformal intervals (3 windows per series) covered 74.5% in cross-validation and 64.2% on the
NordWerk holdout for a nominal 80% (reproduced 2026-09-28). That holdout has now been SEEN and is not used here.
Point forecasts are identical for every method (same model, same fit); only the interval changes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import _load, _scale
from .calendar import horizon, of
from .evaluate import _forecast_at

STUDY = {
    "version": "interval-study-1",
    "level": 80,
    "methods": ["conformal_w3", "pooled_w10", "pooled_w20"],
    "development": {
        "nordwerk_a": "NordWerk Demo fixture seed 20260928: the 12 rolling cross-validation origins only; its holdout "
                      "(already seen: 64.2%) is excluded",
        "uci_retail": "UCI Online Retail II weekly v2 (50 products chosen on training-period data): the 12 rolling "
                      "cross-validation origins only; its holdout is kept for the public-data evaluation"},
    "test": {
        "nordwerk_b": "a NEW NordWerk realisation, seed 20261015 (synthetic): all cross-validation and holdout origins",
        "bike_daily": "UCI Bike Sharing daily total rentals (real): all cross-validation and holdout origins; never "
                      "evaluated before"},
    "selection": "among methods whose development coverage is within 0.75..0.85 on BOTH development datasets, the "
                 "lowest mean scaled interval score (mean of the two datasets); if none qualifies, NO method is "
                 "selected and intervals stay unvalidated",
    "validation": "the selected method is 'validated on these datasets' only if its test coverage is within "
                  "0.75..0.85 on BOTH test datasets; otherwise intervals are labelled not validated and nothing may "
                  "rely on them",
    "interval_score": "Gneiting-Raftery interval score: (hi - lo) + (2/alpha)(lo - y)+ + (2/alpha)(y - hi)+, alpha = 0.2, "
                      "divided by the series scale (in-sample one-step naive MAE on data before the evaluation window)",
    "not_allowed": "no change of rules, methods, datasets or thresholds after seeing development or test results",
}
DATASETS = {
    "nordwerk_a": {"answers": {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}},
    "uci_retail": {"answers": {"quantity_columns": {"quantity": "net_units"}, "missing_weeks": "zero"}},
    "nordwerk_b": {"answers": {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}},
    "bike_daily": {"answers": {"quantity_columns": {"rentals": "rentals"}, "missing_weeks": "zero"}, "frequency": "daily"},
}


def study_sha() -> str:
    return hashlib.sha256(json.dumps(STUDY, sort_keys=True).encode()).hexdigest()


def _origins(panel, params):
    cal = of(params)
    h, gap, holdout, n = horizon(params, cal)
    cut_end = panel["ds"].max() - cal.periods(holdout)
    cv = [cut_end - cal.periods(h + gap + k) for k in range(n)][::-1]
    ho = [cut_end - cal.periods(gap) + cal.periods(h * k) for k in range(holdout // h)]
    return cal, h, gap, cut_end, cv, ho


def run_dataset(data: Path, params: dict, methods: list[str], include_holdout: bool) -> dict:
    cal = of(params)
    panel, _ = _load(data, params["answers"], cal)
    cal, h, gap, cut_end, cv, ho = _origins(panel, params)
    first = cv[0]
    scales = {sid: _scale(g[g["ds"] <= first]["y"].to_numpy()) for sid, g in panel.groupby("unique_id")}
    out = {}
    for m in methods:
        t0 = time.time()
        frames = []
        for o in cv + (ho if include_holdout else []):
            fc, _ = _forecast_at(panel, o, h, gap, {"lgbm_lags"}, cal, m)
            frames.append(fc.assign(split="holdout" if o in ho and o not in cv else "cv"))
        df = pd.concat(frames).merge(panel, on=["unique_id", "ds"], how="left")
        df = df[df["y"].notna()]                      # forecasts past the data's last period have no actual
        df = df[~df["imputed"].astype(bool)]
        if not include_holdout:
            df = df[df["ds"] <= cut_end]
        sc = df["unique_id"].map(scales)
        df = df[sc.notna()].assign(scale=sc[sc.notna()])
        a = 1 - STUDY["level"] / 100
        inside = (df["y"] >= df["lo"]) & (df["y"] <= df["hi"])
        iscore = ((df["hi"] - df["lo"]) + (2 / a) * (df["lo"] - df["y"]).clip(lower=0)
                  + (2 / a) * (df["y"] - df["hi"]).clip(lower=0)) / df["scale"]
        err = (df["yhat"] - df["y"]).abs() / df["scale"]
        out[m] = {"n": int(len(df)), "series": int(df["unique_id"].nunique()), "origins": len(cv) + (len(ho) if include_holdout else 0),
                  "coverage": round(float(inside.mean()), 4),
                  "mean_scaled_width": round(float(((df["hi"] - df["lo"]) / df["scale"]).mean()), 4),
                  "mean_scaled_interval_score": round(float(iscore.mean()), 4),
                  "point_mase": round(float(err.mean()), 4),
                  "coverage_by_horizon": {int(k): round(float(v.mean()), 4) for k, v in inside.groupby(df["horizon"])},
                  "seconds": round(time.time() - t0, 1)}
    return {"frequency": cal.name, "horizon": h, "gap": gap, "evaluation_until": str(cut_end.date()) if not include_holdout else "end",
            "results": out}


def select(dev: dict) -> dict:
    ok = [m for m in STUDY["methods"] if all(0.75 <= dev[d]["results"][m]["coverage"] <= 0.85 for d in dev)]
    if not ok:
        return {"selected": None, "reason": "no method covered 75-85% on both development datasets"}
    score = {m: np.mean([dev[d]["results"][m]["mean_scaled_interval_score"] for d in dev]) for m in ok}
    best = min(score, key=score.get)
    return {"selected": best, "qualified": ok, "mean_scaled_interval_score": {m: round(float(v), 4) for m, v in score.items()},
            "reason": f"{best}: lowest mean scaled interval score among methods covering 75-85% on both development sets"}


def main() -> None:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "freeze":
        (out / "study_frozen.json").write_text(json.dumps({"study": STUDY, "sha256": study_sha(),
                                                           "frozen_at": datetime.now(timezone.utc).isoformat()}, indent=1) + "\n")
        print(study_sha())
        return
    frozen = json.loads((out / "study_frozen.json").read_text())
    if frozen["sha256"] != study_sha():
        raise SystemExit("the study rules changed after they were frozen: refused")
    nw_a, uci, nw_b, bike = map(Path, sys.argv[3:7])
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "dev"
    if phase == "dev":
        dev = {"nordwerk_a": run_dataset(nw_a, DATASETS["nordwerk_a"], STUDY["methods"], False),
               "uci_retail": run_dataset(uci, DATASETS["uci_retail"], STUDY["methods"], False)}
        decision = select(dev)
        doc = {"study_sha256": frozen["sha256"], "development": dev, "decision": decision,
               "decided_at": datetime.now(timezone.utc).isoformat()}
        doc["decision_sha256"] = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()
        (out / "development.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(decision))
    else:
        dec = json.loads((out / "development.json").read_text())["decision"]
        methods = ["conformal_w3"] + ([dec["selected"]] if dec["selected"] and dec["selected"] != "conformal_w3" else [])
        test = {"nordwerk_b": run_dataset(nw_b, DATASETS["nordwerk_b"], methods, True),
                "bike_daily": run_dataset(bike, DATASETS["bike_daily"], methods, True)}
        sel = dec["selected"]
        valid = bool(sel) and all(0.75 <= test[d]["results"][sel]["coverage"] <= 0.85 for d in test)
        doc = {"study_sha256": frozen["sha256"], "decision": dec, "test": test,
               "verdict": ("validated on these datasets" if valid else "NOT validated"),
               "tested_at": datetime.now(timezone.utc).isoformat()}
        (out / "test.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps({"verdict": doc["verdict"], **{d: {m: {k: v[k] for k in ("coverage", "mean_scaled_width", "mean_scaled_interval_score", "n")}
                                                              for m, v in test[d]["results"].items()} for d in test}}))


if __name__ == "__main__":
    main()

