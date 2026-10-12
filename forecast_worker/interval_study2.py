"""Interval study 2: can the LightGBM model's 80% ranges be calibrated? Rules frozen (hashed) before any result.

    python -m forecast_worker.interval_study2 freeze OUT_DIR
    python -m forecast_worker.interval_study2 run OUT_DIR --phase dev  NW_A UCI NW_B BIKE
    python -m forecast_worker.interval_study2 run OUT_DIR --phase test NW_C SMARD_AREAS

Why a second study: study 1 (interval_study.py) was NOT validated: pooled_w10 covered 82.0% on a fresh NordWerk
realisation but 58.0% on Bike Sharing daily. All four of its datasets have now been SEEN, so they are development
data here. The fix is chosen on them; the verdict comes only from two datasets no interval study has touched.
Measurement code is study 1's (run_dataset); only the methods, datasets and rules below are new.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .interval_study import run_dataset

STUDY = {
    "version": "interval-study-2",
    "level": 80,
    "methods": ["pooled_w10", "pooled_w20", "spread_y1"],
    "method_notes": {
        "pooled_w10": "the current method (study 1's choice): 10 consecutive calibration origins",
        "pooled_w20": "20 consecutive calibration origins",
        "spread_y1": "calibration origins spread over the last year: weekly 26 origins 2 weeks apart, daily 52 origins "
                     "7 days apart (hypothesis from study 1: consecutive daily origins give one series too few, too "
                     "similar scores)"},
    "development": {
        "nordwerk_a": "NordWerk seed 20260928: cross-validation origins only (as study 1)",
        "uci_retail": "UCI Online Retail II weekly, 50 products: cross-validation origins only (as study 1)",
        "nordwerk_b": "NordWerk seed 20261015 (study 1's test set, now seen): cross-validation and holdout origins",
        "bike_daily": "UCI Bike Sharing daily (study 1's test set, now seen): cross-validation and holdout origins"},
    "test": {
        "nordwerk_c": "a NEW NordWerk realisation, seed 20261101 (synthetic, weekly): all origins",
        "smard_areas": "SMARD daily electricity consumption (grid load) of the four German control areas 50Hertz, "
                       "Amprion, TenneT, TransnetBW, 2019-01-01 to the last published day at retrieval (real, CC BY "
                       "4.0, Bundesnetzagentur | SMARD.de): all origins; no interval study has used these series"},
    "selection": "among methods whose development coverage is within 0.75..0.85 on ALL FOUR development datasets, the "
                 "lowest mean scaled interval score (mean over the four); if none qualifies, the method with the most "
                 "development datasets inside 0.75..0.85 (ties: lowest mean scaled interval score) is tested, and "
                 "can at best be reported, not validated",
    "validation": "the selected method is 'validated on these datasets' only if it qualified on development AND its "
                  "test coverage is within 0.75..0.85 on BOTH test datasets; otherwise ranges stay labelled not "
                  "validated and nothing may rely on them",
    "reported": "per dataset and method: points n, series, origins, empirical coverage, mean scaled width, mean scaled "
                "interval score (alpha 0.2, study 1's definition), coverage by horizon",
    "not_allowed": "no change of rules, methods, datasets or thresholds after seeing development or test results",
}
DATASETS = {
    "nordwerk_a": ({"answers": {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}}, False),
    "uci_retail": ({"answers": {"quantity_columns": {"quantity": "net_units"}, "missing_weeks": "zero"}}, False),
    "nordwerk_b": ({"answers": {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}}, True),
    "bike_daily": ({"answers": {"quantity_columns": {"rentals": "rentals"}, "missing_weeks": "zero"}, "frequency": "daily"}, True),
    "nordwerk_c": ({"answers": {"quantity_columns": {"qty": "net_units", "quantity": "net_units"}, "missing_weeks": "missing"}}, True),
    "smard_areas": ({"answers": {"quantity_columns": {"consumption_mwh": "consumption_mwh"}, "missing_weeks": "missing"},
                     "frequency": "daily"}, True),
}
PHASES = {"dev": ["nordwerk_a", "uci_retail", "nordwerk_b", "bike_daily"], "test": ["nordwerk_c", "smard_areas"]}


def study_sha() -> str:
    return hashlib.sha256(json.dumps(STUDY, sort_keys=True).encode()).hexdigest()


def inside(c: float) -> bool:
    return 0.75 <= c <= 0.85


def select(dev: dict) -> dict:
    cov = {m: [dev[d]["results"][m]["coverage"] for d in dev] for m in STUDY["methods"]}
    score = {m: float(np.mean([dev[d]["results"][m]["mean_scaled_interval_score"] for d in dev])) for m in STUDY["methods"]}
    ok = [m for m in STUDY["methods"] if all(inside(c) for c in cov[m])]
    if ok:
        best = min(ok, key=score.get)
        return {"selected": best, "qualified": True, "mean_scaled_interval_score": {m: round(v, 4) for m, v in score.items()},
                "reason": f"{best}: lowest mean scaled interval score among methods covering 75-85% on all four development sets"}
    best = min(STUDY["methods"], key=lambda m: (-sum(inside(c) for c in cov[m]), score[m]))
    return {"selected": best, "qualified": False, "mean_scaled_interval_score": {m: round(v, 4) for m, v in score.items()},
            "reason": f"no method covered 75-85% on all four development sets; {best} had the most inside "
                      f"({sum(inside(c) for c in cov[best])}/4) and is tested for the record only"}


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
    phase = sys.argv[sys.argv.index("--phase") + 1]
    paths = [Path(p) for p in sys.argv[sys.argv.index("--phase") + 2:]]
    names = PHASES[phase]
    if len(paths) != len(names):
        raise SystemExit(f"{phase} needs {len(names)} dataset folders: {', '.join(names)}")
    if phase == "dev":
        dev = {n: run_dataset(p, DATASETS[n][0], STUDY["methods"], DATASETS[n][1]) for n, p in zip(names, paths)}
        decision = select(dev)
        doc = {"study_sha256": frozen["sha256"], "development": dev, "decision": decision,
               "decided_at": datetime.now(timezone.utc).isoformat()}
        doc["decision_sha256"] = hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()
        (out / "development.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(json.dumps(decision))
        print(json.dumps({d: {m: {k: v[k] for k in ("coverage", "mean_scaled_width", "mean_scaled_interval_score", "n")}
                              for m, v in dev[d]["results"].items()} for d in dev}))
        return
    dec = json.loads((out / "development.json").read_text())["decision"]
    sel = dec["selected"]
    methods = ["pooled_w10"] + ([sel] if sel != "pooled_w10" else [])     # the incumbent is always reported beside it
    test = {n: run_dataset(p, DATASETS[n][0], methods, DATASETS[n][1]) for n, p in zip(names, paths)}
    valid = dec["qualified"] and all(inside(test[d]["results"][sel]["coverage"]) for d in test)
    doc = {"study_sha256": frozen["sha256"], "decision": dec, "test": test,
           "verdict": "validated on these datasets" if valid else "NOT validated",
           "tested_at": datetime.now(timezone.utc).isoformat()}
    (out / "test.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"verdict": doc["verdict"], **{d: {m: {k: v[k] for k in ("coverage", "mean_scaled_width", "mean_scaled_interval_score", "n")}
                                                      for m, v in test[d]["results"].items()} for d in test}}))


if __name__ == "__main__":
    main()

