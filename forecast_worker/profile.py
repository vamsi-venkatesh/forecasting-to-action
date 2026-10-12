"""Deterministic data profile of a sales upload (brief A2). No model reads the data here.

Every statement in the profile is a computed field, tied to the SHA-256 of the input files it was computed
from. Nothing is normalised silently: a missing week is reported as missing, never as zero; a week sold out
is reported as censored, never as low demand; a renamed column is reported, and the mapping it needs is an
open question until someone answers it.

    python -m forecast_worker.profile DATA_DIR OUT.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

PROFILE_VERSION = "profile-2"          # v2: daily or weekly periods; a mapping to confirm for any dataset
QTY_COLUMNS = ("qty", "quantity", "units", "menge")
NOT_A_TARGET = {"returned", "source_revision", "instant", "yr", "mnth", "season", "holiday", "weekday", "workingday"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    # plain csv: formulas, macros and embedded instructions are text, never evaluated
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def profile(data_dir: Path, *, sales_glob: str = "sales_*.csv", inventory: str = "inventory.csv") -> dict:
    """Weekly (week_start) or daily/other periods (period_start); the period length is measured, not assumed."""
    sales_files = sorted(data_dir.glob(sales_glob))
    if not sales_files:
        raise SystemExit(f"no sales files matching {sales_glob} in {data_dir}")
    inputs = {p.name: _sha(p) for p in sales_files}
    files, questions, all_rows = [], [], []
    schemas = {}
    for p in sales_files:
        cols, rows = _rows(p)
        qty_cols = [c for c in cols if c.lower() in QTY_COLUMNS]
        schemas[p.name] = cols
        files.append({"file": p.name, "rows": len(rows), "columns": cols, "quantity_column": qty_cols[0] if len(qty_cols) == 1 else None})
        for r in rows:
            all_rows.append((p.name, qty_cols[0] if len(qty_cols) == 1 else None, r))
    # a column present in one export and a similar one in another: a mapping to confirm, never assumed
    names = [set(c) for c in schemas.values()]
    renamed = []
    if len(names) > 1:
        only = [sorted(s - set.intersection(*names)) for s in names]
        qty_sets = [sorted(set(o) & set(QTY_COLUMNS)) for o in only]
        if all(qty_sets) and len({tuple(q) for q in qty_sets}) > 1:
            renamed.append({"files": list(schemas), "columns": [q[0] for q in qty_sets]})
            questions.append({"id": "Q-quantity-column", "blocks": ["prepare", "train"],
                              "question": f"The exports name the sold quantity differently ({' / '.join(q[0] for q in qty_sets)}). "
                                          "Are they the same measure, in the same unit, net or gross of returns?",
                              "evidence": {"columns": {f: [c for c in schemas[f] if c.lower() in QTY_COLUMNS] for f in schemas}}})
    # keys and duplicates
    date_col = "week_start" if all("week_start" in c for c in schemas.values()) else "period_start"
    keys = Counter((r.get("series_id"), r.get(date_col)) for _, _, r in all_rows)
    duplicates = sorted([{"series_id": k[0], date_col: k[1], "rows": n} for k, n in keys.items() if n > 1],
                        key=lambda d: (d["series_id"], d[date_col]))
    units = Counter(r.get("unit") for _, _, r in all_rows)
    if len(units) > 1:
        questions.append({"id": "Q-units", "blocks": ["prepare", "train"],
                          "question": f"Rows use more than one unit ({', '.join(f'{u}: {n}' for u, n in units.items())}).",
                          "evidence": {"units": dict(units)}})
    # per series: coverage, missing weeks (no row) vs zero weeks (row with 0), returns
    by: dict[str, dict[date, dict]] = defaultdict(dict)
    unparsed = 0
    for fname, qcol, r in all_rows:
        try:
            wk = date.fromisoformat(r[date_col])
            v = float(r[qcol]) if qcol else None
            q = (int(v) if v.is_integer() else v) if v is not None else None      # decimals are quantities too (kg, hours)
        except (KeyError, ValueError, TypeError):
            unparsed += 1
            continue
        by[r["series_id"]][wk] = {"qty": q, "returned": int(r.get("returned") or 0), "file": fname}
    last_week = max(w for s in by.values() for w in s)
    # the period is measured from the data: the most common gap between consecutive dates of a series
    gaps = Counter((b - a).days for s_ in by.values() for a, b in zip(sorted(s_), sorted(s_)[1:]))
    step_days = gaps.most_common(1)[0][0] if gaps else 7
    frequency = {7: "weekly", 1: "daily"}.get(step_days)
    season, per_year = (52, 104) if step_days == 7 else (7, 14)
    stock = _inventory(data_dir / inventory) if (data_dir / inventory).exists() else {}
    if stock:
        inputs[inventory] = _sha(data_dir / inventory)
    series = []
    for sid in sorted(by):
        weeks = by[sid]
        first = min(weeks)
        expected = [first + timedelta(days=step_days * i) for i in range((last_week - first).days // step_days + 1)]
        missing = [w.isoformat() for w in expected if w not in weeks]
        zeros = [w.isoformat() for w, v in weeks.items() if v["qty"] == 0]
        censored = [w.isoformat() for w in sorted(weeks) if stock.get((sid, w)) == 0]
        q = [v["qty"] for v in weeks.values() if v["qty"] is not None]
        series.append({"series_id": sid, "first_week": first.isoformat(), "last_week": max(weeks).isoformat(),
                       "observed_weeks": len(weeks), "expected_weeks": len(expected), "missing_weeks": missing,
                       "zero_weeks": len(zeros), "stockout_weeks": censored,
                       "returned_units": sum(v["returned"] for v in weeks.values()),
                       "sold_units": sum(q), "mean_weekly": round(sum(q) / len(q), 3) if q else None,
                       "history_ok_for_52w_seasonality": len(weeks) >= 104,
                       "history_ok_for_seasonality": len(weeks) >= per_year})
    short = [s["series_id"] for s in series if not s["history_ok_for_52w_seasonality"]]
    # a series that stops while another starts in the very next period: possibly one product under a new id
    last_all = max(date.fromisoformat(x["last_week"]) for x in series)
    handovers = [{"ended": a["series_id"], "last": a["last_week"], "started": b["series_id"], "first": b["first_week"]}
                 for a in series for b in series
                 if a is not b and date.fromisoformat(a["last_week"]) < last_all
                 and date.fromisoformat(b["first_week"]) == date.fromisoformat(a["last_week"]) + timedelta(days=step_days)]
    if handovers:
        questions.append({"id": "Q-series-continuity", "blocks": [],
                          "question": f"{len(handovers)} series stop exactly when another starts. Is each pair one product "
                                      "whose identifier changed? They are kept apart unless you say so.",
                          "evidence": {"pairs": handovers}})
    with_missing = [s["series_id"] for s in series if s["missing_weeks"]]
    with_stockouts = [s["series_id"] for s in series if s["stockout_weeks"]]
    if with_missing:
        questions.append({"id": "Q-missing-weeks", "blocks": ["train"],
                          "question": f"{len(with_missing)} series have weeks with no row at all. Were those weeks without sales "
                                      "(treat as 0) or not recorded (leave missing)?",
                          "evidence": {s["series_id"]: s["missing_weeks"] for s in series if s["missing_weeks"]}})
    if with_stockouts:
        questions.append({"id": "Q-stockouts", "blocks": [],
                          "question": f"{len(with_stockouts)} series sold while usable stock was zero at the week's end: those "
                                      "weeks show sales limited by stock, not demand. Exclude them from training, keep them, or "
                                      "keep them flagged?",
                          "evidence": {s["series_id"]: s["stockout_weeks"] for s in series if s["stockout_weeks"]}})
    # what to forecast: confirmed by a person for every dataset, unless the renamed-column question already asks
    ident = lambda c: c.lower() in ("id", "sku", "code") or c.lower().endswith(("_id", "_code", "_sku", "_nr", "_no"))
    numeric = sorted({c for f in files for c in f["columns"] if c not in NOT_A_TARGET and c != "series_id" and not ident(c)
                      and all(_is_number(r.get(c)) for _, _, r in all_rows[:200] if r.get(c) not in (None, ""))
                      and any(r.get(c) not in (None, "") for _, _, r in all_rows[:200])})
    # quantity-like names: more than one of them is ambiguous, and then nothing is suggested
    plausible = [c for c in numeric if c.lower() in QTY_COLUMNS
                 or any(w in c.lower() for w in ("qty", "quantity", "units", "menge", "sold", "shipped", "ordered", "rental"))]
    if not renamed:
        proposed = plausible[0] if len(plausible) == 1 else (numeric[0] if len(numeric) == 1 and not plausible else None)
        questions.insert(0, {"id": "Q-mapping", "blocks": ["prepare", "train"],
                             "question": "Confirm what to forecast. The profile SUGGESTS a mapping below; nothing trains "
                                         "until a person confirms it.",
                             "evidence": {"suggested": {"target": proposed, "date": date_col, "series": "series_id",
                                                        "frequency": frequency, "unit": next(iter(units), None)},
                                          "target_candidates": numeric, "measured_step_days": step_days,
                                          "ambiguous": proposed is None,
                                          "missing_periods": sum(len(s_["missing_weeks"]) for s_ in series)}})
    return {"profile_version": PROFILE_VERSION, "inputs": inputs, "date_column": date_col, "frequency": frequency,
            "measured_step_days": step_days, "target_candidates": numeric,
            "inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
            "files": files, "renamed_columns": renamed, "duplicate_keys": duplicates, "unparsed_rows": unparsed,
            "units": dict(units), "last_week": last_week.isoformat(), "series_count": len(series),
            "series_short_for_52w_seasonality": short, "series": series, "questions": questions}


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _inventory(path: Path) -> dict[tuple[str, date], int]:
    _, rows = _rows(path)
    out = {}
    for r in rows:
        try:
            snap = date.fromisoformat(r["snapshot_time"][:10])
            out[(r["series_id"], _monday(snap))] = int(r["usable_on_hand"])
        except (KeyError, ValueError):
            continue
    return out


if __name__ == "__main__":
    result = profile(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("profile_version", "inputs_sha256", "series_count", "unparsed_rows")}
                     | {"duplicates": len(result["duplicate_keys"]), "questions": [q["id"] for q in result["questions"]]}))

