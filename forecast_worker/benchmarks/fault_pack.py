"""Fault pack: seeded, LABELLED modified copies of a prepared dataset, to test that the product notices each
kind of defect. These are fault tests: not untouched public data, and never evidence of forecast accuracy.

    python -m forecast_worker.benchmarks.fault_pack PREPARED_DIR OUT_DIR [--seed 20260928]

Each case is a folder with the modified sales file and a `fault.json` saying what was changed and what the
profile is expected to report:
  missing_periods     rows removed from two series              -> Q-missing-weeks (missing, never zero)
  duplicate_events    rows repeated                             -> duplicate_keys > 0
  ambiguous_dates     some dates written as 03/04/2011          -> unparsed_rows > 0 (never guessed as March or April)
  changed_identifier  one product's id changes half-way         -> Q-series-continuity (a series ends as another begins)
  invalid_units       some rows in "kg"                         -> Q-units
  ambiguous_target    two plausible quantity columns            -> Q-mapping with no suggested target
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

CASES = ("missing_periods", "duplicate_events", "ambiguous_dates", "changed_identifier", "invalid_units", "ambiguous_target")


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def build(src: Path, out: Path, seed: int = 20260928) -> dict:
    sales = sorted(src.glob("sales_*.csv"))[0]
    cols, rows = _read(sales)
    date_col = "week_start" if "week_start" in cols else "period_start"
    series = sorted({r["series_id"] for r in rows})
    rng = random.Random(seed)
    made = {}
    for case in CASES:
        c_cols, c_rows = list(cols), [dict(r) for r in rows]
        expect: dict = {}
        if case == "missing_periods":
            victims = series[:2]
            drop = {(s, d) for s in victims for d in rng.sample(sorted({r[date_col] for r in rows if r["series_id"] == s})[5:-5], 3)}
            c_rows = [r for r in c_rows if (r["series_id"], r[date_col]) not in drop]
            expect = {"question": "Q-missing-weeks", "series": victims, "removed_rows": len(drop)}
        elif case == "duplicate_events":
            dup = rng.sample(c_rows, 5)
            c_rows += [dict(r) for r in dup]
            expect = {"duplicate_keys_at_least": 5}
        elif case == "ambiguous_dates":
            for r in rng.sample(c_rows, 4):
                y, m, d = r[date_col].split("-")
                r[date_col] = f"{d}/{m}/{y}"
            expect = {"unparsed_rows_at_least": 4}
        elif case == "changed_identifier":
            s = series[0]
            dates = sorted({r[date_col] for r in c_rows if r["series_id"] == s})
            cut = dates[len(dates) // 2]
            for r in c_rows:
                if r["series_id"] == s and r[date_col] >= cut:
                    r["series_id"] = s + "-NEW"
            expect = {"question": "Q-series-continuity", "ended": s, "started": s + "-NEW"}
        elif case == "invalid_units":
            for r in rng.sample(c_rows, 6):
                r["unit"] = "kg"
            expect = {"question": "Q-units"}
        elif case == "ambiguous_target":
            qty = next(c for c in c_cols if c in ("quantity", "rentals", "qty"))
            c_cols.append("quantity_shipped")
            for r in c_rows:
                r["quantity_shipped"] = str(max(0, int(float(r[qty])) - rng.randint(0, 3)))
            expect = {"question": "Q-mapping", "suggested_target": None, "ambiguous": True}
        d = out / case
        d.mkdir(parents=True, exist_ok=True)
        with (d / sales.name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=c_cols)
            w.writeheader()
            w.writerows(c_rows)
        (d / "fault.json").write_text(json.dumps({"case": case, "label": "FAULT TEST: modified copy, not public data",
                                                  "source": str(sales.name), "seed": seed, "expect": expect}, indent=1) + "\n")
        made[case] = expect
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20260928)
    a = ap.parse_args()
    print(json.dumps(build(Path(a.src), Path(a.out), a.seed)))

