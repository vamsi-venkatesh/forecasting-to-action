"""Unfamiliar files: a person says what each column means, and the worker writes a standard copy of the data that the
rest of the pipeline (profile, baselines, evaluation, issuing) already understands. Files already in the standard
form (`sales_*.csv` with series_id, week_start|period_start, available_at) pass through unchanged, so every existing
result stays bit-identical.

The confirmed mapping (the answer to question Q-columns):
    {"columns": {"date": "Order date", "date_order": "iso"|"dmy"|"mdy", "series": ["Store", "SKU"] (or [] for one
                 series), "available_at": "Loaded at" | null, "lag_days": 2 | null},
     "quantity_columns": {"Units": "target"}, "frequency": "weekly"|"daily", "missing_weeks": "missing"|"zero"}

Rules, all stated to the person before they confirm:
- rows with the same series and period are ADDED UP (a transaction log becomes a period total);
- weekly means ISO weeks starting Monday; a daily file confirmed as weekly is summed into those weeks;
- when no column says when a row became known, it is assumed known `lag_days` after its period ends (the person
  states the delay; it is never guessed), and that assumption is recorded;
- a date or quantity that cannot be read is counted and left out, never guessed (e.g. 03/04 without a stated order).
Nothing here evaluates formulas or runs code from the file: cells are text until parsed as a date or a number.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path

PLANNING = {"inventory.csv", "receipts.csv", "rules.csv", "context_messages.csv", "supplier_messages.csv", "issued_forecast.csv"}
KNOWN_HINTS = ("avail", "publish", "load", "ingest", "receiv", "arriv", "known", "export", "extract")
ID_HINTS = ("id", "sku", "code", "product", "item", "store", "shop", "region", "country", "site", "location", "series",
            "category", "customer", "warehouse", "channel", "article", "artikel", "filiale", "branch", "segment")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")
DMY = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2}|\d{4})$")


def data_files(d: Path) -> list[Path]:
    return [p for p in sorted(d.glob("*.csv")) if p.name not in PLANNING and not p.name.startswith("batch_")]


def _read(p: Path) -> tuple[list[str], list[dict[str, str]]]:
    with p.open(newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        return [c.strip() for c in (r.fieldnames or [])], [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in r]


def is_standard(p: Path) -> bool:
    cols, _ = _read_header(p)
    return p.name.startswith("sales_") and "series_id" in cols and ("week_start" in cols or "period_start" in cols) \
        and "available_at" in cols


def _read_header(p: Path) -> tuple[list[str], None]:
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return [c.strip() for c in (next(csv.reader(fh), None) or [])], None


def needs_mapping(d: Path) -> bool:
    return any(not is_standard(p) for p in data_files(d))


def _number(v: str) -> float | None:
    v = v.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return float(v)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", v):          # 1,234.5
        return float(v.replace(",", ""))
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?|-?\d+,\d+", v):   # 1.234,5 or 12,5 (decimal comma)
        return float(v.replace(".", "").replace(",", "."))
    return None


def parse_date(v: str, order: str) -> date | None:
    v = v.strip()
    if ISO.match(v):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00").replace(" ", "T") if "T" in v or " " in v else v).date()
        except ValueError:
            return None
    m = DMY.match(v)
    if m and order in ("dmy", "mdy"):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        d_, mo = (a, b) if order == "dmy" else (b, a)
        try:
            return date(y, mo, d_)
        except ValueError:
            return None
    return None


def _moment(v: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _kind(values: list[str]) -> tuple[str, str | None]:
    """'datetime' | 'date' | 'number' | 'text', and for dates the order: 'iso', 'dmy', 'mdy' or 'ambiguous'."""
    vals = [v for v in values if v][:300]
    if not vals:
        return "text", None
    if sum(bool(ISO.match(v)) for v in vals) >= .9 * len(vals):
        return ("datetime" if any(("T" in v or " " in v) for v in vals) else "date"), "iso"
    dm = [DMY.match(v) for v in vals]
    if sum(m is not None for m in dm) >= .9 * len(vals):
        firsts = [int(m.group(1)) for m in dm if m]
        seconds = [int(m.group(2)) for m in dm if m]
        if any(f > 12 for f in firsts) and all(s <= 12 for s in seconds):
            return "date", "dmy"
        if any(s > 12 for s in seconds) and all(f <= 12 for f in firsts):
            return "date", "mdy"
        return "date", "ambiguous"
    if sum(_number(v) is not None for v in vals) >= .95 * len(vals):
        return "number", None
    return "text", None


def propose(d: Path) -> dict:
    """The profile's SUGGESTION for an unfamiliar file: column kinds, a date column, identifying columns chosen so that
    (series, date) is unique where possible, a quantity, an arrival-time column if one is named so. A person confirms."""
    files, cols_all, rows_all = [], None, []
    for p in data_files(d):
        cols, rows = _read(p)
        kinds = {c: _kind([r.get(c, "") for r in rows]) for c in cols}
        files.append({"file": p.name, "rows": len(rows), "columns": [
            {"name": c, "kind": kinds[c][0], "date_order": kinds[c][1],
             "examples": [r.get(c, "") for r in rows[:3]], "distinct": len({r.get(c, "") for r in rows[:5000]})} for c in cols]})
        cols_all = cols if cols_all is None else [c for c in cols_all if c in cols]
        rows_all += rows
    cols_all = cols_all or []
    meta = {c["name"]: c for f in files for c in f["columns"] if c["name"] in cols_all}
    dates = [c for c in cols_all if meta[c]["kind"] == "date"]
    moments = [c for c in cols_all if meta[c]["kind"] == "datetime"]
    known = next((c for c in moments + dates if any(h in c.lower() for h in KNOWN_HINTS)), None)
    date_col = next((c for c in dates + moments if c != known), None)
    numbers = [c for c in cols_all if meta[c]["kind"] == "number" and c not in (date_col, known)]
    idlike = lambda c: any(h in c.lower() for h in ID_HINTS)
    quantities = [c for c in numbers if not idlike(c)]
    texts = [c for c in cols_all if meta[c]["kind"] == "text"] + [c for c in numbers if idlike(c)]
    # a column with one value distinguishes nothing; one with a value per row is a row id, not a series
    candidates = sorted([c for c in texts if 1 < meta[c]["distinct"] < max(3, len(rows_all) * .5)],
                        key=lambda c: (not idlike(c), meta[c]["distinct"]))[:5]
    series: list[str] = []
    if date_col:
        order = meta[date_col]["date_order"] or "iso"
        sample = rows_all[:20000]
        for k in range(0, 3):                                     # the smallest set of columns that makes (series, date) unique
            found = None
            for combo in combinations(candidates, k):
                keys = Counter((tuple(r.get(c, "") for c in combo), r.get(date_col, "")) for r in sample)
                if keys and max(keys.values()) == 1:
                    found = list(combo)
                    break
            if found is not None:
                series = found
                break
        else:
            series = [c for c in candidates if idlike(c)][:3]      # still duplicates (a transaction log): added up per period
        parsed = sorted({x for x in (parse_date(r.get(date_col, ""), order if order != "ambiguous" else "iso") for r in sample) if x})
        gaps = Counter((b - a).days for a, b in zip(parsed, parsed[1:]))
        step = gaps.most_common(1)[0][0] if gaps else None
    else:
        order, step = None, None
    plausible = [c for c in quantities if any(w in c.lower() for w in ("qty", "quantity", "units", "menge", "sold", "sales",
                                                                        "volume", "count", "demand", "orders", "rentals", "stück"))]
    quantity = plausible[0] if len(plausible) == 1 else (quantities[0] if len(quantities) == 1 else None)
    return {"files": files, "columns": cols_all,
            "suggested": {"date": date_col, "date_order": order, "series": series, "quantity": quantity,
                          "available_at": known, "lag_days": None,
                          "frequency": {1: "daily", 7: "weekly"}.get(step or 0, "weekly" if step and step > 1 else "daily"),
                          "step_days": step},
            "quantity_candidates": quantities, "date_candidates": dates + moments,
            "ambiguous": {"date_order": order == "ambiguous", "quantity": quantity is None, "date": date_col is None}}


def standardize_batches(d: Path, answers: dict) -> dict:
    """A recurring source's delivered batches (batch_*.csv) get the same standard copy, under their own names."""
    files = [p for p in sorted(d.glob("batch_*.csv")) if not _has_standard_columns(p)]
    return canonicalize(d, answers, files=files, keep_names=True)


def _has_standard_columns(p: Path) -> bool:
    cols, _ = _read_header(p)
    return "series_id" in cols and ("week_start" in cols or "period_start" in cols) and "available_at" in cols


def canonicalize(d: Path, answers: dict, *, files: list[Path] | None = None, keep_names: bool = False) -> dict:
    """Write the standard copy (sales_<file>.csv) of every unfamiliar data file from the CONFIRMED mapping, in place of
    the original in this job's private staging folder. Returns what was done, with counts of rows left out."""
    m = (answers or {}).get("columns")
    if not m:
        return {"mapped": False}
    date_col, order, series_cols = m.get("date"), m.get("date_order") or "iso", list(m.get("series") or [])
    known_col, lag = m.get("available_at"), m.get("lag_days")
    qmap = answers.get("quantity_columns") or {}
    freq = answers.get("frequency") or "daily"
    if not date_col or not qmap:
        raise SystemExit("the column mapping needs a date column and a quantity")
    if order == "ambiguous":
        raise SystemExit("say whether dates are day/month or month/day before anything is read")
    if not known_col and not (isinstance(lag, int) and 0 <= lag <= 365):
        raise SystemExit("say when a row becomes known: pick the column that records it, or the delay in days after each period")
    qcol = next(iter(qmap))
    step = timedelta(days=7 if freq == "weekly" else 1)
    report = {"mapped": True, "files": [], "rule": {
        "series": series_cols or "one series", "date": f"{date_col} ({order})", "quantity": qcol, "frequency": freq,
        "same period rows": "added up", "arrival": f"column {known_col}" if known_col else f"assumed {lag} day(s) after the period ends"}}
    for p in (files if files is not None else data_files(d)):
        if files is None and is_standard(p):
            continue
        cols, rows = _read(p)
        missing = [c for c in [date_col, qcol, *series_cols, *( [known_col] if known_col else [])] if c not in cols]
        if missing:
            raise SystemExit(f"{p.name}: the confirmed mapping names columns this file does not have: {', '.join(missing)}")
        totals: dict[tuple[str, date], list] = {}
        bad_date = bad_qty = bad_known = 0
        for r in rows:
            day = parse_date(r[date_col], order)
            if day is None:
                bad_date += 1
                continue
            q = _number(r[qcol])
            if q is None:
                bad_qty += 1
                continue
            start = day - timedelta(days=day.weekday()) if freq == "weekly" else day
            sid = "|".join(r[c] for c in series_cols) if series_cols else "all"
            if known_col:
                t = _moment(r[known_col])
                if t is None:
                    bad_known += 1
                    continue
            else:
                t = datetime.combine(start + step, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=lag)
            cur = totals.setdefault((sid, start), [0.0, t])
            cur[0] += q
            cur[1] = max(cur[1], t)
        out = p if keep_names else d / f"sales_{re.sub(r'[^A-Za-z0-9_-]+', '_', p.stem)[:60] or 'data'}.csv"
        date_name = "week_start" if freq == "weekly" else "period_start"
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["series_id", date_name, "available_at", "quantity"])
            for (sid, start), (q, t) in sorted(totals.items()):
                w.writerow([sid, start.isoformat(), t.isoformat(), f"{q:g}"])
        if out != p:
            p.unlink()
        report["files"].append({"file": p.name, "rows": len(rows), "periods": len(totals), "unread_dates": bad_date,
                                "unread_quantities": bad_qty, "unread_arrival_times": bad_known, "written_as": out.name})
    # the standard copy names the confirmed quantity "quantity": the jobs that follow read it under that name
    answers["quantity_columns"] = {**qmap, "quantity": qmap[qcol]}
    return report

