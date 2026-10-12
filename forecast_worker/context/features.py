"""From classified messages to per-series, per-week features, using only what was known at a given time.

Code does all of this: the date comparisons, the "latest message wins" rule and the aggregation. A classifier
(rules, a conventional LLM or Jev) supplies only each message's event type, status and, when a message names
more than one product, which product it is about.

Features for series s, target week t, as of time a (only messages with available_at <= a):
  promo_confirmed  the latest promotion message for s covering t says confirmed
  promo_tentative  the latest one says tentative (or t is the new slot of a postponement)
  supply_delay     a supplier-delay message for s covers t. A SUPPLY constraint, never read as a demand change
  ctx_missing      a message for s covering t could not be classified (abstained, failed): the value is unknown,
                   which is different from "no event"
A postponement's first span is its old slot (not running) and its second span the new slot (tentative).
No feature is a percentage uplift: the learner sees indicators, and the effect is whatever it fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

FEATURES = ["promo_confirmed", "promo_tentative", "supply_delay", "ctx_missing"]


@dataclass(frozen=True)
class Item:
    series_id: str | None
    available_at: datetime
    kind: str                     # promotion | supply_constraint | other | unresolved
    spans: tuple[tuple[date, str], ...]   # (Monday, status) for every covered week


def items(reads: list[dict], classes: dict[str, dict]) -> list[Item]:
    """Join the deterministic reading of each message with one classifier's judgments."""
    out = []
    for r in reads:
        c = classes.get(r["message_id"])
        region = r["region"]
        if c is None or c.get("abstained") or c.get("event_type") is None:
            prods = r["products"]
            sid = f"{prods[0]}|{region}" if len(prods) == 1 and region else None
            weeks = tuple((w, "unknown") for a, b in r["spans"] for w in _weeks(a, b))
            out.append(Item(sid, _t(r["available_at"]), "unresolved", weeks))
            continue
        prod = c.get("product") or (r["products"][0] if len(r["products"]) == 1 else None)
        sid = f"{prod}|{region}" if prod and region else None
        kind, status = c["event_type"], c.get("status")
        spans = r["spans"]
        weeks: list[tuple[date, str]] = []
        if kind == "promotion":
            if status == "postponed":
                if spans:
                    weeks += [(w, "postponed") for w in _weeks(*spans[0])]
                if len(spans) >= 2:
                    weeks += [(w, "tentative") for w in _weeks(*spans[1])]
            else:
                weeks += [(w, status or "unknown") for a, b in spans for w in _weeks(a, b)]
        elif kind == "supply_constraint":
            weeks += [(w, "delay") for a, b in spans for w in _weeks(a, b)]
        out.append(Item(sid, _t(r["available_at"]), kind, tuple(weeks)))
    return sorted(out, key=lambda i: i.available_at)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _weeks(a, b) -> list[date]:
    a, b = (date.fromisoformat(a) if isinstance(a, str) else a), (date.fromisoformat(b) if isinstance(b, str) else b)
    return [a + timedelta(weeks=k) for k in range((b - a).days // 7 + 1)]


def value(its: list[Item], series_id: str, week: date, as_of: datetime) -> dict[str, int]:
    promo_status, supply, missing = None, 0, 0
    for i in its:                                    # sorted by available_at: the last promotion match wins
        if i.available_at > as_of:
            break
        if i.series_id != series_id:
            continue
        for w, status in i.spans:
            if w != week:
                continue
            if i.kind == "promotion":
                promo_status = status
            elif i.kind == "supply_constraint":
                supply = 1
            elif i.kind == "unresolved":
                missing = 1
    return {"promo_confirmed": int(promo_status == "confirmed"), "promo_tentative": int(promo_status == "tentative"),
            "supply_delay": supply, "ctx_missing": missing}


def issue_time(week_after_last_sales: date) -> datetime:
    """When a forecast is made: two days into the week after the last sales week (its sales have arrived)."""
    return datetime.combine(week_after_last_sales, datetime.min.time()).replace(tzinfo=timezone.utc) + timedelta(days=2)


def training_frame(its: list[Item], series: list[str], weeks: list[date]) -> pd.DataFrame:
    """Row (s, t) carries what was known when week t was first forecast at horizon 1: the issue time of the
    origin two weeks earlier (last sales week t-2, issued in week t-1)."""
    rows = []
    for s in series:
        for t in weeks:
            rows.append({"unique_id": s, "ds": pd.Timestamp(t), **value(its, s, t, issue_time(t - timedelta(weeks=1)))})
    return pd.DataFrame(rows)


def future_frame(its: list[Item], series: list[str], targets: list[date], origin_last_week: date) -> pd.DataFrame:
    """Rows for the target weeks of one forecast, all as of that forecast's issue time (nothing later)."""
    at = issue_time(origin_last_week + timedelta(weeks=1))
    return pd.DataFrame([{"unique_id": s, "ds": pd.Timestamp(t), **value(its, s, t, at)} for s in series for t in targets])


