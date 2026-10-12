"""Deterministic reading of a context message: which products and region it names, and which weeks.

Code, not a model: products and regions are a closed vocabulary; week spans are read from the three written
forms the demo inbox uses. The semantic part (what kind of event, which status, which of two named products
it is about) is left to a classifier: rules (below), a conventional LLM, or Jev, compared in the experiment.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from .fixture import PRODUCTS, REGIONS

_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}
_SPAN = re.compile(r"KW (\d{1,2}) and KW (\d{1,2}) \((\d{4})\)|weeks (\d{1,2})-(\d{1,2}) of (\d{4})"
                   r"|two weeks from Monday (\d{1,2}) ([A-Za-z]+) (\d{4})")


def _monday(year: int, week: int) -> date:
    return date.fromisocalendar(year, week, 1)


def spans(text: str) -> list[tuple[date, date]]:
    """Week spans in order of appearance, as (first Monday, last Monday)."""
    out = []
    for m in _SPAN.finditer(text):
        if m.group(1):
            y = int(m.group(3))
            out.append((_monday(y, int(m.group(1))), _monday(y, int(m.group(2)))))
        elif m.group(4):
            y = int(m.group(6))
            out.append((_monday(y, int(m.group(4))), _monday(y, int(m.group(5)))))
        elif m.group(7):
            d = date(int(m.group(9)), _MONTHS[m.group(8).lower()], int(m.group(7)))
            out.append((d, d + timedelta(weeks=1)))
    return out


def products(text: str) -> list[str]:
    low = text.lower()
    found = sorted(((low.find(name.lower()), pid) for pid, name in PRODUCTS.items() if name.lower() in low))
    return [pid for _, pid in found]


def region(text: str) -> str | None:
    low = text.lower()
    hits = {code for code, names in REGIONS.items() for n in names if n.lower() in low}
    # the codes themselves ("DE-S"), as whole upper-case tokens only: "at" in a sentence is never Austria
    hits |= {code for code in REGIONS if re.search(rf"(?<![A-Za-z0-9-]){re.escape(code)}(?![A-Za-z0-9-])", text)}
    return hits.pop() if len(hits) == 1 else None


def read(message: dict) -> dict:
    """The deterministic facts of one message. Nothing here is a judgment."""
    return {"message_id": message["message_id"], "available_at": message["available_at"],
            "products": products(message["text"]), "region": region(message["text"]),
            "spans": [(a.isoformat(), b.isoformat()) for a, b in spans(message["text"])]}


# ------------------------------------------------------------------ the rule-derived baseline classifier
_PROMO = re.compile(r"\b(promotion|promo|campaign|discount|promote)\b", re.I)
_SUPPLY = re.compile(r"\b(supplier|inbound|delivery|ship|logistics)\b", re.I)
_POSTPONED = re.compile(r"\b(postpone\w*|moving|shift|slips|moved|new slot)\b", re.I)
_CANCELLED = re.compile(r"\b(cancel\w*|dropping|scrapped|pulled out|will not happen|it is off|after all)\b", re.I)
_TENTATIVE = re.compile(r"\b(not confirmed|nothing|yet|pending|might|possible|idea|tentative\w*|considering|draft)\b", re.I)
_CONFIRMED = re.compile(r"\b(confirm\w*|sign-off|signed|approved|locked in|going ahead|is on)\b", re.I)


def rule_classify(text: str) -> dict:
    """Keyword rules, written once before any evaluation. Known weakness, kept on purpose: no negation handling
    beyond the order of the checks (the comparison is what the experiment is for)."""
    if _PROMO.search(text):
        event = "promotion"
    elif _SUPPLY.search(text):
        event = "supply_constraint"
    else:
        event = "other"
    status = "not_a_promotion"
    if event == "promotion":
        if len(spans(text)) >= 2 or _POSTPONED.search(text):
            status = "postponed"
        elif _CANCELLED.search(text):
            status = "cancelled"
        elif _TENTATIVE.search(text):
            status = "tentative"
        elif _CONFIRMED.search(text):
            status = "confirmed"
        else:
            status = "tentative"
    found = products(text)
    return {"event_type": event, "status": status, "product": found[0] if found else None, "abstained": False}


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s)

