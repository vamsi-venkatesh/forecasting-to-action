"""The workbench's "context" job, worker side (code only; no model, no network).

Reads an uploaded context_messages.csv and writes context_reads.json:
  - every message with its own dates, text and what code reads from it (products, region, week spans);
  - label columns (label_*, fixture_label) are DROPPED here: they exist only to score classifiers and must
    never reach one, nor be shown as if they were a finding;
  - a message repeated with the same text and publication time is marked as a duplicate of the first;
  - inclusion at the cutoff (the issue time of the latest forecast, or the job's own time): a message counts
    only if it was available by then. A message that arrived later is listed as excluded, with that reason.
The engine then classifies the included, non-duplicate messages through its one Jev adapter.
Accepts both demo formats: `available_at` or `received_at` as the arrival time, `message_id` or `source_event_id`.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .parse import products, region, spans


def _t(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def run(data: Path, out: Path, params: dict) -> dict:
    path = data / "context_messages.csv"
    if not path.exists():
        raise SystemExit("the context job needs an upload named context_messages.csv")
    cutoff = _t(params["cutoff"]) if params.get("cutoff") else datetime.now(timezone.utc).replace(microsecond=0)
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seen: dict[tuple[str, str], str] = {}
    messages = []
    for r in rows:
        mid = r.get("message_id") or r.get("source_event_id")
        arrived = r.get("available_at") or r.get("received_at")
        text = (r.get("text") or "").strip()
        if not mid or not arrived or not text:
            messages.append({"id": mid, "status": "unreadable", "why": "no id, arrival time or text"})
            continue
        key = (hashlib.sha256(text.encode()).hexdigest(), r.get("published_at") or "")
        dup = seen.get(key)
        seen.setdefault(key, mid)
        at = _t(arrived)
        m = {"id": mid, "published_at": r.get("published_at"), "available_at": at.isoformat(), "text": text,
             "products": products(text), "region": region(text),
             "spans": [(a.isoformat(), b.isoformat()) for a, b in spans(text)]}
        if dup:
            m.update(status="duplicate", why=f"same text and publication time as {dup}")
        elif at > cutoff:
            m.update(status="excluded", why="arrived after the forecast cutoff")
        else:
            m.update(status="included", why="available by the cutoff")
        messages.append(m)
    doc = {"cutoff": cutoff.isoformat(), "rule": "a message counts only if it was available by the cutoff; "
           "duplicates count once", "dropped_columns": sorted(k for k in (rows[0] if rows else {})
                                                           if k.startswith("label_") or k == "fixture_label"),
           "messages": messages}
    (out / "context_reads.json").write_text(json.dumps(doc, indent=1) + "\n")
    counts = {s: sum(m["status"] == s for m in messages) for s in ("included", "excluded", "duplicate", "unreadable")}
    return {"outputs": ["context_reads.json"], "summary": {"cutoff": doc["cutoff"], **counts}}

