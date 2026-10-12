"""Supplier messages about deliveries (the 'signals' job): CODE reads what can be read exactly. Which confirmed
receipt a message names (the receipt id, from receipts.csv only), its planned date, any new date written in the
message, and the shift in whole weeks. What the message MEANS (a confirmed delay, a possible one, on time, or
nothing about timing) is left to Jev, asked by the engine after this job. A message received after the forecast's
issue time is not used. Nothing here invents a date or a shift that the message does not state.
"""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

ISO_DAY = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def run(data: Path, out: Path, params: dict) -> dict:
    with (data / "receipts.csv").open(newline="") as fh:
        receipts = {r["receipt_id"]: r for r in csv.DictReader(fh) if r.get("status") == "confirmed" and r.get("expected_date")}
    with (data / "supplier_messages.csv").open(newline="", encoding="utf-8") as fh:
        messages = list(csv.DictReader(fh))[:20]          # bounded: a sample never sends more than 20 messages out
    cutoff = datetime.fromisoformat(params["cutoff"]) if params.get("cutoff") else None
    ids = sorted(receipts, key=len, reverse=True)
    reads = []
    for m in messages:
        text = (m.get("text") or "")[:1000]
        rid = next((i for i in ids if i in text), None)
        entry = {"message_id": m.get("message_id"), "received_at": m.get("received_at"), "text": text,
                 "label": m.get("label"), "receipt_id": rid}
        if cutoff and m.get("received_at") and datetime.fromisoformat(m["received_at"]) > cutoff:
            entry.update(status="after_cutoff", why="received after the forecast was issued")
        elif rid is None:
            entry.update(status="no_receipt", why="names no confirmed receipt of this project")
        else:
            r = receipts[rid]
            old = date.fromisoformat(r["expected_date"])
            dates = [date.fromisoformat(d) for d in ISO_DAY.findall(text)]
            new = next((d for d in dates if d != old), None)
            shift = math.ceil((new - old).days / 7) if new and new > old else None
            entry.update(status="read", series_id=r["series_id"], quantity=r["quantity"], expected_date=old.isoformat(),
                         new_date=new.isoformat() if new else None, shift_weeks=shift)
        reads.append(entry)
    (out / "signals_read.json").write_text(json.dumps({"messages": reads, "cutoff": params.get("cutoff")}, indent=1) + "\n")
    return {"outputs": ["signals_read.json"],
            "summary": {"messages": len(reads), "naming_a_receipt": sum(e["status"] == "read" for e in reads)}}

