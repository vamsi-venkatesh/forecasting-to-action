"""NordWerk Context Demo: a FORWARD-ONLY synthetic fixture for the context-feature experiment (brief 7A).

What it is: 12 fictional series (4 products x 3 regions) with planned promotions whose status changes over
time (tentative -> confirmed / cancelled / postponed), distributor rumours, supplier delays and irrelevant
messages, written as dated business messages. The messages are generated from the EVENT PLAN, before the
event and never from sales values. A promotion that actually runs adds demand in its two weeks; nothing else
does. A supplier delay is a supply constraint and has no demand effect here (kept as a separate feature).

What it can show: whether the pipeline (parse -> classify -> as-of aggregate -> learner) works, and how much
a classifier's mistakes cost against a label ceiling. What it CANNOT show: real business uplift. The effect
size (UPLIFT) is chosen here, so any forecasting gain is a property of this fixture, not evidence that
context helps a real client.

    python -m forecast_worker.context.fixture OUT_DIR [--seed 20260928]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


FIXTURE_VERSION = "nordwerk-context-1"
PRODUCTS = {"P-100": "Hydraulic filter", "P-200": "Seal kit", "P-300": "Pump housing", "P-400": "Valve block"}
REGIONS = {"DE-N": ["Northern Germany", "the north region", "the Hamburg area"],
           "DE-S": ["Southern Germany", "Bavaria", "the south region"],
           "AT": ["Austria", "our Austrian customers", "the Vienna area"]}
WEEKS = 156
START = date(2023, 10, 2)                  # a Monday; ISO weeks
UPLIFT = (0.35, 0.55)                      # share of the weekly mean added in each week a promotion runs
PROMOS_PER_SERIES = 9

CONFIRMED = [
    "Marketing has confirmed the {p} promotion for {r} {span}.",
    "Final sign-off received: the {p} campaign in {r} runs {span}.",
    "The {p} promo for {r} {span} is going ahead, not cancelled after all.",
    "Budget approved, so the {p} discount for {r} is on, {span}.",
    "Go for the {p} promotion in {r} {span}; the retailer has signed.",
    "Just to close the loop: {p} promotion in {r}, {span}, is locked in.",
]
TENTATIVE = [
    "Marketing is considering a {p} promotion for {r} {span}; nothing is signed off yet.",
    "Draft plan: possible {p} campaign in {r} {span}, pending budget.",
    "We might run a {p} discount in {r} {span} if the retailer agrees.",
    "Idea from sales: promote {p} in {r} {span}. Not confirmed.",
    "Tentatively pencilled in: {p} promotion, {r}, {span}.",
    "Is anyone against a {p} promo in {r} {span}? Nothing decided.",
]
CANCELLED = [
    "The {p} promotion for {r} {span} is cancelled.",
    "We are dropping the {p} campaign in {r} {span}; budget was cut.",
    "No {p} promo in {r} {span} after all.",
    "Retailer pulled out, so the {p} promotion for {r} {span} will not happen.",
    "Please remove the {p} campaign ({r}, {span}) from the plan, it is off.",
    "The {p} discount in {r} {span} will not be confirmed; it has been scrapped.",
]
POSTPONED = [
    "The {p} promotion for {r} planned {span} is postponed to {span2}.",
    "Moving the {p} campaign in {r} from {span} to {span2}.",
    "{P} promo ({r}): not {span} any more, now {span2}, still to be signed off.",
    "Retailer asked to shift the {p} promotion in {r} {span}; new slot {span2}.",
    "Delay: the {p} discount for {r} slips from {span} to {span2}.",
]
RUMOUR = [
    "A distributor mentioned a possible large order for {p} in {r}, nothing confirmed.",
    "Heard at the trade fair that a competitor may stop selling {p} in {r}. Unverified.",
    "Customer in {r} asked about {p} prices for next year; just an enquiry.",
]
SUPPLY = [
    "Supplier notice: the inbound {p} batch for {r} is delayed, now expected {span}.",
    "Logistics: {p} delivery to the {r} warehouse slips to {span} because of a port strike.",
    "Our supplier cannot ship {p} for {r} before {span}; demand forecasts are unchanged.",
]
IRRELEVANT = [
    "Office closure for {r} on Friday because of the team event.",
    "Reminder: the {p} product brochure has a typo on page 3, please do not reprint yet.",
    "New parking rules at our site serving {r} from next month.",
    "The {p} datasheet was translated into Czech; files are on the shared drive.",
]
DECOY = "Unlike the {d} campaign last spring, "


def week_start(i: int) -> date:
    return START + timedelta(weeks=i)


def span_text(i: int, rng) -> str:
    """Two weeks starting at week index i, in one of three written forms (the parser reads all three)."""
    a, b = week_start(i).isocalendar(), week_start(i + 1).isocalendar()
    form = int(rng.integers(0, 3))
    if form == 0 and a.year == b.year:
        return f"in KW {a.week} and KW {b.week} ({a.year})"
    if form == 1 and a.year == b.year:
        return f"in weeks {a.week}-{b.week} of {a.year}"
    return f"for two weeks from Monday {week_start(i).day} {week_start(i).strftime('%B %Y')}"


def _stamp(week: int, rng) -> tuple[str, str]:
    pub = datetime.combine(week_start(week), datetime.min.time(), tzinfo=timezone.utc) \
        + timedelta(days=int(rng.integers(0, 5)), hours=int(rng.integers(7, 18)))
    return pub.isoformat(), (pub + timedelta(hours=int(rng.integers(1, 20)))).isoformat()


def build(out: Path, seed: int = 20260928) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    series = [(p, r) for p in PRODUCTS for r in REGIONS]
    sales, messages, events = [], [], []

    def say(week: int, template: str, pid: str, region: str, labels: dict, span: str = "", span2: str = "",
            decoy: bool = False) -> None:
        pub, recv = _stamp(week, rng)
        r_text = REGIONS[region][int(rng.integers(0, 3))]
        text = template.format(p=PRODUCTS[pid], P=PRODUCTS[pid], r=r_text, span=span, span2=span2)
        if decoy:
            other = [q for q in PRODUCTS if q != pid][int(rng.integers(0, 3))]
            text = DECOY.format(d=PRODUCTS[other]) + text[0].lower() + text[1:]
        messages.append({"message_id": f"M-{len(messages):04d}", "published_at": pub, "available_at": recv,
                         "channel": "demo-inbox", "text": text,
                         "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         # labels are fixed by construction; they are never shown to a classifier
                         "label_event_type": labels["event_type"], "label_status": labels.get("status", "not_a_promotion"),
                         "label_product": pid, "label_region": region})

    for pid, region in series:
        sid = f"{pid}|{region}"
        level = rng.uniform(30, 150)
        amp = rng.uniform(0.05, 0.25) * level
        trend = rng.uniform(-0.03, 0.08)
        mean = np.array([max(1.0, level + trend * w + amp * np.sin(2 * np.pi * (w % 52) / 52)) for w in range(WEEKS)])
        running = np.zeros(WEEKS)
        starts, w = [], int(rng.integers(12, 20))
        while len(starts) < PROMOS_PER_SERIES and w < WEEKS - 4:
            starts.append(w)
            w += int(rng.integers(11, 17))
        for e in starts:
            decoy = rng.random() < 0.2
            announce = e - int(rng.integers(5, 9))
            say(announce, TENTATIVE[int(rng.integers(0, len(TENTATIVE)))], pid, region,
                {"event_type": "promotion", "status": "tentative"}, span=span_text(e, rng), decoy=decoy)
            fate, ran_at = rng.random(), None
            if fate < 0.55:                                     # confirmed and runs
                say(e - int(rng.integers(2, 4)), CONFIRMED[int(rng.integers(0, len(CONFIRMED)))], pid, region,
                    {"event_type": "promotion", "status": "confirmed"}, span=span_text(e, rng))
                ran_at = e
            elif fate < 0.72:                                   # cancelled
                say(e - int(rng.integers(2, 4)), CANCELLED[int(rng.integers(0, len(CANCELLED)))], pid, region,
                    {"event_type": "promotion", "status": "cancelled"}, span=span_text(e, rng))
            elif fate < 0.88:                                   # postponed, then usually confirmed
                e2 = e + int(rng.integers(4, 8))
                if e2 < WEEKS - 2:
                    say(e - int(rng.integers(2, 4)), POSTPONED[int(rng.integers(0, len(POSTPONED)))], pid, region,
                        {"event_type": "promotion", "status": "postponed"}, span=span_text(e, rng), span2=span_text(e2, rng))
                    if rng.random() < 0.8:
                        say(e2 - 2, CONFIRMED[int(rng.integers(0, len(CONFIRMED)))], pid, region,
                            {"event_type": "promotion", "status": "confirmed"}, span=span_text(e2, rng))
                        ran_at = e2
            else:                                               # no follow-up mail; runs half the time
                ran_at = e if rng.random() < 0.5 else None
            if ran_at is not None:
                for k in (0, 1):
                    running[ran_at + k] = rng.uniform(*UPLIFT)
            events.append({"series_id": sid, "planned_start": e, "ran_start": ran_at})
        for _ in range(2):
            say(int(rng.integers(8, WEEKS - 2)), RUMOUR[int(rng.integers(0, len(RUMOUR)))], pid, region,
                {"event_type": "other"})
        for _ in range(3):
            d = int(rng.integers(10, WEEKS - 4))
            say(d, SUPPLY[int(rng.integers(0, len(SUPPLY)))], pid, region, {"event_type": "supply_constraint"},
                span=span_text(d + int(rng.integers(1, 4)), rng))
        for _ in range(2):
            say(int(rng.integers(8, WEEKS - 2)), IRRELEVANT[int(rng.integers(0, len(IRRELEVANT)))], pid, region,
                {"event_type": "other"})
        for w in range(WEEKS):
            qty = int(rng.poisson(mean[w] * (1 + running[w])))
            wk = week_start(w)
            avail = datetime.combine(wk, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=7 + int(rng.integers(0, 3)))
            sales.append({"series_id": sid, "product_id": pid, "region_id": region, "week_start": wk.isoformat(),
                          "available_at": avail.isoformat(), "quantity": qty, "unit": "pcs",
                          "promo_ran": int(running[w] > 0)})       # truth column, for scoring by week type only
    messages.sort(key=lambda m: m["available_at"])
    files = {"sales.csv": sales, "context_messages.csv": messages}
    manifest = {"fixture": FIXTURE_VERSION, "fictional": True, "seed": seed, "weeks": WEEKS, "first_week": START.isoformat(),
                "series": len(series), "uplift_range": list(UPLIFT),
                "note": "Synthetic and forward-only. Messages come from the event plan, never from sales. Any forecasting "
                        "gain measured on it is a property of the chosen effect size, not evidence of real uplift.",
                "promotions": {"planned": len(events), "ran": sum(e["ran_start"] is not None for e in events)},
                "messages": len(messages), "files": {}}
    for name, rows in files.items():
        path = out / name
        with path.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        manifest["files"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20260928)
    a = ap.parse_args()
    m = build(Path(a.out), a.seed)
    print(json.dumps({k: m[k] for k in ("fixture", "series", "promotions", "messages")}))

