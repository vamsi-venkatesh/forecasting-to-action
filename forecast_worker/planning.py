"""Planning arithmetic: from an issued forecast to a proposed replenishment (brief §8). Code only, no model.

This is a stated TEST POLICY, not an optimal inventory formula. Units must match (the caller converts); money
is Decimal. The trajectory is week by week, so a receipt that arrives after a shortage does not hide it.

For a series and a protection window of weeks w1..wN (each with a forecast requirement):
  available(t)  = usable_on_hand - commitments + confirmed receipts arriving by the end of week t
  need(t)       = cumulative requirement through week t + buffer
  gap(t)        = need(t) - available(t)
  shortfall     = max(0, max_t gap(t))   the quantity that must arrive before the first week it is needed
  first_short   = the first week with gap(t) > 0
  proposal      = shortfall rounded UP to the pack size, then raised to the minimum order (0 stays 0)
  value         = proposal * unit cost (Decimal), or unknown when no price is recorded (the proposal is then
                  QUALIFIED, never priced by guess)
Unconfirmed receipts never count, and receipts without a date never count; both are listed. A confirmed
receipt dated after the protection window does not count and is listed as late.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_CEILING, Decimal


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    quantity: Decimal
    expected_date: date | None
    confirmed: bool


@dataclass(frozen=True)
class Rule:
    buffer_units: Decimal
    pack_size: Decimal
    min_order_units: Decimal
    unit_cost: Decimal | None
    currency: str = "EUR"


@dataclass
class Plan:
    series_id: str
    weeks: list[str]
    trajectory: list[dict]
    shortfall: Decimal
    first_short_week: str | None
    proposed_quantity: Decimal
    proposed_value: Decimal | None
    currency: str
    qualified: list[str] = field(default_factory=list)
    excluded_receipts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"series_id": self.series_id, "weeks": self.weeks, "trajectory": self.trajectory,
                "shortfall": str(self.shortfall), "first_short_week": self.first_short_week,
                "proposed_quantity": str(self.proposed_quantity),
                "proposed_value": None if self.proposed_value is None else str(self.proposed_value),
                "currency": self.currency, "qualified": self.qualified, "excluded_receipts": self.excluded_receipts}


def _round_up(q: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return q
    return (q / step).to_integral_value(rounding=ROUND_CEILING) * step


def plan(series_id: str, weeks: list[date], requirement: list[Decimal], *, usable_on_hand: Decimal,
         commitments: Decimal, receipts: list[Receipt], rule: Rule) -> Plan:
    if len(weeks) != len(requirement) or not weeks:
        raise ValueError("one requirement per week of the protection window")
    if any(r < 0 for r in requirement):
        raise ValueError("a negative requirement is not a requirement")
    week_end = [w.toordinal() + 6 for w in weeks]
    qualified, excluded = [], []
    counted: list[tuple[int, Decimal]] = []
    for r in receipts:
        if not r.confirmed:
            excluded.append({"receipt_id": r.receipt_id, "why": "unconfirmed"})
        elif r.expected_date is None:
            excluded.append({"receipt_id": r.receipt_id, "why": "no expected date"})
        elif r.expected_date.toordinal() > week_end[-1]:
            excluded.append({"receipt_id": r.receipt_id, "why": "arrives after the protection window (late)"})
        else:
            counted.append((r.expected_date.toordinal(), r.quantity))
    if excluded:
        qualified.append(f"{len(excluded)} receipt(s) not counted: " + ", ".join(f"{e['receipt_id']} ({e['why']})" for e in excluded))
    trajectory, cum, shortfall, first = [], Decimal(0), Decimal(0), None
    for i, w in enumerate(weeks):
        cum += requirement[i]
        arrived = sum((q for d, q in counted if d <= week_end[i]), Decimal(0))
        available = usable_on_hand - commitments + arrived
        need = cum + rule.buffer_units
        gap = need - available
        if gap > 0 and first is None:
            first = w.isoformat()
        shortfall = max(shortfall, gap)
        trajectory.append({"week": w.isoformat(), "requirement": str(requirement[i]), "cumulative_requirement": str(cum),
                           "receipts_arrived": str(arrived), "available": str(available), "need": str(need), "gap": str(gap)})
    shortfall = max(shortfall, Decimal(0))
    qty = Decimal(0)
    if shortfall > 0:
        qty = max(_round_up(shortfall, rule.pack_size), rule.min_order_units)
        qty = _round_up(qty, rule.pack_size)           # the minimum itself is ordered in whole packs
    value = None
    if qty > 0:
        if rule.unit_cost is None:
            qualified.append("no unit price recorded: the value is unknown and must be priced by a person")
        else:
            value = (qty * rule.unit_cost).quantize(Decimal("0.01"))
    return Plan(series_id, [w.isoformat() for w in weeks], trajectory, shortfall, first, qty, value, rule.currency,
                qualified, excluded)

