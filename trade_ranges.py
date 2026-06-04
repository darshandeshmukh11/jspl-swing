"""Buy/sell initiation ranges and plain-language prompts for the next session."""

from __future__ import annotations

from dataclasses import dataclass

from session_plan import SessionPlan


@dataclass
class TradeInitiationGuide:
    """Actionable buy/sell ranges relative to live or last close."""

    reference_price: float
    buy_low: float
    buy_high: float
    sell_low: float
    sell_high: float
    entry_low: float
    entry_high: float
    stop: float
    target_1: float
    target_2: float
    buy_status: str  # IN_ZONE | BELOW_ZONE | ABOVE_ZONE
    sell_status: str  # IN_ZONE | BELOW_ZONE | ABOVE_ZONE
    buy_prompt: str
    sell_prompt: str
    buy_action: str  # INITIATE | PREPARE | WAIT
    sell_action: str  # INITIATE | PREPARE | WAIT


def _zone_status(price: float, low: float, high: float) -> str:
    if low > high:
        low, high = high, low
    if price < low:
        return "BELOW_ZONE"
    if price > high:
        return "ABOVE_ZONE"
    return "IN_ZONE"


def build_trade_initiation_guide(
    plan: SessionPlan,
    live_price: float | None = None,
) -> TradeInitiationGuide:
    price = float(live_price if live_price is not None else plan.close)
    buy_lo, buy_hi = plan.buy_zone_low, plan.buy_zone_high
    sell_lo, sell_hi = plan.sell_zone_low, plan.sell_zone_high
    entry_lo, entry_hi = plan.entry_low, plan.entry_high

    buy_status = _zone_status(price, buy_lo, buy_hi)
    sell_status = _zone_status(price, sell_lo, sell_hi)

    if buy_status == "IN_ZONE":
        buy_action = "INITIATE"
        buy_prompt = (
            f"<b>Initiate BUY</b> in ₹{buy_lo:,.2f}–₹{buy_hi:,.2f} — price is inside the buy zone now. "
            f"Prefer limit near ₹{entry_lo:,.2f}–₹{entry_hi:,.2f}; stop below ₹{plan.stop_long:,.2f}."
        )
    elif buy_status == "BELOW_ZONE":
        buy_action = "PREPARE"
        buy_prompt = (
            f"<b>Prepare BUY</b> — place limits in ₹{buy_lo:,.2f}–₹{buy_hi:,.2f} "
            f"(price ₹{price:,.2f} is below zone). "
            f"Entry band ₹{entry_lo:,.2f}–₹{entry_hi:,.2f}; stop ₹{plan.stop_long:,.2f}."
        )
    else:
        buy_action = "WAIT"
        buy_prompt = (
            f"<b>Wait to BUY</b> — do not chase. Initiate only on pullback to ₹{buy_lo:,.2f}–₹{buy_hi:,.2f} "
            f"(price ₹{price:,.2f} is above zone)."
        )

    if sell_status == "IN_ZONE":
        sell_action = "INITIATE"
        sell_prompt = (
            f"<b>Initiate SELL / book profit</b> in ₹{sell_lo:,.2f}–₹{sell_hi:,.2f} — price is in the sell zone. "
            f"Targets ₹{plan.target_1:,.2f} (T1) · ₹{plan.target_2:,.2f} (T2)."
        )
    elif sell_status == "BELOW_ZONE":
        sell_action = "WAIT"
        sell_prompt = (
            f"<b>Hold for targets</b> — sell zone starts at ₹{sell_lo:,.2f}. "
            f"Book partials toward ₹{plan.target_1:,.2f} / ₹{plan.target_2:,.2f}."
        )
    else:
        sell_action = "PREPARE"
        sell_prompt = (
            f"<b>Consider trimming</b> — price ₹{price:,.2f} is above sell zone "
            f"₹{sell_lo:,.2f}–₹{sell_hi:,.2f}. Trail stop or scale out; stretch target ₹{plan.target_2:,.2f}."
        )

    return TradeInitiationGuide(
        reference_price=round(price, 2),
        buy_low=buy_lo,
        buy_high=buy_hi,
        sell_low=sell_lo,
        sell_high=sell_hi,
        entry_low=entry_lo,
        entry_high=entry_hi,
        stop=plan.stop_long,
        target_1=plan.target_1,
        target_2=plan.target_2,
        buy_status=buy_status,
        sell_status=sell_status,
        buy_prompt=buy_prompt,
        sell_prompt=sell_prompt,
        buy_action=buy_action,
        sell_action=sell_action,
    )
