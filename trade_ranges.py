"""Buy/sell initiation ranges and next-session probable range for risk planning."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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


@dataclass
class NextSessionRange:
    """Probable high/low for the next NSE session — stops & targets for risk."""

    anchor_price: float
    probable_low: float
    probable_high: float
    core_low: float
    core_high: float
    range_width_inr: float
    range_width_pct: float
    avg_daily_range: float
    atr: float
    stop_tight: float
    stop_wide: float
    pivot: float
    s1: float
    s2: float
    r1: float
    r2: float
    target_1: float
    target_2: float
    risk_note: str
    target_note: str


@dataclass
class SessionTradeContext:
    initiation: TradeInitiationGuide
    next_session: NextSessionRange
    data_label: str = ""


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


def _avg_daily_range(df: pd.DataFrame, lookback: int = 20) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    recent = df.tail(lookback)
    spans = (recent["High"].astype(float) - recent["Low"].astype(float)).dropna()
    if spans.empty:
        return 0.0
    return float(spans.mean())


def build_next_session_range(
    plan: SessionPlan,
    df: pd.DataFrame,
    live_price: float | None = None,
) -> NextSessionRange:
    """
    Estimate next-session trading range from ATR, recent daily range, and floor pivots.

    Probable range = intersection of pivot envelope (S2–R2) and volatility band around anchor.
    Core range = tighter ±0.45×ATR band (typical intraday mean reversion span).
    """
    anchor = float(live_price if live_price is not None else plan.close)
    atr = float(plan.atr)
    avg_dr = _avg_daily_range(df, lookback=20) or atr * 1.15
    half_dr = avg_dr * 0.5
    half_atr = atr * 0.5

    est_low = anchor - max(half_atr, half_dr * 0.92)
    est_high = anchor + max(half_atr, half_dr * 0.92)

    probable_low = round(min(est_low, plan.s2, plan.buy_zone_low), 2)
    probable_high = round(max(est_high, plan.r2, plan.sell_zone_high), 2)
    if probable_low > probable_high:
        probable_low, probable_high = round(min(est_low, plan.s2), 2), round(max(est_high, plan.r2), 2)

    core_low = round(max(probable_low, anchor - 0.45 * atr), 2)
    core_high = round(min(probable_high, anchor + 0.45 * atr), 2)
    if core_low >= core_high:
        core_low = round(anchor - 0.35 * atr, 2)
        core_high = round(anchor + 0.35 * atr, 2)

    width = probable_high - probable_low
    width_pct = (width / anchor * 100.0) if anchor else 0.0

    risk_note = (
        f"For a <b>long</b> swing: place stop below <b>₹{plan.stop_long:,.2f}</b> (tight) or "
        f"<b>₹{plan.stop_wide:,.2f}</b> (wide / S2 area). "
        f"If price breaks below <b>₹{probable_low:,.2f}</b>, the probable range is invalid — reduce size or exit."
    )
    target_note = (
        f"Book partials toward <b>₹{plan.target_1:,.2f}</b> (R1 / T1) and <b>₹{plan.target_2:,.2f}</b> (R2 / T2). "
        f"Sell zone <b>₹{plan.sell_zone_low:,.2f}–₹{plan.sell_zone_high:,.2f}</b> aligns with profit-taking inside the range."
    )

    return NextSessionRange(
        anchor_price=round(anchor, 2),
        probable_low=probable_low,
        probable_high=probable_high,
        core_low=core_low,
        core_high=core_high,
        range_width_inr=round(width, 2),
        range_width_pct=round(width_pct, 2),
        avg_daily_range=round(avg_dr, 2),
        atr=round(atr, 2),
        stop_tight=plan.stop_long,
        stop_wide=plan.stop_wide,
        pivot=plan.pivot,
        s1=plan.s1,
        s2=plan.s2,
        r1=plan.r1,
        r2=plan.r2,
        target_1=plan.target_1,
        target_2=plan.target_2,
        risk_note=risk_note,
        target_note=target_note,
    )


def build_session_trade_context(
    plan: SessionPlan,
    df: pd.DataFrame,
    live_price: float | None = None,
    *,
    data_label: str = "",
) -> SessionTradeContext:
    label = data_label or plan.data_label or (
        f"Reference **₹{plan.close:,.2f}** · bar **{plan.as_of}**"
    )
    return SessionTradeContext(
        initiation=build_trade_initiation_guide(plan, live_price),
        next_session=build_next_session_range(plan, df, live_price),
        data_label=label,
    )
