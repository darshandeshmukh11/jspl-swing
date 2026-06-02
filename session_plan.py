"""Next-session trade plan — zones, pivots, ATR stops, triggers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from jspl_config import JSPLSwingConfig
from zones import latest_zones


@dataclass
class SessionPlan:
    as_of: str
    close: float
    atr: float
    atr_pct: float
    buy_zone_low: float
    buy_zone_high: float
    sell_zone_low: float
    sell_zone_high: float
    pivot: float
    s1: float
    s2: float
    r1: float
    r2: float
    stop_long: float
    stop_wide: float
    target_1: float
    target_2: float
    entry_low: float
    entry_high: float
    triggers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_session_plan(df: pd.DataFrame, cfg: JSPLSwingConfig) -> SessionPlan:
    if df.empty:
        raise ValueError("No OHLCV data")

    last = df.iloc[-1]
    close = float(last["Close"])
    atr = float(last["ATR"]) if pd.notna(last.get("ATR")) else close * 0.015
    zones = latest_zones(df)

    piv = {
        "PIVOT": float(last.get("PIVOT", close)),
        "S1": float(last.get("S1", zones.get("support", close * 0.97))),
        "S2": float(last.get("S2", zones.get("support", close * 0.95) * 0.99)),
        "R1": float(last.get("R1", zones.get("resistance", close * 1.03))),
        "R2": float(last.get("R2", zones.get("resistance", close * 1.05) * 1.01)),
    }

    buy_lo = zones.get("buy_low", close * 0.98)
    buy_hi = zones.get("buy_high", close * 1.0)
    sell_lo = zones.get("sell_low", close * 1.02)
    sell_hi = zones.get("sell_high", close * 1.04)

    stop_tight = min(buy_lo - 0.25 * atr, piv["S1"] - 0.1 * atr, close - cfg.dss.stop_atr_mult * atr)
    stop_wide = min(piv["S2"] - 0.15 * atr, stop_tight - 0.5 * atr)
    target_1 = max(sell_lo, piv["R1"])
    target_2 = max(sell_hi, piv["R2"])

    entry_low = round(min(buy_lo, piv["S1"], float(last.get("EMA20", buy_lo))), 2)
    entry_high = round(max(buy_hi, piv["PIVOT"] * 0.998), 2)

    triggers: list[str] = []
    warnings: list[str] = []

    if bool(last.get("BUY_SIGNAL", False)):
        triggers.append("EOD buy signal active — trend + dip zone confluence")
    if close <= buy_hi and close >= buy_lo:
        triggers.append(f"Price inside buy zone ₹{buy_lo:,.2f}–₹{buy_hi:,.2f}")
    if float(last.get("RSI", 50)) <= cfg.rsi_oversold + 5:
        triggers.append(f"RSI supportive ({float(last['RSI']):.1f})")
    if float(last.get("MACD_HIST", 0)) > 0 and float(last.get("MACD_HIST", 0)) > float(df.iloc[-2].get("MACD_HIST", 0)):
        triggers.append("MACD histogram rising (momentum)")
    if float(last.get("ADX", 0)) >= cfg.adx_trend_min:
        triggers.append(f"ADX ≥ {cfg.adx_trend_min} — trend strength OK")
    if int(last.get("SUPERTREND_DIR", -1)) == 1:
        triggers.append("Supertrend bullish")

    if close < float(last.get("EMA50", close)):
        warnings.append("Below 50 EMA — weak swing long bias")
    if float(last.get("RSI", 50)) >= cfg.rsi_overbought:
        warnings.append("RSI overbought — avoid chasing")
    if float(last.get("BB_PCT_B", 0.5)) > 1.0:
        warnings.append("Above upper Bollinger — extended move")

    as_of = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")

    return SessionPlan(
        as_of=as_of,
        close=round(close, 2),
        atr=round(atr, 2),
        atr_pct=round(atr / close * 100, 2),
        buy_zone_low=round(buy_lo, 2),
        buy_zone_high=round(buy_hi, 2),
        sell_zone_low=round(sell_lo, 2),
        sell_zone_high=round(sell_hi, 2),
        pivot=round(piv["PIVOT"], 2),
        s1=round(piv["S1"], 2),
        s2=round(piv["S2"], 2),
        r1=round(piv["R1"], 2),
        r2=round(piv["R2"], 2),
        stop_long=round(stop_tight, 2),
        stop_wide=round(stop_wide, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        entry_low=entry_low,
        entry_high=entry_high,
        triggers=triggers,
        warnings=warnings,
    )
