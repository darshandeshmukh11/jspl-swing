"""
Multi-factor confluence score (0–100) — how traders combine indicators + sentiment.

Weights inspired by systematic swing desks: trend > structure > momentum > sentiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from jspl_config import JSPLSwingConfig
from sentiment import SentimentBucket


@dataclass
class ConfluenceResult:
    score: int
    trend_pts: int
    momentum_pts: int
    structure_pts: int
    volatility_pts: int
    sentiment_pts: int
    label: str
    reasons: list[str] = field(default_factory=list)


def compute_confluence(
    df: pd.DataFrame,
    sentiment_buckets: dict[str, SentimentBucket],
    cfg: JSPLSwingConfig,
    relative_strength_20d: float | None = None,
) -> ConfluenceResult:
    if df.empty:
        return ConfluenceResult(0, 0, 0, 0, 0, 0, "no data", ["No price data"])

    last = df.iloc[-1]
    close = float(last["Close"])
    reasons: list[str] = []

    # Trend (max 30)
    trend_pts = 0
    ema20 = float(last.get("EMA20", close))
    ema50 = float(last.get("EMA50", close))
    adx = float(last.get("ADX", 0)) if pd.notna(last.get("ADX")) else 0
    st_dir = int(last.get("SUPERTREND_DIR", -1))

    if close > ema20 > ema50:
        trend_pts += 18
        reasons.append("EMA stack bullish (20 > 50)")
    elif close > ema50:
        trend_pts += 10
        reasons.append("Above 50 EMA")
    if adx >= cfg.adx_strong:
        trend_pts += 8
        reasons.append(f"ADX strong ({adx:.0f})")
    elif adx >= cfg.adx_trend_min:
        trend_pts += 4
    if st_dir == 1:
        trend_pts += 4
        reasons.append("Supertrend up")
    trend_pts = min(30, trend_pts)

    # Momentum (max 25)
    mom_pts = 0
    rsi = float(last.get("RSI", 50))
    macd_h = float(last.get("MACD_HIST", 0)) if pd.notna(last.get("MACD_HIST")) else 0
    sk = float(last.get("STOCH_K", 50)) if pd.notna(last.get("STOCH_K")) else 50

    if cfg.dss.rsi_buy_min <= rsi <= cfg.dss.rsi_sell_max:
        mom_pts += 10
        reasons.append(f"RSI in swing band ({rsi:.0f})")
    if macd_h > 0:
        mom_pts += 8
        reasons.append("MACD histogram positive")
    if 20 <= sk <= 80:
        mom_pts += 4
    if bool(last.get("MACD_BULL", False)):
        mom_pts += 3
        reasons.append("MACD bullish cross recent")
    mom_pts = min(25, mom_pts)

    # Structure / zones (max 25)
    struct_pts = 0
    if bool(last.get("IN_BUY_ZONE", False)):
        struct_pts += 12
        reasons.append("Inside buy zone")
    elif close <= float(last.get("BUY_ZONE_HIGH", close * 1.02)):
        struct_pts += 6
    if bool(last.get("BUY_SIGNAL", False)):
        struct_pts += 8
        reasons.append("Buy signal fired")
    vol_r = float(last.get("VOL_RATIO", 1))
    if vol_r >= 1.0:
        struct_pts += 5
    struct_pts = min(25, struct_pts)

    # Volatility / risk context (max 10)
    vol_pts = 0
    bb_pct = float(last.get("BB_PCT_B", 0.5)) if pd.notna(last.get("BB_PCT_B")) else 0.5
    if 0.2 <= bb_pct <= 0.85:
        vol_pts += 6
        reasons.append("BB %B in healthy range")
    atr_pct = float(last.get("ATR", close * 0.02)) / close * 100
    if 1.5 <= atr_pct <= 5.0:
        vol_pts += 4
    vol_pts = min(10, vol_pts)

    # Sentiment + RS (max 10)
    sent_pts = 0
    compounds = [
        b.avg_compound
        for b in sentiment_buckets.values()
        if b.count > 0
    ]
    if compounds:
        avg = sum(compounds) / len(compounds)
        if avg >= 0.15:
            sent_pts += 6
            reasons.append("News sentiment bullish")
        elif avg >= 0:
            sent_pts += 3
        elif avg < -0.12:
            reasons.append("News sentiment bearish — caution")
    if relative_strength_20d is not None and relative_strength_20d > 2:
        sent_pts += 4
        reasons.append("Outperforming metal index (20d)")
    elif relative_strength_20d is not None and relative_strength_20d > -1:
        sent_pts += 2
    sent_pts = min(10, sent_pts)

    total = trend_pts + mom_pts + struct_pts + vol_pts + sent_pts
    if total >= 75:
        label = "High conviction"
    elif total >= 58:
        label = "Actionable swing"
    elif total >= 42:
        label = "Mixed — wait for dip"
    else:
        label = "Low conviction"

    return ConfluenceResult(
        score=total,
        trend_pts=trend_pts,
        momentum_pts=mom_pts,
        structure_pts=struct_pts,
        volatility_pts=vol_pts,
        sentiment_pts=sent_pts,
        label=label,
        reasons=reasons,
    )
