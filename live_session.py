"""Merge live LTP into daily OHLCV for realtime buy/sell zones."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from market_live import LiveQuote
from zones import add_zones

IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> datetime:
    return datetime.now(IST)


def _today_ts() -> pd.Timestamp:
    return pd.Timestamp(ist_now().date())


def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(IST)
    else:
        idx = idx.tz_localize(IST)
    out.index = idx.normalize().tz_localize(None)
    return out.sort_index()


def merge_live_into_daily(
    df: pd.DataFrame,
    live_price: float,
    *,
    day_high: Optional[float] = None,
    day_low: Optional[float] = None,
) -> pd.DataFrame:
    """Patch or append today's bar with live LTP (and optional session H/L)."""
    out = normalize_daily_index(df)
    if out.empty:
        return out

    live_price = float(live_price)
    today_ts = _today_ts()
    hi = float(day_high) if day_high and day_high > 0 else live_price
    low = float(day_low) if day_low and day_low > 0 else live_price

    if out.index[-1] >= today_ts:
        o = float(out["Open"].iloc[-1])
        h = max(float(out["High"].iloc[-1]), hi, live_price)
        low_val = min(float(out["Low"].iloc[-1]), low, live_price)
        vol = float(out["Volume"].iloc[-1]) if "Volume" in out.columns else 0.0
        out.iloc[-1, out.columns.get_loc("Open")] = o
        out.iloc[-1, out.columns.get_loc("High")] = h
        out.iloc[-1, out.columns.get_loc("Low")] = low_val
        out.iloc[-1, out.columns.get_loc("Close")] = live_price
        if "Volume" in out.columns:
            out.iloc[-1, out.columns.get_loc("Volume")] = vol
    else:
        prev_close = float(out["Close"].iloc[-1])
        out.loc[today_ts] = {
            "Open": prev_close,
            "High": max(prev_close, hi, live_price),
            "Low": min(prev_close, low, live_price),
            "Close": live_price,
            "Volume": 0.0,
        }
    return out


def apply_live_for_trading(
    df: pd.DataFrame,
    dss_cfg,
    stock: Optional[LiveQuote],
    *,
    use_live: bool = True,
) -> tuple[pd.DataFrame, str, Optional[float], float, str]:
    """
    Return frame with refreshed buy/sell zones and a UI label.

    When ``use_live`` and a valid quote: last bar uses live LTP and zones are recomputed.
    Otherwise: last row as-is (typically last completed EOD bar in the download).
    """
    if df.empty:
        return df, "No data", None, 0.0, ""

    eod_bar = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")
    eod_close = float(df["Close"].iloc[-1])

    if not use_live or stock is None or stock.price <= 0:
        return df, f"EOD bar **{eod_bar}** · close **₹{eod_close:,.2f}**", None, eod_close, eod_bar

    work = merge_live_into_daily(
        df,
        stock.price,
        day_high=stock.day_high,
        day_low=stock.day_low,
    )
    work = add_zones(work, dss_cfg)

    session_bar = pd.Timestamp(work.index[-1]).strftime("%Y-%m-%d")
    now = ist_now().strftime("%H:%M")
    label = (
        f"**Live LTP ₹{stock.price:,.2f}** · {now} IST · "
        f"zones on session bar **{session_bar}** · "
        f"EOD **{eod_bar}** close ₹{eod_close:,.2f}"
    )
    return work, label, stock.price, eod_close, eod_bar
