"""
Live / near-live quotes — JINDALSTEL, Nifty Metal (or steel basket proxy), sector peers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class LiveQuote:
    ticker: str
    name: str
    price: float
    change_pct: float
    day_high: float
    day_low: float
    volume: float
    source_note: str = ""


def _quote_from_info(ticker: str, name: str, note: str = "") -> Optional[LiveQuote]:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(getattr(info, "last_price", None) or getattr(info, "lastPrice", 0) or 0)
        prev = float(getattr(info, "previous_close", None) or getattr(info, "previousClose", 0) or price)
        if price <= 0:
            hist = t.history(period="5d", interval="1d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        chg = (price / prev - 1) * 100 if prev else 0.0
        return LiveQuote(
            ticker=ticker,
            name=name,
            price=round(price, 2),
            change_pct=round(chg, 2),
            day_high=round(float(getattr(info, "day_high", price) or price), 2),
            day_low=round(float(getattr(info, "day_low", price) or price), 2),
            volume=float(getattr(info, "last_volume", 0) or 0),
            source_note=note,
        )
    except Exception:
        return None


def fetch_intraday_sparkline(ticker: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame()
        return df[["Close"]].rename(columns={"Close": "close"})
    except Exception:
        return pd.DataFrame()


def resolve_metal_index(candidates: tuple[str, ...]) -> tuple[str, str]:
    for t in candidates:
        q = _quote_from_info(t, "Nifty Metal")
        if q and q.price > 0:
            return t, "index"
    return "", "basket"


def steel_basket_quote(peer_tickers: list[str]) -> LiveQuote:
    """Equal-weight average of liquid steel peers when index ticker unavailable."""
    prices: list[float] = []
    changes: list[float] = []
    used: list[str] = []
    for t in peer_tickers:
        q = _quote_from_info(t, t)
        if q and q.price > 0:
            prices.append(q.price)
            changes.append(q.change_pct)
            used.append(t)
    if not prices:
        return LiveQuote("basket", "Steel basket", 0.0, 0.0, 0.0, 0.0, 0.0, "unavailable")
    return LiveQuote(
        ticker="STEEL_BASKET",
        name=f"Steel basket ({len(used)} peers)",
        price=round(float(np.mean(prices)), 2),
        change_pct=round(float(np.mean(changes)), 2),
        day_high=0.0,
        day_low=0.0,
        volume=0.0,
        source_note=f"Proxy: {', '.join(used[:4])}…",
    )


def fetch_live_dashboard(
    stock_ticker: str,
    metal_candidates: tuple[str, ...],
    peer_tickers: list[str],
) -> dict:
    stock = _quote_from_info(stock_ticker, "JINDALSTEL") or LiveQuote(
        stock_ticker, "JINDALSTEL", 0.0, 0.0, 0.0, 0.0, 0.0, "unavailable"
    )
    metal_ticker, metal_kind = resolve_metal_index(metal_candidates)
    if metal_kind == "index" and metal_ticker:
        metal = _quote_from_info(metal_ticker, "Nifty Metal", "NSE metal index") or steel_basket_quote(
            peer_tickers
        )
    else:
        metal = steel_basket_quote(peer_tickers)

    peers: list[LiveQuote] = []
    for t in peer_tickers[:6]:
        q = _quote_from_info(t, t.split(".")[0])
        if q:
            peers.append(q)

    rs_20d = compute_relative_strength(stock_ticker, metal_ticker or peer_tickers[0], peer_tickers)
    return {
        "stock": stock,
        "metal": metal,
        "peers": peers,
        "relative_strength_20d": rs_20d,
        "stock_sparkline": fetch_intraday_sparkline(stock_ticker),
        "metal_sparkline": fetch_intraday_sparkline(metal_ticker) if metal_ticker else pd.DataFrame(),
    }


def compute_relative_strength(
    stock_ticker: str,
    benchmark_ticker: str,
    basket_fallback: list[str],
) -> Optional[float]:
    """Stock 20d return minus benchmark 20d return (%). Positive = outperforming metal."""
    try:
        stock_hist = yf.download(stock_ticker, period="1mo", progress=False, auto_adjust=True)
        if isinstance(stock_hist.columns, pd.MultiIndex):
            stock_hist.columns = stock_hist.columns.get_level_values(0)
        if len(stock_hist) < 10:
            return None
        stock_ret = float(stock_hist["Close"].iloc[-1] / stock_hist["Close"].iloc[0] - 1) * 100

        bench_ret = None
        if benchmark_ticker:
            bench_hist = yf.download(benchmark_ticker, period="1mo", progress=False, auto_adjust=True)
            if isinstance(bench_hist.columns, pd.MultiIndex):
                bench_hist.columns = bench_hist.columns.get_level_values(0)
            if len(bench_hist) >= 10:
                bench_ret = float(bench_hist["Close"].iloc[-1] / bench_hist["Close"].iloc[0] - 1) * 100

        if bench_ret is None:
            rets = []
            for t in basket_fallback[:4]:
                h = yf.download(t, period="1mo", progress=False, auto_adjust=True)
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = h.columns.get_level_values(0)
                if len(h) >= 10:
                    rets.append(float(h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100)
            bench_ret = float(np.mean(rets)) if rets else 0.0

        return round(stock_ret - bench_ret, 2)
    except Exception:
        return None
