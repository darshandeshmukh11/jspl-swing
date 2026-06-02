"""
Extended technical analysis — frameworks commonly used by swing traders.

Adds: floor pivots, MACD, Bollinger Bands, ADX, Stochastic, Supertrend-style ATR trail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def floor_pivots_from_bar(high: float, low: float, close: float) -> dict[str, float]:
    """Classic floor pivots for the *next* session from prior H/L/C."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return {
        "PIVOT": round(pivot, 2),
        "S1": round(s1, 2),
        "S2": round(s2, 2),
        "R1": round(r1, 2),
        "R2": round(r2, 2),
    }


def add_floor_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session pivot levels from each bar's H/L/C (last row = tomorrow's plan)."""
    out = df.copy()
    pivots = {k: [] for k in ("PIVOT", "S1", "S2", "R1", "R2")}
    for i in range(len(out)):
        row = out.iloc[i]
        p = floor_pivots_from_bar(float(row["High"]), float(row["Low"]), float(row["Close"]))
        for k in pivots:
            pivots[k].append(p[k])
    for k, vals in pivots.items():
        out[k] = vals
    return out


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.astype(float).ewm(span=span, adjust=False).mean()


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.astype(float).rolling(period).mean()
    std = close.astype(float).rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.rolling(period).mean()


def compute_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].astype(float).rolling(period).min()
    high_max = df["High"].astype(float).rolling(period).max()
    close = df["Close"].astype(float)
    k_raw = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    k = k_raw.rolling(smooth_k).mean()
    d = k.rolling(smooth_k).mean()
    return k, d


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """ATR-based Supertrend (popular intraday/swing trail)."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(1, index=df.index, dtype=int)

    for i in range(1, len(df)):
        if np.isnan(atr.iloc[i]):
            continue
        prev_st = st.iloc[i - 1]
        if np.isnan(prev_st):
            st.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
            continue
        if close.iloc[i] > prev_st:
            st.iloc[i] = max(lower_band.iloc[i], prev_st) if direction.iloc[i - 1] == 1 else lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            st.iloc[i] = min(upper_band.iloc[i], prev_st) if direction.iloc[i - 1] == -1 else upper_band.iloc[i]
            direction.iloc[i] = -1

    return st, direction


def add_advanced_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)

    macd, macd_sig, macd_hist = compute_macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["MACD"] = macd
    out["MACD_SIGNAL"] = macd_sig
    out["MACD_HIST"] = macd_hist
    out["MACD_BULL"] = (macd_hist > 0) & (macd_hist.shift(1) <= 0)

    bb_u, bb_m, bb_l = compute_bollinger(close, cfg.bb_period)
    out["BB_UPPER"] = bb_u
    out["BB_MID"] = bb_m
    out["BB_LOWER"] = bb_l
    width = (bb_u - bb_l) / bb_m.replace(0, np.nan)
    out["BB_WIDTH"] = width
    out["BB_PCT_B"] = (close - bb_l) / (bb_u - bb_l).replace(0, np.nan)

    out["ADX"] = compute_adx(out, 14)
    stoch_k, stoch_d = compute_stochastic(out, cfg.stoch_period)
    out["STOCH_K"] = stoch_k
    out["STOCH_D"] = stoch_d

    st_line, st_dir = compute_supertrend(out, 10, 3.0)
    out["SUPERTREND"] = st_line
    out["SUPERTREND_DIR"] = st_dir

    out = add_floor_pivots(out)
    return out
