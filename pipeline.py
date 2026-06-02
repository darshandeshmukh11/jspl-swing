"""Full enrichment pipeline: backtest core + advanced TA + signals."""

from __future__ import annotations

import pandas as pd

import bootstrap_path  # noqa: F401

from advanced_ta import add_advanced_indicators
from jspl_config import JSPLSwingConfig
from data import fetch_fundamentals, fetch_ohlcv
from indicators import add_indicators
from signals import analyze_indicator_adherence, generate_signals, simulate_swing_tranche
from zones import add_zones, latest_zones


def build_enriched_frame(cfg: JSPLSwingConfig) -> tuple[pd.DataFrame, object, object, object]:
    dss = cfg.dss
    ohlcv = fetch_ohlcv(dss.symbol, dss.years)
    fundamentals = fetch_fundamentals(dss.symbol)
    df = add_indicators(ohlcv, dss)
    df = add_zones(df, dss)
    df = add_advanced_indicators(df, cfg)
    df = generate_signals(df, dss)
    adherence = analyze_indicator_adherence(df)
    swing_bt = simulate_swing_tranche(df, dss)
    return df, fundamentals, adherence, swing_bt


def latest_row_dict(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {k: (float(v) if pd.notna(v) and isinstance(v, (int, float)) else v) for k, v in row.items()}
