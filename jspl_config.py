"""JINDALSTEL swing DSS configuration."""

from __future__ import annotations

from dataclasses import dataclass

import bootstrap_path  # noqa: F401

from config import DSSConfig as _DSSConfig


@dataclass
class JSPLSwingConfig:
    """Wraps shared backtest DSSConfig with JSPL / metal-specific settings."""

    symbol: str = "JINDALSTEL"
    years: int = 3
    core_holding_qty: int = 4460
    swing_pct: float = 10.0
    profit_target_pct: float = 2.5
    stop_atr_mult: float = 1.5
    min_reward_risk: float = 1.5
    risk_per_trade_inr: float = 25_000.0

    sentiment_refresh_sec: int = 300
    price_refresh_sec: int = 90
    use_finbert: bool = False
    max_headlines_per_bucket: int = 25

    metal_index_candidates: tuple[str, ...] = (
        "^CNXMETAL",
        "NIFTY_METAL.NS",
        "CNXMETAL.NS",
    )
    steel_peers: tuple[str, ...] = (
        "TATASTEEL",
        "JSWSTEEL",
        "HINDALCO",
        "SAIL",
        "HINDZINC",
        "VEDL",
        "NMDC",
    )

    adx_trend_min: float = 22.0
    adx_strong: float = 28.0
    rsi_oversold: float = 35.0
    rsi_overbought: float = 72.0
    bb_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_period: int = 14

    @property
    def dss(self) -> _DSSConfig:
        return _DSSConfig(
            symbol=self.symbol,
            years=self.years,
            core_holding_qty=self.core_holding_qty,
            swing_pct=self.swing_pct,
            profit_target_pct=self.profit_target_pct,
            stop_atr_mult=self.stop_atr_mult,
        )

    @property
    def yahoo_ticker(self) -> str:
        return self.dss.yahoo_ticker

    @property
    def peer_yahoo_tickers(self) -> list[str]:
        from data import to_yahoo_nse

        return [to_yahoo_nse(s) for s in self.steel_peers]
