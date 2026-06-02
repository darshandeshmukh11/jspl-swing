"""Plotly charts — zones, pivots, MACD, confluence levels."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jspl_config import JSPLSwingConfig
from session_plan import SessionPlan


def build_jspl_chart(
    df: pd.DataFrame,
    cfg: JSPLSwingConfig,
    plan: SessionPlan | None = None,
    trades=None,
) -> go.Figure:
    dss = cfg.dss
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.40, 0.12, 0.12, 0.12, 0.12],
        subplot_titles=(
            "Price · zones · pivots · Supertrend",
            "Volume",
            "RSI · Stochastic",
            "MACD",
            f"ATR ({dss.atr_period})",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    for col, name, color in [
        ("EMA20", "EMA 20", "#f59e0b"),
        ("EMA50", "EMA 50", "#3b82f6"),
        ("VWAP", "VWAP", "#c084fc"),
        ("SUPERTREND", "Supertrend", "#06b6d4"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], mode="lines", name=name, line=dict(color=color, width=1.5)),
                row=1,
                col=1,
            )

    tail = df.tail(90)
    for fill_cols, color, label in [
        (("BUY_ZONE_HIGH", "BUY_ZONE_LOW"), "rgba(34,197,94,0.12)", "Buy zone"),
        (("SELL_ZONE_HIGH", "SELL_ZONE_LOW"), "rgba(239,68,68,0.10)", "Sell zone"),
    ]:
        hi, lo = fill_cols
        fig.add_trace(
            go.Scatter(
                x=tail.index.tolist() + tail.index.tolist()[::-1],
                y=tail[hi].tolist() + tail[lo].tolist()[::-1],
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                name=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if plan:
        for y, name, color, dash in [
            (plan.pivot, "Pivot", "#c4b5fd", "dash"),
            (plan.s1, "S1", "#f87171", "solid"),
            (plan.s2, "S2", "#fca5a5", "dot"),
            (plan.r1, "R1", "#4ade80", "solid"),
            (plan.r2, "R2", "#86efac", "dot"),
            (plan.stop_long, "Stop", "#ef4444", "dash"),
            (plan.target_1, "T1", "#22c55e", "dash"),
        ]:
            fig.add_hline(y=y, line_dash=dash, line_color=color, annotation_text=name, row=1, col=1)

    buys = df[df["BUY_SIGNAL"]] if "BUY_SIGNAL" in df.columns else pd.DataFrame()
    sells = df[df["SELL_SIGNAL"]] if "SELL_SIGNAL" in df.columns else pd.DataFrame()
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Low"] * 0.992,
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
            ),
            row=1,
            col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["High"] * 1.008,
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
            ),
            row=1,
            col=1,
        )

    if "BB_UPPER" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_UPPER"], line=dict(color="#64748b", width=1, dash="dot"), name="BB upper"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_LOWER"], line=dict(color="#64748b", width=1, dash="dot"), name="BB lower"),
            row=1,
            col=1,
        )

    colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, opacity=0.5, name="Vol"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#eab308")), row=3, col=1)
    if "STOCH_K" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"], name="Stoch %K", line=dict(color="#a78bfa")), row=3, col=1)
    fig.add_hline(y=dss.rsi_buy_min, line_dash="dot", line_color="#22c55e", row=3, col=1)
    fig.add_hline(y=dss.rsi_sell_max, line_dash="dot", line_color="#ef4444", row=3, col=1)

    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#38bdf8")), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#f97316")), row=4, col=1)
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist", marker_color="#64748b", opacity=0.6),
            row=4,
            col=1,
        )

    fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR", fill="tozeroy", line=dict(color="#22d3ee")), row=5, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=920,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=80, b=40),
    )
    fig.update_yaxes(title_text="₹", row=1, col=1)
    return fig
