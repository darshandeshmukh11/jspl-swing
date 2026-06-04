#!/usr/bin/env python3
"""
JINDALSTEL swing trading decision support — Streamlit UI (self-contained).

Run from this directory:
  pip install -r requirements.txt && streamlit run app.py
"""

from __future__ import annotations

import bootstrap_path  # noqa: F401

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import build_jspl_chart
from confluence import compute_confluence
from jspl_config import JSPLSwingConfig
from market_live import fetch_live_dashboard
from pipeline import build_enriched_frame
from research import build_analyst_view
from risk import build_trade_decision
from sentiment import fetch_all_sentiment, sentiment_to_dataframe
from session_plan import build_session_plan
from trade_ranges import build_trade_initiation_guide

st.set_page_config(
    page_title="JINDALSTEL Swing DSS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "_Research support only — not investment advice. Free data may be delayed. "
    "Past signals ≠ future results._"
)


def _dark_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0a0a0a; }
        [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #333; }
        h1,h2,h3 { color: #f4f4f5 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar() -> tuple[JSPLSwingConfig, bool]:
    st.sidebar.header("JSPL Swing DSS")
    st.sidebar.caption("JINDALSTEL.NS · Nifty Metal · steel sentiment")

    risk_inr = st.sidebar.number_input("Risk per trade (₹)", 5000, 200_000, 25_000, 5000)
    min_rr = st.sidebar.slider("Min reward:risk", 1.0, 3.0, 1.5, 0.1)
    swing_pct = st.sidebar.slider("Swing tranche %", 5, 25, 10)
    stop_atr = st.sidebar.slider("Stop (× ATR)", 1.0, 3.0, 1.5, 0.1)
    years = st.sidebar.selectbox("History", [2, 3, 5], index=1)
    use_finbert = st.sidebar.checkbox("FinBERT (slow, needs transformers)", value=False)
    auto = st.sidebar.checkbox("Auto-refresh", value=False)
    if auto:
        st.sidebar.slider("Refresh (sec)", 60, 600, 300, 30, key="refresh_sec")

    cfg = JSPLSwingConfig(
        years=int(years),
        swing_pct=float(swing_pct),
        stop_atr_mult=float(stop_atr),
        min_reward_risk=float(min_rr),
        risk_per_trade_inr=float(risk_inr),
        use_finbert=use_finbert,
    )
    return cfg, auto


@st.cache_data(ttl=3600, show_spinner="Loading OHLCV & indicators…")
def _load_technicals(cfg: JSPLSwingConfig):
    return build_enriched_frame(cfg)


@st.cache_data(ttl=300, show_spinner="Fetching news sentiment…")
def _load_sentiment(cfg: JSPLSwingConfig):
    return fetch_all_sentiment(
        cfg.yahoo_ticker,
        cfg.peer_yahoo_tickers,
        cfg.max_headlines_per_bucket,
        cfg.use_finbert,
    )


@st.cache_data(ttl=90, show_spinner="Live quotes…")
def _load_live(cfg: JSPLSwingConfig):
    return fetch_live_dashboard(
        cfg.yahoo_ticker,
        cfg.metal_index_candidates,
        cfg.peer_yahoo_tickers,
    )


def _bias_color(bias: str) -> str:
    return {
        "STRONG_LONG": "#22c55e",
        "LONG": "#4ade80",
        "WAIT": "#eab308",
        "REDUCE": "#f97316",
        "AVOID": "#ef4444",
    }.get(bias, "#94a3b8")


def _action_badge(action: str) -> str:
    colors = {
        "INITIATE": "#22c55e",
        "PREPARE": "#3b82f6",
        "WAIT": "#eab308",
    }
    return colors.get(action, "#94a3b8")


def _render_trade_initiation_ranges(guide, decision, as_of: str) -> None:
    """Prominent buy/sell initiation ranges and prompts."""
    st.subheader("Where to initiate buy or sell")
    st.caption(
        f"Reference price: **₹{guide.reference_price:,.2f}** · Zones from EOD bar **{as_of}**"
    )

    b_col, s_col = st.columns(2)

    with b_col:
        st.markdown(
            f'<div style="padding:14px;border-radius:8px;border:1px solid #333;'
            f'border-left:5px solid {_action_badge(guide.buy_action)};background:#141414">'
            f'<p style="margin:0 0 8px;color:#a1a1aa;font-size:0.85rem">BUY initiation</p>'
            f'<p style="margin:0;font-size:1.35rem;font-weight:600;color:#4ade80">'
            f"₹{guide.buy_low:,.2f} – ₹{guide.buy_high:,.2f}</p>"
            f'<p style="margin:6px 0 0;color:#71717a;font-size:0.8rem">'
            f"Entry band ₹{guide.entry_low:,.2f}–₹{guide.entry_high:,.2f} · "
            f"Stop ₹{guide.stop:,.2f}</p>"
            f'<p style="margin:10px 0 0;color:#e4e4e7;font-size:0.95rem">{guide.buy_prompt}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if guide.buy_status == "IN_ZONE":
            st.success(f"Action: **{guide.buy_action}** — price is in the buy zone")
        elif guide.buy_action == "PREPARE":
            st.info(f"Action: **{guide.buy_action}** — set limits in the buy range")
        else:
            st.warning(f"Action: **{guide.buy_action}**")

    with s_col:
        st.markdown(
            f'<div style="padding:14px;border-radius:8px;border:1px solid #333;'
            f'border-left:5px solid {_action_badge(guide.sell_action)};background:#141414">'
            f'<p style="margin:0 0 8px;color:#a1a1aa;font-size:0.85rem">SELL initiation</p>'
            f'<p style="margin:0;font-size:1.35rem;font-weight:600;color:#f87171">'
            f"₹{guide.sell_low:,.2f} – ₹{guide.sell_high:,.2f}</p>"
            f'<p style="margin:6px 0 0;color:#71717a;font-size:0.8rem">'
            f"T1 ₹{guide.target_1:,.2f} · T2 ₹{guide.target_2:,.2f}</p>"
            f'<p style="margin:10px 0 0;color:#e4e4e7;font-size:0.95rem">{guide.sell_prompt}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if guide.sell_status == "IN_ZONE":
            st.success(f"Action: **{guide.sell_action}** — price is in the sell zone")
        elif guide.sell_action == "INITIATE":
            st.success(f"Action: **{guide.sell_action}**")
        elif guide.sell_action == "PREPARE":
            st.info(f"Action: **{guide.sell_action}**")
        else:
            st.warning(f"Action: **{guide.sell_action}**")

    if decision.bias in ("AVOID", "REDUCE"):
        st.error(
            f"Bias is **{decision.bias}** — treat buy ranges as reference only; "
            "prefer risk reduction or wait until gates pass."
        )
    elif decision.bias == "WAIT":
        st.warning(
            "Bias is **WAIT** — use buy zone for staged limits; confirm triggers on the **Next session** tab."
        )

    with st.expander("Quick reference — all price levels", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Use": "Buy zone (initiate long)", "Low ₹": guide.buy_low, "High ₹": guide.buy_high},
                    {"Use": "Sell zone (book / trim)", "Low ₹": guide.sell_low, "High ₹": guide.sell_high},
                    {"Use": "Suggested entry", "Low ₹": guide.entry_low, "High ₹": guide.entry_high},
                    {"Use": "Stop loss", "Low ₹": guide.stop, "High ₹": "—"},
                    {"Use": "Target 1 / 2", "Low ₹": guide.target_1, "High ₹": guide.target_2},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    _dark_css()
    cfg, auto_refresh = _sidebar()

    if st.sidebar.button("Refresh now", type="primary"):
        st.cache_data.clear()

    st.title("JINDALSTEL Swing Decision Support")
    st.caption("Steel macro · sector · stock sentiment · confluence · ATR risk plan")
    st.markdown(DISCLAIMER)

    live = _load_live(cfg)
    sentiment = _load_sentiment(cfg)
    df, fundamentals, adherence, swing_bt = _load_technicals(cfg)

    plan = build_session_plan(df, cfg)
    confluence = compute_confluence(
        df,
        sentiment,
        cfg,
        live.get("relative_strength_20d"),
    )
    decision = build_trade_decision(
        df,
        plan,
        confluence,
        sentiment,
        cfg,
        live.get("relative_strength_20d"),
    )

    # --- Top metrics ---
    stock = live["stock"]
    metal = live["metal"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("JINDALSTEL", f"₹{stock.price:,.2f}", f"{stock.change_pct:+.2f}%")
    c2.metric(metal.name, f"₹{metal.price:,.2f}" if metal.price else "—", f"{metal.change_pct:+.2f}%")
    c3.metric("Confluence", f"{confluence.score}/100", confluence.label)
    c4.metric("Bias", decision.bias, f"R:R {decision.reward_risk:.1f}")
    c5.metric("ATR (next session)", f"₹{plan.atr:,.2f}", f"{plan.atr_pct:.2f}% of price")
    rs = live.get("relative_strength_20d")
    c6.metric("RS vs metal (20d)", f"{rs:+.1f}%" if rs is not None else "—")

    guide = build_trade_initiation_guide(plan, stock.price)
    _render_trade_initiation_ranges(guide, decision, plan.as_of)

    st.markdown(
        f'<div style="padding:12px;border-left:4px solid {_bias_color(decision.bias)};'
        f'background:#1a1a1a;margin:8px 0">'
        f"<b>Trade bias: {decision.bias}</b> · Entry {decision.entry_zone} · "
        f"Stop ₹{decision.stop:,.2f} · T1 ₹{decision.target_1:,.2f} · T2 ₹{decision.target_2:,.2f} · "
        f"Qty {decision.suggested_qty:,} (risk ₹{decision.risk_inr:,.0f})</div>",
        unsafe_allow_html=True,
    )

    tab_dash, tab_news, tab_ta, tab_plan, tab_risk, tab_research = st.tabs(
        ["Dashboard", "Sentiment", "Technicals", "Next session", "Risk gates", "Research"]
    )

    with tab_dash:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader("Sentiment summary")
            st.dataframe(sentiment_to_dataframe(sentiment), use_container_width=True, hide_index=True)
            for key, bucket in sentiment.items():
                st.progress(
                    min(1.0, max(0.0, (bucket.avg_compound + 1) / 2)),
                    text=f"{bucket.name}: {bucket.label} ({bucket.avg_compound:+.2f})",
                )
        with col_r:
            st.subheader("Steel peers")
            if live["peers"]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Ticker": p.ticker,
                                "Price": p.price,
                                "Chg %": p.change_pct,
                            }
                            for p in live["peers"]
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            if metal.source_note:
                st.caption(metal.source_note)
            spark = live.get("stock_sparkline")
            if spark is not None and not spark.empty:
                fig_s = go.Figure()
                fig_s.add_trace(
                    go.Scatter(x=spark.index, y=spark["close"], mode="lines", line=dict(color="#22c55e"))
                )
                fig_s.update_layout(template="plotly_dark", height=220, margin=dict(l=40, r=20, t=30, b=30))
                st.plotly_chart(fig_s, use_container_width=True)

        st.subheader("Confluence breakdown")
        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Trend", f"{confluence.trend_pts}/30")
        bc2.metric("Momentum", f"{confluence.momentum_pts}/25")
        bc3.metric("Structure", f"{confluence.structure_pts}/25")
        bc4.metric("Volatility", f"{confluence.volatility_pts}/10")
        bc5.metric("Sentiment+RS", f"{confluence.sentiment_pts}/10")
        for r in confluence.reasons:
            st.markdown(f"- {r}")

    with tab_news:
        for key, bucket in sentiment.items():
            st.subheader(f"{bucket.name} — {bucket.label.upper()}")
            if not bucket.headlines:
                st.info("No headlines fetched.")
                continue
            rows = [
                {
                    "Title": h.title,
                    "VADER": f"{h.vader_compound:+.2f}",
                    "FinBERT": h.finbert_label or "—",
                    "Source": h.source,
                }
                for h in bucket.headlines[:15]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_ta:
        st.plotly_chart(build_jspl_chart(df, cfg, plan, swing_bt.trades), use_container_width=True)
        st.subheader("Indicator snapshot (latest)")
        last = df.iloc[-1]
        ind_df = pd.DataFrame(
            [
                {"Indicator": "Close", "Value": f"₹{float(last['Close']):,.2f}"},
                {"Indicator": "RSI", "Value": f"{float(last['RSI']):.1f}"},
                {"Indicator": "ADX", "Value": f"{float(last.get('ADX', 0)):.1f}"},
                {"Indicator": "MACD hist", "Value": f"{float(last.get('MACD_HIST', 0)):.3f}"},
                {"Indicator": "Stoch %K", "Value": f"{float(last.get('STOCH_K', 0)):.1f}"},
                {"Indicator": "BB %B", "Value": f"{float(last.get('BB_PCT_B', 0)):.2f}"},
                {"Indicator": "Vol ratio", "Value": f"{float(last['VOL_RATIO']):.2f}×"},
                {"Indicator": "Supertrend", "Value": "UP" if int(last.get("SUPERTREND_DIR", -1)) == 1 else "DOWN"},
            ]
        )
        st.dataframe(ind_df, use_container_width=True, hide_index=True)

        zdf = pd.DataFrame(
            [
                {"Zone": "Buy", "Low": plan.buy_zone_low, "High": plan.buy_zone_high},
                {"Zone": "Sell", "Low": plan.sell_zone_low, "High": plan.sell_zone_high},
            ]
        )
        st.dataframe(zdf, use_container_width=True, hide_index=True)

    with tab_plan:
        st.subheader(f"Next session plan (from {plan.as_of} close)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry zone", f"₹{plan.entry_low:,.2f} – ₹{plan.entry_high:,.2f}")
        p2.metric("Stop (tight)", f"₹{plan.stop_long:,.2f}")
        p3.metric("Stop (wide S2)", f"₹{plan.stop_wide:,.2f}")
        p4.metric("ATR", f"₹{plan.atr:,.2f}")

        piv = pd.DataFrame(
            [
                {"Level": "Pivot", "₹": plan.pivot},
                {"Level": "S1", "₹": plan.s1},
                {"Level": "S2", "₹": plan.s2},
                {"Level": "R1", "₹": plan.r1},
                {"Level": "R2", "₹": plan.r2},
                {"Level": "Target 1", "₹": plan.target_1},
                {"Level": "Target 2", "₹": plan.target_2},
            ]
        )
        st.dataframe(piv, use_container_width=True, hide_index=True)

        st.markdown("**Triggers**")
        for t in plan.triggers:
            st.success(t)
        if not plan.triggers:
            st.warning("No active triggers — wait for dip into buy zone or signal.")
        if plan.warnings:
            st.markdown("**Warnings**")
            for w in plan.warnings:
                st.warning(w)

    with tab_risk:
        st.subheader("Gate checklist")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Passed**")
            for g in decision.gates_passed:
                st.success(g)
        with g2:
            st.markdown("**Failed / caution**")
            for g in decision.gates_failed:
                st.error(g)
        for n in decision.notes:
            st.info(n)

        st.subheader("Loss minimization rules")
        st.markdown(
            f"""
- **ATR stop:** {cfg.stop_atr_mult:.1f}× ATR below entry (~₹{plan.stop_long:,.2f})
- **Position size:** risk ₹{cfg.risk_per_trade_inr:,.0f} ÷ (entry − stop) → **{decision.suggested_qty:,}** shares (capped at swing tranche {cfg.dss.swing_qty:,})
- **Min R:R:** {cfg.min_reward_risk:.1f} (current **{decision.reward_risk:.2f}**)
- **Sentiment gate:** block new longs when macro+sector+stock VADER avg &lt; −0.12
- **Trend gate:** prefer longs only above 50 EMA
- **Profit target on swing leg:** +{cfg.profit_target_pct:.1f}% (from backtest config)
            """
        )

    with tab_research:
        narrative = build_analyst_view(
            cfg.symbol,
            df,
            fundamentals,
            adherence,
            swing_bt,
            cfg.dss,
        )
        for line in narrative:
            st.markdown(line)

        st.subheader("Simulated swing tranche")
        if swing_bt.trades:
            rows = [
                {
                    "Side": t.side,
                    "Date": pd.Timestamp(t.date).strftime("%Y-%m-%d"),
                    "Price": t.price,
                    "Qty": t.qty,
                    "Reason": t.reason,
                    "P&L": t.pnl,
                }
                for t in swing_bt.trades[-12:]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric("Win rate (sim)", f"{swing_bt.win_rate:.0f}%")
        st.metric("Signals buy / sell", f"{swing_bt.signals_buy} / {swing_bt.signals_sell}")

    if auto_refresh:
        st.sidebar.caption("Auto-refresh: use **Refresh now** or reload the page (Streamlit reruns on interaction).")
    st.markdown(DISCLAIMER)


if __name__ == "__main__":
    main()
