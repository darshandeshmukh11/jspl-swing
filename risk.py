"""Risk gates, position sizing, trade bias — loss minimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from jspl_config import JSPLSwingConfig
from confluence import ConfluenceResult
from session_plan import SessionPlan
from sentiment import SentimentBucket


@dataclass
class TradeDecision:
    bias: str  # STRONG_LONG | LONG | WAIT | REDUCE | AVOID
    score: int
    entry_zone: str
    stop: float
    target_1: float
    target_2: float
    suggested_qty: int
    reward_risk: float
    risk_inr: float
    gates_passed: list[str] = field(default_factory=list)
    gates_failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _sentiment_ok(buckets: dict[str, SentimentBucket], min_compound: float = -0.12) -> tuple[bool, str]:
    scores = [b.avg_compound for b in buckets.values() if b and b.count > 0]
    if not scores:
        return True, "sentiment data sparse — neutral pass"
    avg = sum(scores) / len(scores)
    if avg < min_compound:
        return False, f"aggregate sentiment {avg:.2f} < {min_compound}"
    return True, f"sentiment OK ({avg:.2f})"


def build_trade_decision(
    df: pd.DataFrame,
    plan: SessionPlan,
    confluence: ConfluenceResult,
    sentiment_buckets: dict[str, SentimentBucket],
    cfg: JSPLSwingConfig,
    relative_strength_20d: float | None = None,
) -> TradeDecision:
    dss = cfg.dss
    entry_mid = (plan.entry_low + plan.entry_high) / 2
    stop = plan.stop_long
    risk_per_share = max(entry_mid - stop, 0.01)
    target = plan.target_1
    reward = max(target - entry_mid, 0)
    rr = reward / risk_per_share if risk_per_share else 0

    qty = int(cfg.risk_per_trade_inr / risk_per_share) if risk_per_share else 0
    qty = min(qty, dss.swing_qty) if dss.swing_qty else qty
    qty = max(0, qty)

    passed: list[str] = []
    failed: list[str] = []

    if confluence.score >= 58:
        passed.append(f"Confluence {confluence.score}/100")
    else:
        failed.append(f"Confluence low ({confluence.score}/100)")

    sent_ok, sent_msg = _sentiment_ok(sentiment_buckets)
    if sent_ok:
        passed.append(sent_msg)
    else:
        failed.append(sent_msg)

    if rr >= cfg.min_reward_risk:
        passed.append(f"R:R {rr:.2f} ≥ {cfg.min_reward_risk}")
    else:
        failed.append(f"R:R {rr:.2f} < {cfg.min_reward_risk}")

    last = df.iloc[-1] if not df.empty else None
    if last is not None and float(last["Close"]) > float(last.get("EMA50", 0)):
        passed.append("Above 50 EMA")
    else:
        failed.append("Below 50 EMA")

    if relative_strength_20d is not None and relative_strength_20d >= -2:
        passed.append(f"RS vs metal {relative_strength_20d:+.1f}% (20d)")
    elif relative_strength_20d is not None:
        failed.append(f"Underperforming metal {relative_strength_20d:+.1f}%")

    if plan.triggers:
        passed.append(f"{len(plan.triggers)} technical trigger(s)")

    fail_count = len(failed)
    pass_count = len(passed)

    if confluence.score >= 72 and fail_count <= 1:
        bias = "STRONG_LONG"
    elif confluence.score >= 58 and fail_count <= 2 and sent_ok:
        bias = "LONG"
    elif confluence.score >= 45 and fail_count <= 3:
        bias = "WAIT"
    elif fail_count >= 4 or not sent_ok:
        bias = "AVOID"
    else:
        bias = "REDUCE"

    notes = list(plan.warnings) + confluence.reasons[:4]
    if bias in ("AVOID", "REDUCE"):
        qty = 0

    return TradeDecision(
        bias=bias,
        score=confluence.score,
        entry_zone=f"₹{plan.entry_low:,.2f} – ₹{plan.entry_high:,.2f}",
        stop=stop,
        target_1=plan.target_1,
        target_2=plan.target_2,
        suggested_qty=qty,
        reward_risk=round(rr, 2),
        risk_inr=round(risk_per_share * qty, 0) if qty else 0,
        gates_passed=passed,
        gates_failed=failed,
        notes=notes,
    )
