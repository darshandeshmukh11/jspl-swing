"""
News sentiment — VADER (fast) + optional FinBERT (financial headlines).

Sources: Google News RSS (feedparser), yfinance ticker news.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER = None

_FINBERT_PIPELINE = None


@dataclass
class HeadlineScore:
    title: str
    source: str
    published: str
    vader_compound: float
    finbert_label: Optional[str] = None
    finbert_score: Optional[float] = None


@dataclass
class SentimentBucket:
    name: str
    headlines: list[HeadlineScore] = field(default_factory=list)
    avg_compound: float = 0.0
    bullish_pct: float = 0.0
    bearish_pct: float = 0.0
    count: int = 0
    label: str = "neutral"


def _clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:500]


def _vader_compound(text: str) -> float:
    if not text or _VADER is None:
        return 0.0
    return float(_VADER.polarity_scores(text)["compound"])


def _fetch_google_news_rss(query: str, max_items: int) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    items: list[dict] = []
    for entry in feed.entries[:max_items]:
        published = getattr(entry, "published", "") or ""
        items.append(
            {
                "title": _clean_title(getattr(entry, "title", "")),
                "source": getattr(entry, "source", {}).get("title", "Google News") if hasattr(entry, "source") else "Google News",
                "published": published,
            }
        )
    return items


def _fetch_yfinance_news(ticker: str, max_items: int) -> list[dict]:
    items: list[dict] = []
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return items
    for n in raw[:max_items]:
        title = _clean_title(n.get("title", ""))
        if not title:
            continue
        pub = n.get("providerPublishTime")
        published = (
            datetime.fromtimestamp(pub, tz=timezone.utc).isoformat() if pub else ""
        )
        items.append(
            {
                "title": title,
                "source": n.get("publisher", "Yahoo"),
                "published": published,
            }
        )
    return items


def _finbert_score(text: str) -> tuple[Optional[str], Optional[float]]:
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        try:
            from transformers import pipeline

            _FINBERT_PIPELINE = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
        except Exception:
            return None, None
    try:
        out = _FINBERT_PIPELINE(text[:512])[0]
        label = str(out.get("label", ""))
        score = float(out.get("score", 0))
        return label, score
    except Exception:
        return None, None


def score_headlines(
    raw_items: list[dict],
    use_finbert: bool = False,
) -> list[HeadlineScore]:
    seen: set[str] = set()
    scored: list[HeadlineScore] = []
    for item in raw_items:
        title = item.get("title", "")
        if not title or title in seen:
            continue
        seen.add(title)
        compound = _vader_compound(title)
        fb_label, fb_score = (None, None)
        if use_finbert:
            fb_label, fb_score = _finbert_score(title)
        scored.append(
            HeadlineScore(
                title=title,
                source=str(item.get("source", "")),
                published=str(item.get("published", "")),
                vader_compound=compound,
                finbert_label=fb_label,
                finbert_score=fb_score,
            )
        )
    return scored


def aggregate_bucket(name: str, headlines: list[HeadlineScore]) -> SentimentBucket:
    if not headlines:
        return SentimentBucket(name=name, label="no data")
    compounds = [h.vader_compound for h in headlines]
    avg = float(sum(compounds) / len(compounds))
    bull = sum(1 for c in compounds if c >= 0.05) / len(compounds) * 100
    bear = sum(1 for c in compounds if c <= -0.05) / len(compounds) * 100
    if avg >= 0.12:
        label = "bullish"
    elif avg <= -0.12:
        label = "bearish"
    else:
        label = "neutral"
    return SentimentBucket(
        name=name,
        headlines=headlines,
        avg_compound=round(avg, 4),
        bullish_pct=round(bull, 1),
        bearish_pct=round(bear, 1),
        count=len(headlines),
        label=label,
    )


def fetch_all_sentiment(
    yahoo_ticker: str,
    peer_tickers: list[str],
    max_per_bucket: int = 25,
    use_finbert: bool = False,
    *,
    symbol: str = "",
    company_name: str = "",
    sector: str = "",
    industry: str = "",
    is_steel_stock: bool = False,
) -> dict[str, SentimentBucket]:
    from news_queries import (
        JINDAL_QUERIES,
        STEEL_MACRO_QUERIES,
        STEEL_SECTOR_QUERIES,
        sector_queries,
        stock_queries,
    )

    macro_queries = STEEL_MACRO_QUERIES if is_steel_stock else sector_queries(sector, industry)
    macro_raw: list[dict] = []
    for q in macro_queries[:3]:
        macro_raw.extend(_fetch_google_news_rss(q, max_per_bucket // 3 + 1))

    sector_raw: list[dict] = []
    sector_query_list = STEEL_SECTOR_QUERIES if is_steel_stock else sector_queries(sector, industry)
    for q in sector_query_list[:2]:
        sector_raw.extend(_fetch_google_news_rss(q, max_per_bucket // 2 + 1))
    for pt in peer_tickers[:4]:
        sector_raw.extend(_fetch_yfinance_news(pt, 5))

    stock_raw: list[dict] = []
    stock_query_list = JINDAL_QUERIES if is_steel_stock else stock_queries(symbol or yahoo_ticker, company_name)
    for q in stock_query_list:
        stock_raw.extend(_fetch_google_news_rss(q, max_per_bucket // 2 + 1))
    stock_raw.extend(_fetch_yfinance_news(yahoo_ticker, max_per_bucket))

    stock_label = company_name or symbol or yahoo_ticker

    return {
        "sector_macro": aggregate_bucket(
            "Sector / macro",
            score_headlines(macro_raw[:max_per_bucket], use_finbert),
        ),
        "sector_peers": aggregate_bucket(
            "Sector peers",
            score_headlines(sector_raw[:max_per_bucket], use_finbert),
        ),
        "stock": aggregate_bucket(
            stock_label,
            score_headlines(stock_raw[:max_per_bucket], use_finbert),
        ),
    }


def sentiment_to_dataframe(buckets: dict[str, SentimentBucket]) -> pd.DataFrame:
    rows = []
    for key, b in buckets.items():
        rows.append(
            {
                "bucket": b.name,
                "label": b.label,
                "avg_compound": b.avg_compound,
                "bullish_%": b.bullish_pct,
                "bearish_%": b.bearish_pct,
                "headlines": b.count,
            }
        )
    return pd.DataFrame(rows)
