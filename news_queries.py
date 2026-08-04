"""Google News RSS query strings for sentiment buckets."""

from __future__ import annotations

STEEL_MACRO_QUERIES: list[str] = [
    "India steel sector NSE",
    "steel prices India demand",
    "Indian steel exports tariffs",
    "HRC steel India",
    "metals mining India NSE",
]

JINDAL_QUERIES: list[str] = [
    "Jindal Steel Power JINDALSTEL",
    "Jindal Steel NSE",
    "JSPL steel India",
]

STEEL_SECTOR_QUERIES: list[str] = [
    "Tata Steel JSW Steel Hindalco",
    "SAIL Vedanta steel India",
    "Nifty Metal index steel stocks",
]

GENERIC_MARKET_QUERIES: list[str] = [
    "Indian stock market NSE BSE",
    "Nifty Sensex market news",
    "India equity market outlook",
]


def stock_queries(symbol: str, company_name: str = "") -> list[str]:
    """Search-engine-friendly queries for an arbitrary NSE-listed stock."""
    sym = symbol.strip().upper()
    name = (company_name or "").strip()
    queries = [f"{sym} NSE stock", f"{sym} share price India"]
    if name and name.upper() != sym:
        queries.insert(0, f"{name} {sym}")
        queries.append(f"{name} share news")
    return queries


def sector_queries(sector: str = "", industry: str = "") -> list[str]:
    """Sector/industry-level queries built from fundamentals when available."""
    label = (industry or sector or "").strip()
    if not label:
        return list(GENERIC_MARKET_QUERIES)
    return [
        f"{label} sector India NSE",
        f"{label} India news",
        f"{label} companies India stock",
    ]
