# JINDALSTEL Swing DSS

Self-contained Streamlit app for **JINDALSTEL.NS** swing decision support.  
No dependency on other folders in the monorepo — deploy this directory alone.

## Features

- Steel macro + sector + stock **news sentiment** (VADER; optional FinBERT)
- **Nifty Metal** live quote (or steel peer basket fallback)
- Technical stack: EMA, RSI, VWAP, ATR, MACD, ADX, Bollinger, Stochastic, Supertrend, floor pivots
- **Buy/sell initiation ranges** with prompts (INITIATE / PREPARE / WAIT) vs live price
- **Next-session probable range** (ATR + 20d avg range + pivots) with stop-loss and target map
- Buy/sell zones on chart, confluence score, next-session plan, ATR risk sizing

## Local run

```bash
cd jspl-swing   # this folder is the project root
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push **only this folder** as a Git repo root, **or** set the app path in a monorepo:
   - **Main file:** `app.py` (if repo root is `jspl-swing`)
   - **Main file:** `test/jspl-swing/app.py` (if repo root is parent monorepo)
2. **Python version:** 3.10+
3. Dependencies: **`requirements.txt` only** (no `packages.txt`).
4. No secrets required for Yahoo/RSS data.

### Monorepo settings (Streamlit Cloud)

| Setting | Value |
|---------|--------|
| Repository | your GitHub repo |
| Branch | `main` |
| Main file path | `test/jspl-swing/app.py` |
| App root (if available) | `test/jspl-swing` |

### Standalone repo (recommended)

Copy or publish `jspl-swing/` as its own repository so the Cloud app root matches the project root and imports resolve without path tricks.

## Project layout

```
jspl-swing/
├── app.py              # Streamlit entrypoint
├── bootstrap_path.py   # sys.path for Cloud/local
├── config.py           # DSSConfig (technical defaults)
├── jspl_config.py      # JINDALSTEL + metal settings
├── data.py             # Yahoo OHLCV + fundamentals
├── indicators.py
├── zones.py
├── trade_ranges.py   # Buy/sell initiation prompts
├── signals.py
├── research.py
├── advanced_ta.py
├── sentiment.py
├── market_live.py
├── confluence.py
├── session_plan.py
├── risk.py
├── pipeline.py
├── charts.py
├── requirements.txt    # Python deps (pip) — required for Cloud
└── .streamlit/config.toml
```

## Optional: FinBERT

```bash
pip install transformers torch
```

Enable **FinBERT** in the sidebar (first run downloads the model).

## Disclaimer

Research tool only — not investment advice. Market data may be delayed.
