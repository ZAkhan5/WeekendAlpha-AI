# WeekendAlpha AI

WeekendAlpha AI is a premium, advisory-only AI Trading Desk built for the hackathon. It turns a market symbol into a compact research workflow: price context, technical indicators, regime classification, an explainable thesis, scenario ranges, a transparent risk budget, news/event context, and a multi-asset watchlist.

**Human remains the final decision-maker.** This application never connects to a broker, places trades, manages wallets, or acts autonomously. The risk engine is a planning aid, not financial advice.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The app uses Yahoo Finance when available. If network access or the optional live data dependency is unavailable, it automatically switches to deterministic demo data and keeps the entire analytical workflow usable.

## Product surface

- **Desk controls:** instrument, lookback, reference capital, risk-per-idea, and refresh.
- **Market view:** live/demo ticker strip, price with 20D/50D trend, RSI, MACD, volatility, and volume pulse.
- **AI research:** directional regime, confidence, evidence, invalidation level, and plain-language thesis.
- **Scenario map:** upside, base, and downside cases with target levels and probabilities.
- **Risk engine:** account risk budget, stop distance, volatility/confidence adjusted advisory allocation, and explicit assumptions.
- **Paper Trade Station:** a no-order review workflow for hypothetical stance, entry, invalidation, target, reward/risk, and scenario-weighted outcome.
- **Decision Stress Test:** finds similar historical indicator setups and reports 5D, 10D, and 20D forward outcomes without look-ahead leakage.
- **Research questions:** accepts English or Roman Urdu questions and answers from the supplied market context, with validated Qwen JSON and a local fallback.
- **News & events:** concise contextual headlines with tone labels and a fallback feed.
- **Security defaults:** no secrets in source control, XSRF protection, restricted local CORS origins, and no execution surface.

## Architecture

`data_layer.py` owns market/demo data and news; `analytics.py` owns indicators, regime logic, thesis generation, and scenarios; `qwen_client.py` optionally sends structured OHLCV/indicator context to Bitget Qwen and validates strict JSON; `risk_engine.py` owns sizing and exposure math; `app.py` composes the Streamlit terminal. The modules intentionally use small, inspectable calculations so the output is explainable during a live demo.

## Optional Qwen integration

The research pipeline is:

`Market Data -> Technical Indicators -> Structured Research Context -> Qwen -> Validated JSON -> Existing Research UI`

Configure the optional Bitget Qwen endpoint through environment variables only:

```bash
export BITGET_QWEN_API_KEY="your-key"
export BITGET_QWEN_BASE_URL="https://hackathon.bitgetops.com/v1"
export BITGET_QWEN_MODEL="qwen3.8-max"
```

Or copy `.env.example` for local reference and export the values through your shell or deployment platform. The app never displays or logs the API key. Qwen receives only available market/technical values and an explicit statement that live news and events are unavailable. It is instructed to provide research only, with no wallet, broker, order, or execution access.

When the key is absent, the UI shows `AI ENGINE · LOCAL FALLBACK` and uses the deterministic local research engine. When Qwen responds with valid schema-conforming JSON, it shows `AI ENGINE · Qwen LIVE`. Invalid JSON, invalid confidence or scenario weights, authentication failures, rate limits, timeouts, network failures, and malformed responses are retried once and then fall back safely.

## Security and scope

Do not put credentials in the repository. `.env` and `.env.*` files are ignored except for the template `.env.example`; Streamlit secrets belong in `.streamlit/secrets.toml`, which is also ignored. This project is intentionally research-only: no wallet keys, brokerage credentials, order endpoints, or autonomous agents are included. Qwen research confidence is not a probability of profit or direction, and all scenario weights are validated to total 100% before display.