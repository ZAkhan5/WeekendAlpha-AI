"""Market data and research feed helpers with an offline-first fallback."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any

import numpy as np
import pandas as pd

ASSET_UNIVERSE = {
    "BTC-USD": {"name": "Bitcoin", "sector": "Digital assets", "kind": "crypto", "price": 64280.0},
    "ETH-USD": {"name": "Ethereum", "sector": "Digital assets", "kind": "crypto", "price": 3485.0},
    "SOL-USD": {"name": "Solana", "sector": "Digital assets", "kind": "crypto", "price": 146.0},
    "SPY": {"name": "S&P 500 ETF", "sector": "US equities", "kind": "equity", "price": 542.0},
    "QQQ": {"name": "Nasdaq 100 ETF", "sector": "US equities", "kind": "equity", "price": 465.0},
    "NVDA": {"name": "NVIDIA", "sector": "Semiconductors", "kind": "equity", "price": 121.0},
}
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _seed(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8], 16)


def demo_prices(symbol: str, periods: int = 180) -> pd.DataFrame:
    metadata = ASSET_UNIVERSE.get(symbol, {"price": 100.0})
    rng = np.random.default_rng(_seed(symbol))
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=periods, freq="D")
    base = float(metadata["price"])
    drift = rng.normal(0.0008, 0.0004)
    shocks = rng.normal(0, 0.018 if metadata.get("kind") == "crypto" else 0.011, periods)
    cycle = np.sin(np.arange(periods) / 13.0) * 0.004
    close = base * np.exp(np.cumsum(drift + shocks + cycle))
    close = close / close[-1] * base
    open_price = close * (1 + rng.normal(0, 0.004, periods))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.014, periods))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.014, periods))
    volume = rng.lognormal(17 if metadata.get("kind") == "crypto" else 16, 0.35, periods)
    return pd.DataFrame({"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def fetch_prices(symbol: str, period: str = "6mo", force_demo: bool = False) -> tuple[pd.DataFrame, str, str | None]:
    if symbol not in ASSET_UNIVERSE:
        return pd.DataFrame(columns=PRICE_COLUMNS), "INVALID", f"Unknown instrument: {symbol}"
    if force_demo:
        return demo_prices(symbol), "DEMO", "Demo mode selected by operator"
    try:
        import yfinance as yf
        data = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(data, pd.DataFrame) and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not set(PRICE_COLUMNS).issubset(data.columns):
                return demo_prices(symbol), "DEMO", "Live feed was missing required OHLCV fields"
            data = data[PRICE_COLUMNS].dropna()
            if len(data) > 20 and data["Close"].notna().all():
                return data, "LIVE", None
    except Exception as exc:
        return demo_prices(symbol), "DEMO", str(exc)
    return demo_prices(symbol), "DEMO", "Live feed returned no usable rows"


def snapshot(symbol: str, frame: pd.DataFrame) -> dict[str, Any]:
    close = frame["Close"].astype(float)
    last = float(close.iloc[-1])
    prior = float(close.iloc[-2]) if len(close) > 1 else last
    week = float(close.iloc[-6]) if len(close) > 5 else float(close.iloc[0])
    return {"symbol": symbol, "name": ASSET_UNIVERSE.get(symbol, {}).get("name", symbol), "price": last, "day_change": (last / prior - 1) * 100 if prior else 0, "week_change": (last / week - 1) * 100 if week else 0, "volume": float(frame["Volume"].iloc[-1]), "updated": datetime.now(timezone.utc).strftime("%H:%M UTC")}


def get_news(symbol: str) -> list[dict[str, str]]:
    subject = ASSET_UNIVERSE.get(symbol, {}).get("name", symbol)
    return [
        {"time": "now", "source": "Demo context", "title": f"Synthetic research cue: {subject} is being evaluated against its short-term trend", "tone": "Constructive"},
        {"time": "now", "source": "Demo context", "title": "Synthetic research cue: rates and dollar are macro variables to verify", "tone": "Watch"},
        {"time": "now", "source": "Demo context", "title": f"Synthetic research cue: validate liquidity and event risk around {symbol}", "tone": "Neutral"},
        {"time": "now", "source": "Demo context", "title": "No live news or event API is connected in this build", "tone": "Catalyst"},
    ]


def format_price(value: float) -> str:
    if value >= 1000:
        return f"${value:,.0f}"
    if value >= 10:
        return f"${value:,.2f}"
    return f"${value:,.4f}"
