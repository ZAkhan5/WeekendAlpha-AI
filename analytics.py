"""Technical indicators, regime classification, thesis, and scenario helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _normalise_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = {"SMA20", "SMA50", "RSI", "Volatility", "Returns"}
    if required.issubset(set(data.columns)):
        return data.copy()
    return enrich(data)


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float)
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()
    data["EMA12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    true_range = pd.concat([data["High"] - data["Low"], (data["High"] - close.shift()).abs(), (data["Low"] - close.shift()).abs()], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()
    data["Returns"] = close.pct_change()
    data["Volatility"] = data["Returns"].rolling(20).std() * np.sqrt(365)
    return data


def classify_regime(data: pd.DataFrame) -> dict[str, str | float]:
    if data.empty:
        return {"regime": "Unavailable", "bias": "NEUTRAL", "color": "#f3c969", "confidence": 0.0, "volatility": 0.0}
    row = data.iloc[-1]
    price, sma20, sma50 = float(row["Close"]), float(row["SMA20"]), float(row["SMA50"])
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    vol = float(row["Volatility"]) if pd.notna(row["Volatility"]) else 0.25
    if price > sma20 > sma50 and rsi < 72:
        regime, bias, color = "Risk-on trend", "BULLISH", "#43d17d"
    elif price < sma20 < sma50 and rsi > 28:
        regime, bias, color = "Risk-off trend", "BEARISH", "#ff6677"
    else:
        regime, bias, color = "Range / transition", "NEUTRAL", "#f3c969"
    confidence = min(94, max(54, 62 + abs(price / sma50 - 1) * 280 + abs(rsi - 50) * 0.3))
    return {"regime": regime, "bias": bias, "color": color, "confidence": confidence, "volatility": vol}


def format_data_freshness(status: str, detail: str) -> str:
    status_upper = (status or "").upper()
    if status_upper == "LIVE":
        clean_detail = detail.strip() if detail else "Updated just now"
        return f"LIVE\n{clean_detail}"
    if status_upper == "DEMO":
        return "DEMO\nDeterministic demo dataset"
    return "UNAVAILABLE\nNo usable market feed"


def build_research_snapshot(
    symbol: str,
    price: float,
    regime: str,
    bias: str,
    confidence: int,
    data_freshness: str,
    ai_engine: str,
    core_view: str,
    status: str,
    session: str | None = None,
) -> dict[str, str | float | int]:
    resolved_session = session or ("Tokenized 24/7" if symbol in {"BTC-USD", "ETH-USD", "SOL-USD"} else "US Session")
    return {
        "instrument": symbol,
        "price": float(price),
        "session": resolved_session,
        "regime": regime,
        "bias": bias,
        "confidence": int(confidence),
        "data_freshness": data_freshness,
        "ai_engine": ai_engine,
        "core_view": core_view,
        "status": status,
    }


def build_research_chain(data: pd.DataFrame, regime_name: str) -> list[dict[str, str]]:
    frame = _normalise_frame(data)
    row = frame.iloc[-1]
    price = float(row["Close"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    direction = "uptrend" if price >= sma20 else "range" if abs(price - sma20) < 0.02 * sma20 else "pullback"
    return [
        {"label": "MARKET", "purpose": "Price context", "detail": f"Price ${price:,.2f} vs 20D ${sma20:,.2f} and 50D ${sma50:,.2f}."},
        {"label": "SIGNALS", "purpose": "Trend and momentum", "detail": f"RSI {rsi:.0f} and regime {regime_name.lower()} frame the live signal quality."},
        {"label": "EVIDENCE", "purpose": "Observed facts", "detail": f"Observed readings are limited to the available OHLCV and indicator values in the desk context."},
        {"label": "AI RESEARCH", "purpose": "Analytical synthesis", "detail": f"Qwen interprets only the supplied market context and does not invent external drivers."},
        {"label": "THESIS", "purpose": "Decision narrative", "detail": f"Current evidence supports a {direction} bias under a {regime_name.lower()} regime."},
        {"label": "SCENARIOS", "purpose": "Research cases", "detail": "Upside, base, and downside views remain research scenarios, not guaranteed probabilities."},
        {"label": "INVALIDATION", "purpose": "Break conditions", "detail": "A break in trend, volatility expansion, or a regime shift would challenge the thesis."},
        {"label": "HUMAN DECISION", "purpose": "Final judgment", "detail": "The operator decides whether evidence is sufficient to act or defer."},
    ]


def build_research_gaps(source_map: dict[str, bool]) -> list[str]:
    labels = {
        "live_news": "Live news context unavailable.",
        "options_flow": "Options flow data not connected.",
        "order_book": "Order-book data not connected.",
        "macro_data": "Macro data not connected.",
    }
    return [labels[key] for key in labels if not source_map.get(key, True)]


def build_signal_alignment(data: pd.DataFrame, regime_name: str) -> dict[str, Any]:
    frame = _normalise_frame(data)
    row = frame.iloc[-1]
    price = float(row["Close"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    volume_ratio = float(row["Volume"]) / float(frame["Volume"].rolling(20).mean().iloc[-1]) if pd.notna(frame["Volume"].rolling(20).mean().iloc[-1]) else 1.0
    vol = float(row["Volatility"]) if pd.notna(row["Volatility"]) else 0.25

    trend = "supportive" if price > sma20 > sma50 else "conflicting" if price < sma20 < sma50 else "neutral"
    momentum = "supportive" if 45 <= rsi <= 70 else "conflicting" if rsi < 35 or rsi > 75 else "neutral"
    volume = "supportive" if volume_ratio >= 1.0 else "conflicting" if volume_ratio < 0.8 else "neutral"
    volatility_state = "supportive" if vol <= 0.75 else "conflicting" if vol >= 1.2 else "neutral"
    regime = "supportive" if "trend" in regime_name.lower() else "neutral" if "range" in regime_name.lower() else "conflicting"

    alignment = {
        "trend": trend,
        "momentum": momentum,
        "volume": volume,
        "volatility": volatility_state,
        "regime": regime,
    }
    conflicting = sum(1 for value in alignment.values() if value == "conflicting")
    alignment["signal_conflict"] = conflicting >= 1
    alignment["summary"] = "Signal conflict detected" if alignment["signal_conflict"] else "Signal alignment is orderly in the current research window"
    return alignment


def build_why_this_view(data: pd.DataFrame, regime_name: str, bias: str) -> dict[str, Any]:
    frame = _normalise_frame(data)
    row = frame.iloc[-1]
    price = float(row["Close"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    volume_ratio = float(row["Volume"]) / float(frame["Volume"].rolling(20).mean().iloc[-1]) if pd.notna(frame["Volume"].rolling(20).mean().iloc[-1]) else 1.0
    vol = float(row["Volatility"]) if pd.notna(row["Volatility"]) else 0.25

    trend_state = "SUPPORTIVE" if price > sma20 > sma50 else "MIXED" if abs(price - sma20) < 0.03 * sma20 else "CONFLICTING"
    momentum_state = "SUPPORTIVE" if 45 <= rsi <= 70 else "MIXED" if 35 <= rsi <= 75 else "CONFLICTING"
    volume_state = "SUPPORTIVE" if volume_ratio >= 1.0 else "MIXED" if 0.8 <= volume_ratio < 1.0 else "CONFLICTING"
    volatility_state = "ELEVATED" if vol >= 0.8 else "CONTAINED"
    regime_state = "SUPPORTIVE" if "trend" in regime_name.lower() else "MIXED" if "range" in regime_name.lower() else "CONFLICTING"

    items = [
        {"label": "TREND", "state": trend_state},
        {"label": "MOMENTUM", "state": momentum_state},
        {"label": "VOLUME", "state": volume_state},
        {"label": "VOLATILITY", "state": volatility_state},
        {"label": "REGIME", "state": regime_state},
    ]
    supportive_count = sum(1 for item in items if item["state"] == "SUPPORTIVE")
    signal_agreement = f"Signal agreement: {supportive_count} / {len(items)}"
    if bias.upper() == "BULLISH" and trend_state == "SUPPORTIVE":
        rationale = "Trend and momentum are supportive, but the desk should still watch confirmation in volume and invalidation." 
    else:
        rationale = "The view is built from observed data and remains conditional on confirmation and invalidation." 
    return {"items": items, "agreement": signal_agreement, "rationale": rationale}


def rank_opportunities(universe: dict[str, pd.DataFrame], primary_symbol: str, regime_name: str) -> list[dict[str, Any]]:
    """Score each asset on a 0-100 opportunity scale for desk ranking."""
    ranked: list[dict[str, Any]] = []
    for symbol, frame in universe.items():
        if frame is None or frame.empty:
            continue
        normalized = _normalise_frame(frame)
        row = normalized.iloc[-1]
        close = float(row["Close"])
        sma20 = float(row["SMA20"])
        sma50 = float(row["SMA50"])
        rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
        volume_ratio = float(row["Volume"]) / float(normalized["Volume"].rolling(20).mean().iloc[-1]) if pd.notna(normalized["Volume"].rolling(20).mean().iloc[-1]) else 1.0
        volatility = float(row["Volatility"]) if pd.notna(row["Volatility"]) else 0.25
        state = classify_regime(normalized)
        trend_score = 40 if close > sma20 > sma50 else 20 if close > sma50 else 15
        momentum_score = max(10, min(35, 35 - abs(rsi - 55) * 0.7))
        volume_score = 15 if volume_ratio >= 1.0 else 8 if volume_ratio >= 0.8 else 4
        volatility_score = 10 if volatility <= 0.75 else 7 if volatility <= 1.0 else 4
        regime_score = 15 if "trend" in str(regime_name).lower() and state["regime"] == regime_name else 12 if "range" in str(regime_name).lower() else 8
        score = min(100.0, trend_score + momentum_score + volume_score + volatility_score + regime_score)
        ranked.append({
            "symbol": symbol,
            "name": str(symbol),
            "bias": str(state["bias"]),
            "regime": str(state["regime"]),
            "score": round(score, 1),
            "confidence": round(float(state["confidence"]), 1),
            "is_primary": symbol == primary_symbol,
        })
    ranked.sort(key=lambda item: (item["is_primary"], item["score"]), reverse=True)
    return ranked[:5]


def summarize_backtest(data: pd.DataFrame, horizon: int = 14) -> dict[str, float | int]:
    """Return a compact signal-quality summary from recent price action."""
    frame = _normalise_frame(data)
    if frame.empty or len(frame) < horizon + 5:
        return {"signal_count": 0, "win_rate": 0.0, "avg_return": 0.0, "best_return": 0.0, "worst_return": 0.0}
    close = frame["Close"].astype(float)
    future_returns = (close.shift(-horizon) / close) - 1
    signal_mask = (close > frame["SMA20"]) & (frame["RSI"].fillna(50) > 45) & (frame["RSI"].fillna(50) < 70)
    signal_returns = future_returns[signal_mask].dropna()
    if signal_returns.empty:
        return {"signal_count": 0, "win_rate": 0.0, "avg_return": 0.0, "best_return": 0.0, "worst_return": 0.0}
    wins = (signal_returns > 0).sum()
    return {
        "signal_count": int(len(signal_returns)),
        "win_rate": round(float(wins / len(signal_returns)), 3),
        "avg_return": round(float(signal_returns.mean()), 4),
        "best_return": round(float(signal_returns.max()), 4),
        "worst_return": round(float(signal_returns.min()), 4),
    }


def build_thesis(symbol: str, data: pd.DataFrame, regime: dict[str, str | float]) -> dict[str, object]:
    frame = _normalise_frame(data)
    row = frame.iloc[-1]
    price = float(row["Close"])
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    trend = "above" if price >= float(row["SMA50"]) else "below"
    thesis = f"{symbol} is trading {trend} its 50-day trend with momentum reading {rsi:.0f}. The current {regime['regime'].lower()} regime favors defined-risk exposure and patience around confirmation."
    evidence = [
        f"Price is {abs(price / float(row['SMA50']) - 1) * 100:.1f}% from the 50-day average.",
        f"RSI at {rsi:.0f} suggests {'room before overbought' if rsi < 65 else 'momentum is extended'}.",
        f"Annualized realized volatility is {float(regime['volatility']) * 100:.1f}%.",
        f"Volume is {float(row['Volume']) / data['Volume'].rolling(20).mean().iloc[-1]:.2f}x its 20-day average.",
    ]
    invalidation = f"A sustained close {'below' if trend == 'above' else 'above'} the 50-day average with expanding volume would invalidate the setup."
    bull_case = f"A close above the recent range with RSI holding below 70 would support {symbol} trend continuation."
    bear_case = f"A close below the 50-day average with expanding volume would weaken the {symbol} setup."
    why_now = f"Price is {abs(price / float(row['SMA50']) - 1) * 100:.1f}% from its 50-day average while RSI is {rsi:.0f}; this is a defined observation point, not a prediction."
    return {"headline": f"A measured {str(regime['bias']).lower()} setup with a clear line in the sand", "thesis": thesis, "evidence": evidence, "invalidation": invalidation, "bull_case": bull_case, "bear_case": bear_case, "why_now": why_now}


def scenarios(data: pd.DataFrame, horizon: int, confidence: int) -> list[dict[str, object]]:
    frame = _normalise_frame(data)
    last = float(frame["Close"].iloc[-1])
    rolling_vol = frame["Returns"].rolling(20).std().iloc[-1]
    vol = float(rolling_vol) if pd.notna(rolling_vol) else 0.02
    move = max(0.025, vol * np.sqrt(horizon))
    confidence = max(0, min(100, int(confidence)))
    upside = max(20, min(60, round(confidence * 0.65)))
    downside = max(15, min(35, round((100 - confidence) * 0.55 + 10)))
    base = 100 - upside - downside
    if base < 15:
        downside = 100 - upside - 15
        base = 15
    return [
        {"label": "Upside case", "prob": upside, "target": last * (1 + move * 1.4), "color": "#43d17d", "detail": "Trend continuation with improving breadth and supportive flows."},
        {"label": "Base case", "prob": base, "target": last * (1 + move * 0.1), "color": "#f3c969", "detail": "Range formation while the desk waits for a higher-quality catalyst."},
        {"label": "Downside case", "prob": downside, "target": last * (1 - move * 1.3), "color": "#ff6677", "detail": "Risk-off impulse breaks the short-term trend and forces de-risking."},
    ]
