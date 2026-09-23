"""Historical decision stress testing for explainable research setups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analytics import enrich


def _volatility_regime(value: float) -> str:
    if value >= 0.8:
        return "elevated"
    if value <= 0.35:
        return "contained"
    return "normal"


def _macd_state(macd: float, signal: float | None = None) -> str:
    if signal is not None:
        return "bullish" if macd >= signal else "bearish"
    return "bullish" if macd >= 0 else "bearish"


def find_similar_historical_setups(
    asset_data: pd.DataFrame,
    current_indicators: dict[str, Any],
    lookback_days: int = 365,
) -> dict[str, Any]:
    """Find historical rows with similar RSI, MACD, trend, and volatility state.

    The current row is excluded from matching. Forward returns are only included
    when the requested horizon is fully available, preventing look-ahead bias.
    """
    if asset_data is None or asset_data.empty:
        return {"matches": [], "summary": {"match_count": 0, "win_rate_pct": 0.0, "avg_return_pct": {}, "best_case_pct": {}, "worst_case_pct": {}}}

    frame = asset_data.copy()
    if not {"SMA20", "SMA50", "RSI", "MACD", "Signal", "Volatility"}.issubset(frame.columns):
        frame = enrich(frame)
    frame = frame.sort_index().tail(max(1, int(lookback_days)))
    current = current_indicators
    current_rsi = float(current.get("rsi", current.get("RSI", frame["RSI"].iloc[-1])))
    current_macd = float(current.get("macd", current.get("MACD", frame["MACD"].iloc[-1])))
    current_signal = current.get("macd_signal", current.get("Signal"))
    current_signal_value = float(current_signal) if isinstance(current_signal, (int, float)) else None
    current_macd_state = str(current_signal).lower() if isinstance(current_signal, str) else _macd_state(current_macd, current_signal_value)
    current_sma20 = float(current.get("sma20", current.get("SMA20", frame["SMA20"].iloc[-1])))
    current_sma50 = float(current.get("sma50", current.get("SMA50", frame["SMA50"].iloc[-1])))
    current_trend = "above" if current_sma20 >= current_sma50 else "below"
    current_vol = float(current.get("volatility", current.get("Volatility", frame["Volatility"].iloc[-1])))
    current_vol_regime = str(current.get("volatility_regime", _volatility_regime(current_vol))).lower()

    matches: list[dict[str, Any]] = []
    for position, (date, row) in enumerate(frame.iloc[:-1].iterrows()):
        values = [row.get("RSI"), row.get("MACD"), row.get("SMA20"), row.get("SMA50"), row.get("Volatility")]
        if any(pd.isna(value) for value in values):
            continue
        historical_trend = "above" if float(row["SMA20"]) >= float(row["SMA50"]) else "below"
        historical_state = _macd_state(float(row["MACD"]), float(row["Signal"]) if pd.notna(row.get("Signal")) else None)
        if abs(float(row["RSI"]) - current_rsi) > 5:
            continue
        if historical_trend != current_trend or historical_state != current_macd_state:
            continue
        if _volatility_regime(float(row["Volatility"])) != current_vol_regime:
            continue
        close = float(row["Close"])
        forward_returns: dict[str, float] = {}
        for horizon in (5, 10, 20):
            future_position = position + horizon
            if future_position < len(frame):
                future_close = float(frame["Close"].iloc[future_position])
                forward_returns[f"{horizon}d"] = (future_close / close - 1) * 100
        if forward_returns:
            matches.append({"date": date.strftime("%Y-%m-%d"), "rsi": round(float(row["RSI"]), 2), "forward_returns": {key: round(value, 3) for key, value in forward_returns.items()}})

    summary: dict[str, Any] = {"match_count": len(matches), "win_rate_pct": 0.0, "avg_return_pct": {}, "best_case_pct": {}, "worst_case_pct": {}}
    for horizon in ("5d", "10d", "20d"):
        values = [match["forward_returns"][horizon] for match in matches if horizon in match["forward_returns"]]
        if values:
            summary["avg_return_pct"][horizon] = round(float(sum(values) / len(values)), 3)
            summary["best_case_pct"][horizon] = round(float(max(values)), 3)
            summary["worst_case_pct"][horizon] = round(float(min(values)), 3)
            summary["win_rate_pct"] = round(float(sum(value > 0 for value in values) / len(values) * 100), 1)
    return {"matches": matches, "summary": summary}
