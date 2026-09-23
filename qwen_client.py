"""Optional Bitget Qwen research adapter.

The adapter is deliberately research-only: it sends market observations and
technical indicators, validates the returned JSON, and never exposes an order
or wallet capability to the model.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_BASE_URL = "https://hackathon.bitgetops.com/v1"
DEFAULT_MODEL = "qwen3.8-max"
BIAS_VALUES = {"BULLISH", "BEARISH", "NEUTRAL"}
SCENARIO_KEYS = ("upside", "base", "downside")


def _setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        secret_value = st.secrets.get(name, "")
    except Exception:
        secret_value = ""
    return str(secret_value).strip() or default


def qwen_configured() -> bool:
    return bool(_setting("BITGET_QWEN_API_KEY"))


def qwen_cache_identity() -> str:
    """Return a non-secret cache discriminator for configuration changes."""
    base_url = _setting("BITGET_QWEN_BASE_URL", DEFAULT_BASE_URL)
    model = _setting("BITGET_QWEN_MODEL", DEFAULT_MODEL)
    return f"configured={qwen_configured()};base={base_url};model={model}"


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if pd.notna(parsed) else default


def build_research_context(symbol: str, data: pd.DataFrame, source: str, updated: str) -> dict[str, Any]:
    """Create a compact context containing only values available in the app."""
    row = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else row
    close = _number(row.get("Close"), 0.0)
    prior_close = _number(previous.get("Close"), close) or close
    return {
        "instrument": symbol,
        "data_source": source,
        "last_update": updated,
        "news_events": "UNAVAILABLE: no live news or events API is connected.",
        "observations": {
            "close": close,
            "daily_change_pct": ((close / prior_close) - 1) * 100 if prior_close else 0.0,
            "sma20": _number(row.get("SMA20")),
            "sma50": _number(row.get("SMA50")),
            "rsi14": _number(row.get("RSI")),
            "macd": _number(row.get("MACD")),
            "atr14": _number(row.get("ATR")),
            "annualized_volatility": _number(row.get("Volatility")),
            "volume": _number(row.get("Volume")),
            "volume_vs_20d": _number(row.get("Volume")) / _number(data["Volume"].rolling(20).mean().iloc[-1], 1.0),
        },
    }


def _system_prompt() -> str:
    return """You are an AI financial research analyst for WeekendAlpha AI. Your job is research support, not autonomous trading or chatbot conversation. Use only the supplied market data, indicators, and explicit context. Never invent facts, data, news, macro conditions, orders, flows, or other market inputs.

Required behaviors:
- Distinguish OBSERVED facts from INTERPRETED conclusions and AI assessment.
- Acknowledge conflicting signals when the data is mixed.
- Explain what would invalidate the thesis.
- Keep the research concise, professional, and human-in-the-loop.
- Avoid guaranteed predictions or direct trade instructions.
- State clearly when information is unavailable.
- Research confidence is a 0-100 assessment of evidence quality and signal clarity, not a probability of profit or direction.
- Scenario weights are communication weights only and must total exactly 100.
- Do not expose secrets or credentials.

Return strict JSON only, with exactly this shape:
{"directional_bias":"BULLISH | BEARISH | NEUTRAL","research_confidence":0,"market_regime":"","thesis":"","why_now":"","bull_case":"","bear_case":"","key_invalidation":"","evidence":[""],"scenario_map":{"upside":{"weight":0,"condition":"","reaction":""},"base":{"weight":0,"condition":"","reaction":""},"downside":{"weight":0,"condition":"","reaction":""}}}

Rules:
- All text fields must be short, specific, and grounded in supplied context.
- evidence must list only observable points from the context.
- thesis should summarize the evidence and explain the current stance without claiming certainty.
- why_now must explain the current view using available technical evidence and any stated regime context.
- bull_case and bear_case explain the current directional case without trading instructions.
- key_invalidation should describe the conditions that would materially weaken the thesis.
- scenario weights must be integers from 0 to 100 and total exactly 100.
- Do not provide trade instructions, order instructions, wallet advice, brokerage actions, execution steps, or any autonomous command.
- The human remains the final decision-maker."""


def _user_prompt(context: dict[str, Any]) -> str:
    return "Analyze this structured market context. Use only the values present below; do not add news, earnings, macro events, flows, prices, or other facts.\n\n" + json.dumps(context, separators=(",", ":"), allow_nan=False)


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("Qwen content was not a JSON string")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Qwen JSON root was not an object")
    return payload


def validate_research(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("directional_bias", "research_confidence", "market_regime", "thesis", "why_now", "bull_case", "bear_case", "key_invalidation", "evidence", "scenario_map")
    if any(key not in payload for key in required):
        raise ValueError("Qwen response omitted required research fields")
    bias = str(payload["directional_bias"]).upper()
    if bias not in BIAS_VALUES:
        raise ValueError("Qwen directional bias was invalid")
    confidence = _number(payload["research_confidence"])
    if confidence is None or not 0 <= confidence <= 100:
        raise ValueError("Qwen research confidence was outside 0-100")
    evidence = payload["evidence"]
    if not isinstance(evidence, list) or not evidence or any(not isinstance(item, str) for item in evidence):
        raise ValueError("Qwen evidence was invalid")
    scenario_map = payload["scenario_map"]
    if not isinstance(scenario_map, dict) or any(key not in scenario_map for key in SCENARIO_KEYS):
        raise ValueError("Qwen scenario map was incomplete")
    normalized_scenarios: dict[str, dict[str, Any]] = {}
    weights = []
    for key in SCENARIO_KEYS:
        scenario = scenario_map[key]
        if not isinstance(scenario, dict):
            raise ValueError("Qwen scenario was not an object")
        weight = _number(scenario.get("weight"))
        if weight is None or weight < 0 or weight > 100 or int(weight) != weight:
            raise ValueError("Qwen scenario weight was invalid")
        weights.append(int(weight))
        normalized_scenarios[key] = {"weight": int(weight), "condition": str(scenario.get("condition", "")), "reaction": str(scenario.get("reaction", ""))}
    if sum(weights) != 100:
        raise ValueError("Qwen scenario weights did not total 100")
    text_fields = ("market_regime", "thesis", "why_now", "bull_case", "bear_case", "key_invalidation")
    if any(not isinstance(payload[field], str) or not payload[field].strip() for field in text_fields):
        raise ValueError("Qwen research text was incomplete")
    return {"directional_bias": bias, "research_confidence": int(round(confidence)), "market_regime": payload["market_regime"].strip(), "thesis": payload["thesis"].strip(), "why_now": payload["why_now"].strip(), "bull_case": payload["bull_case"].strip(), "bear_case": payload["bear_case"].strip(), "key_invalidation": payload["key_invalidation"].strip(), "evidence": evidence[:8], "scenario_map": normalized_scenarios}


def _request(api_key: str, base_url: str, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model, "temperature": 0, "messages": messages}, timeout=30)
    if response.status_code in {401, 403}:
        raise RuntimeError("Qwen authentication failed")
    if response.status_code == 429:
        raise RuntimeError("Qwen rate limit reached")
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _extract_json(content)


def run_qwen_research(context: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return validated Qwen research or a safe fallback reason.

    The API key is read only from the environment and is never included in
    returned errors, UI strings, or logs.
    """
    api_key = _setting("BITGET_QWEN_API_KEY")
    if not api_key:
        return None, "LOCAL FALLBACK"
    base_url = _setting("BITGET_QWEN_BASE_URL", DEFAULT_BASE_URL)
    model = _setting("BITGET_QWEN_MODEL", DEFAULT_MODEL)
    messages = [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": _user_prompt(context)}]
    try:
        return validate_research(_request(api_key, base_url, model, messages)), "Qwen LIVE"
    except Exception:
        correction = messages + [{"role": "user", "content": "Your previous response failed schema validation. Return corrected strict JSON only. Scenario weights must total exactly 100 and research_confidence must be 0-100. Use no invented facts."}]
        try:
            return validate_research(_request(api_key, base_url, model, correction)), "Qwen LIVE"
        except Exception:
            return None, "LOCAL FALLBACK"


def _fallback_query_answer(query_text: str, market_context: dict[str, Any]) -> dict[str, Any]:
    """Answer common research questions without a network call."""
    query = query_text.lower()
    observations = market_context.get("observations", market_context)
    rsi = _number(observations.get("rsi14", observations.get("RSI")), 50.0) or 50.0
    regime = str(market_context.get("market_regime", market_context.get("regime", "Unavailable")))
    bias = str(market_context.get("directional_bias", market_context.get("bias", "NEUTRAL")))
    if "overbought" in query:
        answer = f"RSI is {rsi:.1f}; overbought is typically considered above 70, so this reading is {'overbought' if rsi > 70 else 'not overbought'}."
        metrics = ["rsi14"]
    elif "oversold" in query:
        answer = f"RSI is {rsi:.1f}; oversold is typically considered below 30, so this reading is {'oversold' if rsi < 30 else 'not oversold'}."
        metrics = ["rsi14"]
    elif any(word in query for word in ("trend", "direction", "bias", "regime")):
        answer = f"The current research regime is {regime} with a {bias} directional bias. This is research context, not a trade instruction."
        metrics = ["market_regime", "directional_bias"]
    elif "volatility" in query or "volatile" in query:
        volatility = _number(observations.get("annualized_volatility", observations.get("Volatility")), 0.0) or 0.0
        answer = f"Annualized realized volatility is {volatility * 100:.1f}%."
        metrics = ["annualized_volatility"]
    else:
        answer = "I can answer questions about the supplied RSI, trend, regime, bias, price, volume, and volatility values."
        metrics = []
    return {"answer": answer, "referenced_metrics": metrics, "confidence": 72 if metrics else 45}


def validate_natural_language_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the small strict JSON contract used by the research query UI."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        raise ValueError("Natural language answer was invalid")
    metrics = payload.get("referenced_metrics")
    confidence = _number(payload.get("confidence"))
    if not isinstance(metrics, list) or any(not isinstance(item, str) for item in metrics):
        raise ValueError("Referenced metrics were invalid")
    if confidence is None or not 0 <= confidence <= 100:
        raise ValueError("Natural language confidence was outside 0-100")
    return {"answer": payload["answer"].strip(), "referenced_metrics": metrics[:12], "confidence": int(round(confidence))}


def answer_natural_language_query(query_text: str, market_context_dict: dict[str, Any]) -> dict[str, Any]:
    """Answer an English or Roman Urdu research question using supplied data only."""
    if not query_text.strip():
        return _fallback_query_answer(query_text, market_context_dict)
    api_key = _setting("BITGET_QWEN_API_KEY")
    if not api_key:
        return _fallback_query_answer(query_text, market_context_dict)
    system = "You answer financial research questions, not trade instructions. Use only supplied data. Understand English and Roman Urdu. Return strict JSON only: {\"answer\":\"\",\"referenced_metrics\":[],\"confidence\":0}. Never invent facts, news, or prices. Confidence is evidence quality, not profit probability."
    user = "Question:\n" + query_text + "\n\nMarket context:\n" + json.dumps(market_context_dict, separators=(",", ":"), allow_nan=False)
    try:
        payload = _request(api_key, _setting("BITGET_QWEN_BASE_URL", DEFAULT_BASE_URL), _setting("BITGET_QWEN_MODEL", DEFAULT_MODEL), [{"role": "system", "content": system}, {"role": "user", "content": user}])
        return validate_natural_language_response(payload)
    except Exception:
        return _fallback_query_answer(query_text, market_context_dict)
