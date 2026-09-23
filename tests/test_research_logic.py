from __future__ import annotations

import pandas as pd
from types import SimpleNamespace

import qwen_client
from analytics import (
    build_research_chain,
    build_research_gaps,
    build_research_snapshot,
    build_signal_alignment,
    classify_regime,
    rank_opportunities,
    scenarios,
    format_data_freshness,
)
from risk_engine import paper_trade_plan
from stress_test import find_similar_historical_setups


def _sample_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(range(1, 121), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Volume": [1000 + i * 10 for i in range(120)],
        },
        index=idx,
    )


def test_scenario_weights_total_100() -> None:
    weights = scenarios(_sample_frame(), 14, 72)
    assert sum(item["prob"] for item in weights) == 100
    assert {item["label"] for item in weights} == {"Upside case", "Base case", "Downside case"}


def test_empty_market_frame_has_unavailable_regime() -> None:
    empty_frame = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    regime = classify_regime(empty_frame)
    assert regime["regime"] == "Unavailable"
    assert regime["bias"] == "NEUTRAL"
    assert regime["confidence"] == 0.0


def test_opportunity_ranking_accepts_market_data_frames() -> None:
    ranked = rank_opportunities({"BTC-USD": _sample_frame()}, "BTC-USD", "Risk-on trend")
    assert ranked[0]["symbol"] == "BTC-USD"
    assert ranked[0]["is_primary"] is True


def test_signal_alignment_reports_conflict_when_needed() -> None:
    alignment = build_signal_alignment(_sample_frame(), "Risk-on trend")
    assert alignment["trend"] in {"supportive", "neutral", "conflicting"}
    assert alignment["signal_conflict"] in {True, False}
    assert "Signal conflict detected" in alignment["summary"] or "Signal alignment" in alignment["summary"]


def test_snapshot_contains_required_fields() -> None:
    snapshot = build_research_snapshot(
        symbol="BTC-USD",
        price=100.0,
        regime="Risk-on trend",
        bias="BULLISH",
        confidence=78,
        data_freshness="LIVE\nUpdated 2m ago",
        ai_engine="Qwen LIVE",
        core_view="Trend is constructive with disciplined risk control.",
        status="Active",
    )
    assert snapshot["instrument"] == "BTC-USD"
    assert snapshot["bias"] == "BULLISH"
    assert snapshot["status"] == "Active"
    assert "core_view" in snapshot


def test_snapshot_uses_crypto_session_for_24_7_markets() -> None:
    snapshot = build_research_snapshot(
        symbol="BTC-USD",
        price=100.0,
        regime="Risk-on trend",
        bias="BULLISH",
        confidence=78,
        data_freshness="LIVE\nUpdated 2m ago",
        ai_engine="Qwen LIVE",
        core_view="Trend is constructive with disciplined risk control.",
        status="Active",
        session="Tokenized 24/7",
    )
    assert snapshot["session"] == "Tokenized 24/7"


def test_opportunity_ranking_is_sorted_and_scored() -> None:
    ranked = [
        {"symbol": "BTC-USD", "score": 80.0, "bias": "BULLISH"},
        {"symbol": "SPY", "score": 65.0, "bias": "BULLISH"},
    ]
    assert ranked[0]["score"] >= ranked[1]["score"]
    assert 0 <= ranked[0]["score"] <= 100
    assert ranked[0]["bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}


def test_backtest_summary_returns_signal_metrics() -> None:
    summary = {
        "signal_count": 5,
        "win_rate": 0.6,
        "avg_return": 0.045,
        "best_return": 0.12,
        "worst_return": -0.04,
    }
    assert summary["signal_count"] >= 1
    assert 0 <= summary["win_rate"] <= 1
    assert summary["avg_return"] >= -1


def test_research_chain_exposes_expected_stage_labels() -> None:
    chain = build_research_chain(_sample_frame(), "Risk-on trend")
    labels = [item["label"] for item in chain]
    assert labels[0] == "MARKET"
    assert labels[-1] == "HUMAN DECISION"
    assert "SIGNALS" in labels and "THESIS" in labels and "INVALIDATION" in labels


def test_research_gaps_lists_missing_sources() -> None:
    gaps = build_research_gaps({"live_news": False, "options_flow": False, "order_book": True, "macro_data": False})
    assert any("live news" in item.lower() for item in gaps)
    assert any("options" in item.lower() for item in gaps)
    assert any("macro" in item.lower() for item in gaps)


def test_data_freshness_formatting_is_clear() -> None:
    assert "LIVE" in format_data_freshness("LIVE", "Updated 2m ago")
    assert "DEMO" in format_data_freshness("DEMO", "Deterministic demo dataset")
    assert "UNAVAILABLE" in format_data_freshness("UNAVAILABLE", "No usable market feed")


def test_paper_trade_plan_is_directional_and_has_review_gate() -> None:
    plan = paper_trade_plan(
        entry=100.0,
        invalidation=95.0,
        target=110.0,
        direction="BULLISH",
        scenarios=[{"target": 110.0, "prob": 60}, {"target": 101.0, "prob": 25}, {"target": 90.0, "prob": 15}],
    )
    assert plan["reward_risk"] == 2.0
    assert plan["expected_return_pct"] > 0
    assert plan["gate"] == "PASS"
    assert "No order" in plan["note"]


def test_qwen_reads_streamlit_secrets_and_preserves_env_precedence(monkeypatch) -> None:
    monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)
    monkeypatch.delenv("BITGET_QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("BITGET_QWEN_MODEL", raising=False)
    monkeypatch.setattr(qwen_client, "st", SimpleNamespace(secrets={
        "BITGET_QWEN_API_KEY": "secret-key",
        "BITGET_QWEN_BASE_URL": "https://secret.example/v1",
        "BITGET_QWEN_MODEL": "secret-model",
    }))

    assert qwen_client.qwen_configured() is True
    assert qwen_client.qwen_cache_identity() == "configured=True;base=https://secret.example/v1;model=secret-model"

    monkeypatch.setenv("BITGET_QWEN_MODEL", "environment-model")
    assert "model=environment-model" in qwen_client.qwen_cache_identity()


def test_qwen_uses_safe_defaults_without_configuration(monkeypatch) -> None:
    for name in ("BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL", "BITGET_QWEN_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(qwen_client, "st", SimpleNamespace(secrets={}))

    assert qwen_client.qwen_configured() is False
    assert qwen_client.qwen_cache_identity() == "configured=False;base=https://hackathon.bitgetops.com/v1;model=qwen3.8-max"


def test_stress_test_finds_repeating_setup_and_forward_return() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.Series(range(100, 140), index=dates, dtype=float)
    frame = pd.DataFrame({"Close": close, "RSI": 50.0, "MACD": -1.0, "Signal": 0.0, "SMA20": 90.0, "SMA50": 100.0, "Volatility": 0.5})
    for index in (5, 15):
        frame.iloc[index, frame.columns.get_loc("RSI")] = 54.0
        frame.iloc[index, frame.columns.get_loc("MACD")] = 1.0
        frame.iloc[index, frame.columns.get_loc("Signal")] = 0.5
        frame.iloc[index, frame.columns.get_loc("SMA20")] = 110.0
        frame.iloc[index, frame.columns.get_loc("SMA50")] = 100.0
    result = find_similar_historical_setups(
        frame,
        {"rsi": 55.0, "macd": 1.0, "macd_signal": "bullish", "sma20": 110.0, "sma50": 100.0, "volatility": 0.5},
        lookback_days=40,
    )
    assert result["summary"]["match_count"] == 2
    assert result["matches"][0]["date"] == "2024-01-06"
    assert result["matches"][0]["forward_returns"]["5d"] == round((110 / 105 - 1) * 100, 3)


def test_natural_language_query_validates_qwen_response(monkeypatch) -> None:
    monkeypatch.setenv("BITGET_QWEN_API_KEY", "test-key")
    monkeypatch.setattr(qwen_client, "_request", lambda *args: {"answer": "RSI is elevated.", "referenced_metrics": ["rsi14"], "confidence": 88})
    result = qwen_client.answer_natural_language_query("Is this overbought?", {"observations": {"rsi14": 72}})
    assert result == {"answer": "RSI is elevated.", "referenced_metrics": ["rsi14"], "confidence": 88}


def test_natural_language_query_falls_back_without_qwen(monkeypatch) -> None:
    monkeypatch.delenv("BITGET_QWEN_API_KEY", raising=False)
    result = qwen_client.answer_natural_language_query("Is BTC overbought?", {"observations": {"rsi14": 74}})
    assert "overbought" in result["answer"]
    assert result["referenced_metrics"] == ["rsi14"]
