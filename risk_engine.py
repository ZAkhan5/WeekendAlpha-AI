"""Transparent position sizing and portfolio risk calculations."""

from __future__ import annotations

import pandas as pd


def calculate(account: float, risk_pct: float, entry: float, stop: float, confidence: int, volatility: float) -> dict[str, float | str]:
    risk_budget = account * risk_pct / 100
    per_unit = abs(entry - stop)
    units = risk_budget / per_unit if per_unit else 0
    raw_notional = units * entry
    volatility_multiplier = max(0.35, min(1.0, 0.22 / max(volatility, 0.01)))
    confidence_multiplier = 0.65 + confidence / 300
    notional = raw_notional * volatility_multiplier * confidence_multiplier
    final_units = notional / entry if entry else 0
    return {"risk_budget": risk_budget, "units": final_units, "notional": notional, "stop_distance": per_unit, "risk_multiple": notional * (per_unit / entry) / account if account and entry else 0, "note": "Sizing is advisory only. Review liquidity, slippage, and correlation before acting."}


def portfolio_metrics(positions: pd.DataFrame) -> dict[str, float]:
    if positions.empty:
        return {"gross": 0, "net": 0, "risk": 0, "positions": 0}
    return {"gross": float(positions["Notional"].abs().sum()), "net": float(positions["Notional"].sum()), "risk": float(positions["Risk"].sum()), "positions": int(len(positions))}


def paper_trade_plan(entry: float, invalidation: float, target: float, direction: str, scenarios: list[dict[str, object]]) -> dict[str, float | str]:
    """Summarize a hypothetical setup without creating or submitting an order."""
    risk_distance = abs(entry - invalidation)
    reward_distance = abs(target - entry)
    expected_target = sum(float(item["target"]) * float(item["prob"]) for item in scenarios) / 100 if scenarios else target
    sign = -1 if direction.upper() == "BEARISH" else 1
    expected_return = sign * (expected_target - entry) / entry if entry else 0.0
    risk_pct = risk_distance / entry if entry else 0.0
    reward_risk = reward_distance / risk_distance if risk_distance else 0.0
    gate = "PASS" if reward_risk >= 1.5 and expected_return > 0 else "REVIEW"
    return {
        "risk_pct": risk_pct * 100,
        "reward_risk": reward_risk,
        "expected_return_pct": expected_return * 100,
        "expected_target": expected_target,
        "gate": gate,
        "note": "Paper setup only. No order, wallet, or broker action is connected.",
    }
