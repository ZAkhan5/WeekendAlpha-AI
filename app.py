from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from analytics import (
    build_research_chain,
    build_research_gaps,
    build_research_snapshot,
    build_signal_alignment,
    build_thesis,
    build_why_this_view,
    classify_regime,
    enrich,
    format_data_freshness,
    rank_opportunities,
    scenarios,
    summarize_backtest,
)
from data_layer import ASSET_UNIVERSE, fetch_prices, format_price, get_news, snapshot
from qwen_client import answer_natural_language_query, build_research_context, qwen_cache_identity, run_qwen_research
from risk_engine import calculate, paper_trade_plan, portfolio_metrics
from stress_test import find_similar_historical_setups

st.set_page_config(page_title="WeekendAlpha AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#080a0e; --panel:#10141b; --panel-2:#151a22; --line:#27303c; --muted:#7f8b9d; --ink:#edf2f7; --green:#43d17d; --red:#ff6677; --yellow:#f3c969; --blue:#72a7ff; }
.stApp { background: radial-gradient(circle at 82% -5%, #1a2735 0, #080a0e 38%), var(--bg); color:var(--ink); font-family:'Space Grotesk', sans-serif; }
[data-testid='stSidebar'] { background:#0c1016; border-right:1px solid var(--line); }
[data-testid='stSidebar'] * { font-family:'Space Grotesk', sans-serif; }
.block-container { padding: 1.3rem 2.2rem 3rem; max-width: 1600px; }
.mono, code { font-family:'DM Mono', monospace !important; }
.eyebrow { color:var(--green); font:500 11px 'DM Mono'; letter-spacing:.13em; text-transform:uppercase; }
.brand { font-size:28px; font-weight:700; letter-spacing:-.06em; margin:0; }
.brand span { color:var(--green); }
.subtle { color:var(--muted); font-size:13px; }
.rule { border-top:1px solid var(--line); margin:17px 0; }
.panel { background:linear-gradient(145deg, rgba(21,26,34,.96), rgba(14,18,24,.96)); border:1px solid var(--line); border-radius:8px; padding:18px; height:100%; }
.hero { background:linear-gradient(110deg, #10251d 0%, #111922 55%, #151b28 100%); border:1px solid #315541; border-radius:10px; padding:22px 24px; margin:4px 0 18px; }
.hero-title { font-size:32px; font-weight:700; letter-spacing:-.04em; margin:4px 0 8px; }
.hero-title span { color:var(--green); }
.hero-copy { max-width:760px; color:#b5c0cc; font-size:14px; line-height:1.5; }
.hero-route { color:#8ee5ad; font:11px 'DM Mono'; letter-spacing:.08em; margin-top:17px; }
.panel-title { display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; font-weight:600; font-size:15px; }
.tag { display:inline-flex; padding:4px 8px; border:1px solid #365044; background:#15291f; border-radius:4px; color:var(--green); font:500 10px 'DM Mono'; letter-spacing:.06em; }
.quote { font:600 29px 'DM Mono'; letter-spacing:-.06em; }
.up { color:var(--green); } .down { color:var(--red); } .warn { color:var(--yellow); }
.ticker { display:flex; gap:24px; overflow:hidden; border-block:1px solid var(--line); padding:9px 0; margin:4px 0 22px; white-space:nowrap; }
.ticker-item { font:12px 'DM Mono'; color:var(--muted); } .ticker-item b { color:var(--ink); margin-right:6px; }
.section-label { font:600 18px 'Space Grotesk'; margin: 24px 0 12px; }
.metric-card { background:#0d1218; border:1px solid var(--line); border-radius:6px; padding:14px; }
.metric-label { color:var(--muted); font:10px 'DM Mono'; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { margin-top:7px; font:600 20px 'DM Mono'; }
.story { border-bottom:1px solid var(--line); padding:10px 0; } .story:last-child { border:0; }
.story-title { font-size:13px; line-height:1.35; margin:4px 0; } .story-meta { color:var(--muted); font:10px 'DM Mono'; }
.signal { border-left:2px solid var(--green); padding:4px 0 4px 12px; margin:8px 0; font-size:13px; line-height:1.45; }
.signal.red { border-color:var(--red); } .signal.yellow { border-color:var(--yellow); }
.pulse { background:linear-gradient(105deg, #12251d, #111922 58%, #18202b); border:1px solid #315541; border-radius:8px; padding:15px 18px; margin:18px 0 20px; }
.pulse-flow { display:flex; align-items:center; gap:7px; overflow:auto; padding-top:10px; white-space:nowrap; }
.pulse-node { color:var(--muted); font:11px 'DM Mono'; border:1px solid var(--line); padding:7px 9px; border-radius:4px; }
.pulse-node.active { color:var(--green); border-color:var(--green); background:#102b20; }
.pulse-arrow { color:#546071; font:13px 'DM Mono'; }
.research-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px 22px; }
.research-item { border-top:1px solid var(--line); padding-top:9px; }
.research-item b { display:block; color:var(--muted); font:10px 'DM Mono'; letter-spacing:.08em; text-transform:uppercase; margin-bottom:5px; }
.source-row { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid var(--line); padding:8px 0; font-size:12px; }
.source-row:last-child { border:0; }
@media (max-width: 800px) { .block-container { padding:1rem .8rem 2rem; } .research-grid { grid-template-columns:1fr; } .pulse-flow { gap:4px; } .pulse-node { font-size:9px; padding:6px; } .pulse-arrow { font-size:10px; } }
.foot { color:#596576; font:10px 'DM Mono'; text-align:center; padding-top:18px; }
div[data-testid='stTabs'] button { color:var(--muted); } div[data-testid='stTabs'] button[aria-selected='true'] { color:var(--green); }
.stButton button { border:1px solid var(--line); background:#151b23; color:var(--ink); border-radius:5px; }
.stButton button:hover { border-color:var(--green); color:var(--green); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_universe(period: str, demo_mode: bool) -> dict[str, tuple[pd.DataFrame, str, str | None]]:
    universe: dict[str, tuple[pd.DataFrame, str, str | None]] = {}
    for ticker in ASSET_UNIVERSE:
        prices, source, issue = fetch_prices(ticker, period, force_demo=demo_mode)
        universe[ticker] = (enrich(prices), source, issue)
    return universe


@st.cache_data(ttl=300, show_spinner=False)
def load_ai_research(context_json: str, config_identity: str) -> tuple[dict | None, str]:
    return run_qwen_research(json.loads(context_json))


def metric_card(label: str, value: str, delta: str = "", tone: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {tone}">{value}</div><div class="subtle mono">{delta}</div></div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="eyebrow">PRIVATE RESEARCH TERMINAL</div><p class="brand">Weekend<span>Alpha</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">AI-assisted market intelligence.<br>Human decision, always.</p>', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Desk Controls</div>', unsafe_allow_html=True)
    symbol = st.selectbox("Primary instrument", list(ASSET_UNIVERSE), format_func=lambda item: f"{item}  ·  {ASSET_UNIVERSE[item]['name']}")
    timeframe = st.select_slider("Lookback", options=["1mo", "3mo", "6mo", "1y"], value="6mo")
    demo_mode = st.checkbox("Use deterministic demo data", value=False, help="Forces synthetic OHLCV data and marks the affected market surfaces as DEMO DATA.")
    account_size = st.number_input("Reference capital", min_value=1000.0, value=100000.0, step=5000.0, format="%.0f")
    risk_pct = st.slider("Max risk / idea", 0.25, 3.0, 1.0, 0.25)
    query_text = st.text_input("Ask a research question", placeholder="Is BTC overbought? / trend kya hai?", key="research_query")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Operating mode</div><p class="subtle">◉ Advisory only<br>◉ No brokerage connection<br>◉ No autonomous execution</p>', unsafe_allow_html=True)
    if st.button("↻  Refresh market data", width="stretch"):
        load_universe.clear()
        st.rerun()

market = load_universe(timeframe, demo_mode)
data, source, issue = market[symbol]
if data.empty:
    st.error(f"No usable market data is available for {symbol}.")
    st.caption("The instrument was rejected or its OHLCV feed was unavailable. Choose another supported symbol or enable deterministic demo data.")
    st.stop()

state = classify_regime(data)
thesis = build_thesis(symbol, data, state)
snap = snapshot(symbol, data)
row = data.iloc[-1]
news = get_news(symbol)
is_demo = source == "DEMO"
has_demo = any(asset_source == "DEMO" for _, asset_source, _ in market.values())

research_context = build_research_context(symbol, data, source, snap["updated"])
qwen_research, ai_status = load_ai_research(json.dumps(research_context, sort_keys=True, allow_nan=False), qwen_cache_identity())
if qwen_research:
    state = {**state, "bias": qwen_research["directional_bias"], "confidence": qwen_research["research_confidence"], "regime": qwen_research["market_regime"]}
    thesis = {
        "headline": "Qwen research view · validated structured output",
        "thesis": qwen_research["thesis"],
        "why_now": qwen_research["why_now"],
        "bull_case": qwen_research["bull_case"],
        "bear_case": qwen_research["bear_case"],
        "invalidation": qwen_research["key_invalidation"],
        "evidence": qwen_research["evidence"],
    }

market_window = "US Session" if symbol in {"SPY", "QQQ", "NVDA"} else "Tokenized 24/7"
market_meaning = "Traditional equity session flow dominates the research window; crypto and tokenized pairs remain continuous but are still read against the same technical discipline." if market_window == "US Session" else "This instrument trades continuously; the desk treats the current tape as 24/7 research, while still distinguishing session structure from trend evidence."

if is_demo:
    data_status = "DEMO"
elif source == "LIVE":
    data_status = "LIVE"
else:
    data_status = "UNAVAILABLE"

freshness_label = format_data_freshness(data_status, f"Updated {snap['updated']}")

if float(state["confidence"]) >= 70 and float(row["Close"]) >= float(row["SMA20"]) >= float(row["SMA50"]):
    thesis_status = "Active"
elif float(state["confidence"]) <= 50:
    thesis_status = "Mixed"
elif float(row["Close"]) < float(row["SMA50"]) and float(row["RSI"]) < 40:
    thesis_status = "Invalidated"
else:
    thesis_status = "Mixed"

research_snapshot = build_research_snapshot(
    symbol=symbol,
    price=float(snap["price"]),
    regime=str(state["regime"]),
    bias=str(state["bias"]),
    confidence=int(float(state["confidence"])),
    data_freshness=freshness_label,
    ai_engine=ai_status,
    core_view=thesis["thesis"],
    status=thesis_status,
    session=market_window,
)

st.markdown('<div class="eyebrow">WeekendAlpha research snapshot</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero"><div class="eyebrow">DECISION SUPPORT · {html.escape(ai_status)}</div><div class="hero-title"><span>{symbol}</span> research, made reviewable.</div><div class="hero-copy">Move from observed market data to a thesis, stress-tested historical context, and a paper-only risk review. Every output stays explainable and human-controlled.</div><div class="hero-route">OBSERVE  →  INTERPRET  →  STRESS TEST  →  PAPER REVIEW</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'''<div class="panel"><div class="panel-title"><span>{symbol}</span><span class="tag">{ai_status}</span></div><div class="research-grid"><div class="research-item"><b>Instrument</b>{research_snapshot["instrument"]}</div><div class="research-item"><b>Price</b>{format_price(float(research_snapshot["price"]))}</div><div class="research-item"><b>Session</b>{research_snapshot["session"]}</div><div class="research-item"><b>Regime</b>{research_snapshot["regime"]}</div><div class="research-item"><b>Bias</b>{research_snapshot["bias"]}</div><div class="research-item"><b>Confidence</b>{research_snapshot["confidence"]}/100</div><div class="research-item"><b>Data freshness</b>{research_snapshot["data_freshness"]}</div><div class="research-item"><b>AI engine</b>{research_snapshot["ai_engine"]}</div></div><div class="rule"></div><div class="signal">CORE VIEW · {html.escape(str(research_snapshot["core_view"]))}</div><div class="signal yellow">THESIS STATUS · {research_snapshot["status"]}</div></div>''',
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="pulse"><div class="panel-title" style="margin:0"><span>24/7 MARKET PULSE</span><span class="tag">RESEARCH CONTINUITY</span></div><div class="subtle">CURRENT MARKET WINDOW · """
    + market_window
    + """</div><div class="pulse-flow"><span class="pulse-node">US Session</span><span class="pulse-arrow">→</span><span class="pulse-node">After Hours</span><span class="pulse-arrow">→</span><span class="pulse-node">Overnight</span><span class="pulse-arrow">→</span><span class="pulse-node">Weekend</span><span class="pulse-arrow">→</span><span class="pulse-node active">"""
    + ("Tokenized 24/7" if market_window == "Tokenized 24/7" else "US Session")
    + """</span></div><div class="signal yellow">WHAT THIS MEANS FOR RESEARCH · """
    + html.escape(market_meaning)
    + """</div></div>""",
    unsafe_allow_html=True,
)

ticker_items = []
for ticker in ["BTC-USD", "ETH-USD", "SPY", "QQQ", "NVDA"]:
    ticker_data, _, _ = market[ticker]
    ticker_snap = snapshot(ticker, ticker_data)
    tone = "up" if ticker_snap["day_change"] >= 0 else "down"
    ticker_items.append(f'<span class="ticker-item"><b>{ticker}</b><span class="{tone}">{format_price(ticker_snap["price"])} {ticker_snap["day_change"]:+.2f}%</span></span>')
st.markdown(f'<div class="ticker">{"".join(ticker_items)}</div>', unsafe_allow_html=True)

if is_demo:
    st.warning("DEMO DATA · Synthetic, deterministic market data is active. It is not a live quote and must not be presented as one.")
elif has_demo:
    st.warning("MIXED DATA · At least one watchlist instrument is using DEMO DATA because its live feed was unavailable.")
if issue:
    st.caption(f"Data note: {html.escape(issue)}")

if query_text.strip():
    query_context = {**research_context, "market_regime": state["regime"], "directional_bias": state["bias"]}
    query_answer = answer_natural_language_query(query_text, query_context)
    st.markdown('<div class="section-label">Research answer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel"><div class="panel-title"><span>QUESTION · {html.escape(query_text)}</span><span class="tag">{int(query_answer["confidence"])}% CONFIDENCE</span></div><div style="font-size:16px;line-height:1.5">{html.escape(query_answer["answer"])}</div><div class="subtle mono" style="margin-top:12px">REFERENCED · {html.escape(", ".join(query_answer["referenced_metrics"]) or "No metric matched")}</div></div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1.55, 1], gap="large")
with left:
    chart_tab, levels_tab = st.tabs(["Price & momentum", "Signal matrix"])
    with chart_tab:
        price_tone = "up" if snap["day_change"] >= 0 else "down"
        source_label = "DEMO DATA" if is_demo else "LIVE"
        st.markdown(f'<div class="panel"><div class="panel-title"><span>{symbol} · {ASSET_UNIVERSE[symbol]["name"]}</span><span class="tag">{source_label}</span></div><div class="quote">{format_price(snap["price"])} <span class="{price_tone}" style="font-size:14px">{snap["day_change"]:+.2f}% today</span></div></div>', unsafe_allow_html=True)
        chart_data = data[["Close", "SMA20", "SMA50"]].dropna()
        st.line_chart(chart_data, color=["#edf2f7", "#43d17d", "#72a7ff"], height=350)
        st.markdown(f'<div class="subtle mono">Last update {snap["updated"]} · {len(data)} observations · close / 20D average / 50D average</div>', unsafe_allow_html=True)
    with levels_tab:
        st.dataframe(data[["Close", "SMA20", "SMA50", "RSI", "MACD", "Volatility"]].tail(12).round(3), width="stretch", height=390)
with right:
    st.markdown('<div class="panel"><div class="panel-title"><span>Market regime</span><span class="tag">AI READ</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="quote" style="font-size:24px">{state["regime"]}</div><p class="subtle">Current directional bias</p><div class="signal">{state["bias"]} · research confidence {float(state["confidence"]):.0f}/100</div><div class="signal yellow">Realized volatility · {float(state["volatility"])*100:.1f}% annualized</div><div class="signal red">Research support only. Human remains the final decision-maker.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Desk snapshot</div>', unsafe_allow_html=True)
metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Last price", format_price(snap["price"]), f"1W {snap['week_change']:+.2f}%")
with metric_cols[1]:
    metric_card("RSI (14)", f"{float(row['RSI']):.1f}", "momentum pulse", "up" if 45 < row["RSI"] < 70 else "warn")
with metric_cols[2]:
    metric_card("Trend spread", f"{(float(row['SMA20']) / float(row['SMA50']) - 1) * 100:+.2f}%", "20D vs 50D", "up" if row["SMA20"] > row["SMA50"] else "down")
with metric_cols[3]:
    metric_card("Volume pulse", f"{float(row['Volume']) / data['Volume'].rolling(20).mean().iloc[-1]:.2f}x", "vs 20D average", "warn")

opportunity_rankings = rank_opportunities({ticker: asset[0] for ticker, asset in market.items()}, symbol, state["regime"])
backtest_summary = summarize_backtest(data, horizon=14)
st.markdown('<div class="section-label">Opportunity board</div>', unsafe_allow_html=True)
ranking_cols = st.columns(len(opportunity_rankings) or 1)
for col, entry in zip(ranking_cols, opportunity_rankings):
    with col:
        tone = "up" if entry["bias"] == "BULLISH" else "down" if entry["bias"] == "BEARISH" else "warn"
        primary = "PRIMARY IDEA" if entry["is_primary"] else "WATCHLIST"
        st.markdown(
            f'<div class="panel"><div class="panel-title" style="margin-bottom:6px"><span>{entry["symbol"]}</span><span class="tag">{primary}</span></div><div class="metric-value {tone}">{entry["score"]:.1f}/100</div><div class="subtle mono">{entry["bias"]} · confidence {entry["confidence"]:.1f}</div><div class="subtle mono">{entry["regime"]}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-label">Signal quality / backtest</div>', unsafe_allow_html=True)
backtest_cols = st.columns(5)
with backtest_cols[0]:
    metric_card("Signals", str(backtest_summary["signal_count"]))
with backtest_cols[1]:
    metric_card("Win rate", f"{backtest_summary['win_rate'] * 100:.1f}%")
with backtest_cols[2]:
    metric_card("Avg return", f"{backtest_summary['avg_return'] * 100:+.2f}%")
with backtest_cols[3]:
    metric_card("Best", f"{backtest_summary['best_return'] * 100:+.2f}%")
with backtest_cols[4]:
    metric_card("Worst", f"{backtest_summary['worst_return'] * 100:+.2f}%")

research_col, risk_col = st.columns([1.15, 0.85], gap="large")
with research_col:
    st.markdown('<div class="section-label">AI research thesis</div>', unsafe_allow_html=True)
    evidence_html = "".join(f'<div class="signal">{html.escape(item)}</div>' for item in thesis["evidence"])
    st.markdown(
        f'''<div class="panel"><div class="panel-title"><span class="eyebrow">{html.escape(thesis["headline"])}</span><span class="tag">AI ENGINE · {html.escape(ai_status)}</span></div><div class="research-grid"><div class="research-item"><b>Directional bias</b>{state["bias"]}</div><div class="research-item"><b>Research confidence /100</b>{float(state["confidence"]):.0f}/100 <span class="subtle">not a probability</span></div><div class="research-item"><b>Market regime</b>{html.escape(str(state["regime"]))}</div><div class="research-item"><b>OBSERVED</b>{html.escape(thesis["evidence"][0]) if thesis["evidence"] else 'Unavailable'}</div><div class="research-item"><b>INTERPRETED</b>{html.escape(thesis["why_now"]) if thesis.get("why_now") else 'Unavailable'}</div><div class="research-item"><b>INVALIDATION</b>{html.escape(thesis["invalidation"])}</div></div><div class="rule"></div><div style="font-size:16px;line-height:1.5">{html.escape(thesis["thesis"])}</div><div class="rule"></div><div class="subtle mono">EVIDENCE / SUPPORTING INDICATORS</div>{evidence_html}<div class="signal red">Key invalidation · {html.escape(thesis["invalidation"])}</div><p class="foot">AI provides research and decision support. Human remains the final decision-maker.</p></div>''',
        unsafe_allow_html=True,
    )
with risk_col:
    st.markdown('<div class="section-label">Risk engine</div>', unsafe_allow_html=True)
    entry = st.number_input("Reference Price", min_value=0.01, value=float(snap["price"]), format="%.4f")
    stop = st.number_input("Invalidation", min_value=0.01, value=float(snap["price"] * 0.94), format="%.4f")
    risk = calculate(account_size, risk_pct, entry, stop, int(state["confidence"]), float(state["volatility"]))
    st.markdown(
        f'<div class="panel"><div class="metric-label">THESIS → RISK → INVALIDATION</div><div class="quote" style="font-size:25px">{format_price(float(risk["notional"]))}</div><p class="subtle">{float(risk["units"]):,.3f} units · risk budget {format_price(float(risk["risk_budget"]))}</p><div class="rule"></div><div class="signal yellow">Volatility state · {float(state["volatility"])*100:.1f}% annualized</div><div class="signal yellow">Risk level · {risk_pct:.2f}% of reference capital</div><div class="signal yellow">Maximum reference risk · {format_price(float(risk["risk_budget"]))}</div><div class="signal red">Key invalidation · {html.escape(thesis["invalidation"])}</div><p class="foot">Reference framework only. Not a trade instruction.</p></div>',
        unsafe_allow_html=True,
    )

stress_indicators = {
    "rsi": float(row["RSI"]),
    "macd": float(row["MACD"]),
    "macd_signal": "bullish" if float(row["MACD"]) >= float(row["Signal"]) else "bearish",
    "sma20": float(row["SMA20"]),
    "sma50": float(row["SMA50"]),
    "volatility": float(row["Volatility"]),
}
stress_result = find_similar_historical_setups(data, stress_indicators)
st.markdown('<div class="section-label">Decision stress test</div>', unsafe_allow_html=True)
stress_summary = stress_result["summary"]
if stress_result["matches"]:
    stress_cols = st.columns(4)
    with stress_cols[0]:
        metric_card("Similar setups", str(stress_summary["match_count"]))
    with stress_cols[1]:
        metric_card("Win rate", f'{stress_summary["win_rate_pct"]:.1f}%')
    with stress_cols[2]:
        metric_card("Avg 5D", f'{stress_summary["avg_return_pct"].get("5d", 0):+.2f}%')
    with stress_cols[3]:
        metric_card("Best / worst 5D", f'{stress_summary["best_case_pct"].get("5d", 0):+.2f}% / {stress_summary["worst_case_pct"].get("5d", 0):+.2f}%')
    stress_table = pd.DataFrame(stress_result["matches"])
    stress_table["forward_returns"] = stress_table["forward_returns"].map(lambda values: " · ".join(f"{key} {value:+.2f}%" for key, value in values.items()))
    st.dataframe(stress_table, width="stretch", hide_index=True)
else:
    st.markdown('<div class="panel"><div class="signal yellow">No similar historical setups found in the available research window.</div><div class="subtle">The stress test requires matching RSI, MACD state, moving-average relationship, and volatility regime. No match is treated as uncertainty, not as a signal.</div></div>', unsafe_allow_html=True)

scenario_col, news_col = st.columns([1.15, 0.85], gap="large")
with scenario_col:
    st.markdown('<div class="section-label">Research scenario weighting</div>', unsafe_allow_html=True)
    horizon = st.slider("Scenario horizon (days)", 3, 30, 14, key="horizon")
    scenario_set = scenarios(data, horizon, int(state["confidence"]))
    if qwen_research:
        for item, key in zip(scenario_set, ("upside", "base", "downside")):
            qwen_case = qwen_research["scenario_map"][key]
            item["prob"] = qwen_case["weight"]
            item["detail"] = f'{qwen_case["condition"]} {qwen_case["reaction"]}'.strip()
    total = sum(int(item["prob"]) for item in scenario_set)
    if total != 100:
        for i in range(len(scenario_set)):
            scenario_set[i]["prob"] = int(scenario_set[i]["prob"]) if int(scenario_set[i]["prob"]) else 1
        diff = 100 - sum(int(item["prob"]) for item in scenario_set)
        scenario_set[-1]["prob"] = int(scenario_set[-1]["prob"]) + diff
    for item in scenario_set:
        st.markdown(
            f'<div class="panel" style="margin-bottom:8px;padding:12px 15px;border-left:3px solid {item["color"]}"><div class="panel-title" style="margin:0"><span>{item["label"]}</span><span class="mono">{item["prob"]}% · {format_price(float(item["target"]))}</span></div><div class="subtle">{item["detail"]}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">Paper trade station</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title"><span>HYPOTHETICAL SETUP</span><span class="tag">NO LIVE ORDERS</span></div><div class="subtle">Turn the research view into a reviewable paper setup. This does not connect to a broker or simulate fills.</div>', unsafe_allow_html=True)
    paper_cols = st.columns(4)
    with paper_cols[0]:
        paper_direction = st.selectbox("Paper stance", ["BULLISH", "BEARISH"], index=0 if str(state["bias"]) != "BEARISH" else 1, key="paper_stance")
    with paper_cols[1]:
        paper_entry = st.number_input("Entry reference", min_value=0.01, value=float(snap["price"]), format="%.4f", key="paper_entry")
    with paper_cols[2]:
        paper_invalidation = st.number_input("Invalidation", min_value=0.01, value=float(snap["price"] * (0.94 if paper_direction == "BULLISH" else 1.06)), format="%.4f", key="paper_invalidation")
    with paper_cols[3]:
        paper_target = st.number_input("Target reference", min_value=0.01, value=float(scenario_set[1]["target"]), format="%.4f", key="paper_target")
    paper_plan = paper_trade_plan(paper_entry, paper_invalidation, paper_target, paper_direction, scenario_set)
    paper_tone = "up" if paper_plan["gate"] == "PASS" else "warn"
    st.markdown(
        f'<div class="research-grid"><div class="research-item"><b>Review gate</b><span class="{paper_tone}">{paper_plan["gate"]}</span></div><div class="research-item"><b>Reward / risk</b>{float(paper_plan["reward_risk"]):.2f}R</div><div class="research-item"><b>Defined risk</b>{float(paper_plan["risk_pct"]):.2f}% to invalidation</div><div class="research-item"><b>Scenario-weighted return</b><span class="{paper_tone}">{float(paper_plan["expected_return_pct"]):+.2f}%</span></div></div><div class="signal yellow">Expected reference level · {format_price(float(paper_plan["expected_target"]))}</div><div class="signal red">{paper_plan["note"]}</div></div>',
        unsafe_allow_html=True,
    )
with news_col:
    st.markdown('<div class="section-label">News & events</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="subtle mono">LIVE NEWS FEED · NOT CONNECTED</div><div class="signal yellow">Context availability: research is limited to market data, technical indicators, and manually supplied desk context.</div><div class="signal">The research engine can use OHLCV, trend, volume, volatility, and explicit regime context.</div><div class="signal red">The research engine cannot use live news, macro feeds, order-book data, or options flow in this build.</div>', unsafe_allow_html=True)
    for story in news:
        tone_class = "up" if story["tone"] == "Constructive" else "warn" if story["tone"] in ["Watch", "Catalyst"] else ""
        st.markdown(f'<div class="story"><div class="story-meta">{story["source"]} · {story["time"]} · <span class="{tone_class}">{story["tone"]}</span></div><div class="story-title">{html.escape(story["title"])}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Research decision chain</div>', unsafe_allow_html=True)
chain = build_research_chain(data, state["regime"])
chain_html = []
for index, step in enumerate(chain):
    arrow = '<span class="pulse-arrow">↓</span>' if index < len(chain) - 1 else ''
    chain_html.append(f'<div class="pulse-node active" style="min-width:120px;display:inline-flex;align-items:center;justify-content:center;">{step["label"]}<br><span class="subtle mono" style="font-size:9px;">{step["purpose"]}</span></div>{arrow}')
st.markdown(f'<div class="panel"><div class="subtle mono">Decision flow</div><div class="pulse-flow">{"".join(chain_html)}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">WHY THIS VIEW</div>', unsafe_allow_html=True)
why = build_why_this_view(data, state["regime"], str(state["bias"]))
why_rows = "".join(f'<div class="research-item"><b>{item["label"]}</b>{item["state"]}</div>' for item in why["items"])
st.markdown(f'<div class="panel"><div class="research-grid">{why_rows}</div><div class="rule"></div><div class="signal">{html.escape(why["agreement"])}</div><div class="signal yellow">{html.escape(why["rationale"])}</div></div>', unsafe_allow_html=True)

signal_alignment = build_signal_alignment(data, state["regime"])
st.markdown('<div class="section-label">Signal alignment</div>', unsafe_allow_html=True)
st.markdown(f'<div class="panel"><div class="research-grid"><div class="research-item"><b>Trend</b>{signal_alignment["trend"]}</div><div class="research-item"><b>Momentum</b>{signal_alignment["momentum"]}</div><div class="research-item"><b>Volume</b>{signal_alignment["volume"]}</div><div class="research-item"><b>Volatility</b>{signal_alignment["volatility"]}</div><div class="research-item"><b>Regime</b>{signal_alignment["regime"]}</div><div class="research-item"><b>Summary</b>{signal_alignment["summary"]}</div></div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">What would change the thesis?</div>', unsafe_allow_html=True)
st.markdown(f'<div class="panel"><div class="signal">Market condition · {html.escape(str(state["regime"]))}</div><div class="signal">Technical condition · {html.escape(str(thesis["invalidation"]))}</div><div class="signal red">Risk condition · Volatility expansion, a failed trend break, or a regime change would materially reduce confidence.</div></div>', unsafe_allow_html=True)

research_gaps = build_research_gaps({"live_news": False, "options_flow": False, "order_book": False, "macro_data": False})
st.markdown('<div class="section-label">Research gaps</div>', unsafe_allow_html=True)
st.markdown('<div class="panel">' + ''.join(f'<div class="signal yellow">{html.escape(gap)}</div>' for gap in research_gaps) + '</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Watchlist & exposure</div>', unsafe_allow_html=True)
watch_cols = st.columns(len(ASSET_UNIVERSE))
for watch_col, ticker in zip(watch_cols, ASSET_UNIVERSE):
    ticker_data, ticker_source, _ = market[ticker]
    ticker_snap = snapshot(ticker, ticker_data)
    ticker_state = classify_regime(ticker_data)
    tone = "up" if ticker_snap["day_change"] >= 0 else "down"
    aligned_status = "aligned" if ticker_state["bias"] == state["bias"] else "mixed"
    with watch_col:
        ticker_tag = "DEMO DATA" if ticker_source == "DEMO" else "LIVE"
        st.markdown(
            f'<div class="panel"><div class="panel-title" style="margin-bottom:5px"><span class="metric-label">{ticker}</span><span class="tag">{ticker_tag}</span></div><div style="font-size:12px;margin:5px 0">{ASSET_UNIVERSE[ticker]["name"]}</div><div class="metric-value">{format_price(ticker_snap["price"])}</div><div class="{tone} mono">{ticker_snap["day_change"]:+.2f}% daily</div><div class="subtle" style="margin-top:8px">{ticker_state["regime"]} · {ticker_state["bias"]}</div><div class="subtle mono">confidence {float(ticker_state["confidence"]):.0f}/100</div><div class="subtle mono">data status {aligned_status}</div></div>',
            unsafe_allow_html=True,
        )

positions = pd.DataFrame([{"Symbol": symbol, "Notional": float(risk["notional"]), "Risk": float(risk["risk_budget"]), "View": str(state["bias"])}])
portfolio = portfolio_metrics(positions)
st.caption(f"Desk exposure: {format_price(portfolio['gross'])} gross · {format_price(portfolio['risk'])} defined risk · {portfolio['positions']} active idea. Human review required before any external action.")
source_status = "DEMO DATA" if is_demo else "MIXED LIVE/DEMO" if has_demo else "LIVE market data"
ai_source = "Qwen qwen3.8-max · validated JSON" if qwen_research else "Local deterministic research rules"
st.markdown(f'<div class="panel"><div class="panel-title"><span>Research sources / data sources</span><span class="tag">{source_status}</span></div><div class="source-row"><span>Market data</span><span class="mono">{source_status} · updated {snap["updated"]}</span></div><div class="source-row"><span>Technical analysis</span><span class="mono">Local deterministic indicators · OHLCV derived</span></div><div class="source-row"><span>News / events</span><span class="mono">No live API connected · unavailable to AI</span></div><div class="source-row"><span>AI analysis</span><span class="mono">{ai_source}</span></div></div>', unsafe_allow_html=True)

st.markdown('<div class="foot">WEEKENDALPHA AI · RESEARCH SYSTEM · NOT INVESTMENT ADVICE · NO EXECUTION CONNECTED · BUILD 0.1.0</div>', unsafe_allow_html=True)

