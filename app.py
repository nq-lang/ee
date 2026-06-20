"""
app.py — Quantitative Backtesting Terminal
==========================================
Launch: streamlit run app.py
"""

# ── Module path bootstrap (MUST run before any local imports) ─────────────────
import sys, os as _os, glob as _glob

def _add(p):
    if p and _os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# 1. Directory of this file
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_add(_HERE)

# 2. Any immediate subdirectory that contains metrics.py (handles nested repos)
for _f in _glob.glob(_os.path.join(_HERE, '*/metrics.py')):
    _add(_os.path.dirname(_f))

# 3. Parent directory (handles quant_terminal/app.py layout)
_add(_os.path.dirname(_HERE))

# 4. cwd fallback
_add(_os.getcwd())
for _f in _glob.glob(_os.path.join(_os.getcwd(), '*/metrics.py')):
    _add(_os.path.dirname(_f))

del _add, _glob, _os, _HERE
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations


import io
import json
import os
import time
import traceback
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Local modules ────────────────────────────────────────────────────────────
from config import (
    INSTRUMENTS, OPTIONS_UNDERLYINGS, DEFAULT_CONFIG, DEFAULT_COMBINE,
    DEFAULT_MONTE_CARLO, COLORS, APP_VERSION,
)
from data_loader import (
    get_futures_data, get_options_chain,
    load_csv_upload, auto_detect_instrument,
    write_normalised_for_cpp,
)
from strategy_parser import parse_cpp_strategy, demo_strategy_definition
from py_strategy import load_py_strategy, PyStrategyAdapter, PyParseReport, EXAMPLE_PY_STRATEGY
from backtest_engine import BacktestEngine, BacktestConfig, run_cpp_backtest
from metrics import compute_all_metrics
from monte_carlo import MonteCarloConfig, run_monte_carlo, METHODS
from charts import (
    equity_drawdown_chart, annual_pnl_chart, monthly_heatmap,
    monte_carlo_fan_chart, monte_carlo_histogram,
    candlestick_chart, dow_chart, tod_chart, quarterly_chart,
)
from ui_components import (
    inject_css, section_header, kpi_card_row, strategy_subtitle,
    sidebar_status_bar, checklist, terminal_log,
    signal_table, metric_grid,
    instrument_spec_card, banner_success, banner_warning,
    banner_error, banner_info,
)
from section_candlestick import render_section_candlestick
from native_data import (
    get_native_data, get_best_available, list_available,
    prewarm_small_datasets, CATALOGUE, st_cached_native,
)
from volatility_regime   import tag_trades_with_regime, compute_regime_analytics, REGIME_COLORS
from pnl_distribution    import build_pnl_distribution, build_return_stats_table
from section_montecarlo  import render_section_montecarlo
from section_remodeler   import render_section_remodeler
from multi_api_loader    import fetch_ohlcv, get_loader
from export import (
    export_trade_log_csv, export_equity_curve_csv,
    export_monte_carlo_csv, export_config_json, export_pdf_report,
)

# ────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        # Navigation
        "section":              "1. CONFIGURATION & INPUTS",
        # Data
        "raw_df":               None,
        "data_report":          None,
        "data_filename":        "none",
        # Strategy
        "parse_report":         None,
        "strategy_filename":    "none",
        "use_demo_strategy":    True,
        "cpp_source":           "",
        # Config (mirrors DEFAULT_CONFIG)
        "instrument":           "NQ",
        "starting_balance":     50_000.0,
        "num_contracts":        1,
        "sizing_mode":          "Fixed",
        "risk_per_trade_pct":   1.0,
        "commission_per_side":  0.50,
        "exchange_fee":         0.85,
        "nfa_fee":              0.02,
        "slippage_ticks":       1,
        "profit_target_mode":   "Strategy",
        "profit_target_value":  1_000.0,
        "stop_loss_mode":       "Strategy",
        "stop_loss_value":      500.0,
        "use_partial_exits":    False,
        "partial_exit_pct":     50.0,
        "max_bars_in_trade":    0,
        "eod_exit":             True,
        "eod_exit_time":        "15:45",
        "session_filter":       "Full Session",
        "day_of_week_filter":   [0, 1, 2, 3, 4],
        "date_range_mode":      "Full CSV History",
        "custom_start":         None,
        "custom_end":           None,
        "daily_loss_limit":     0.0,
        "display_scale":        1,
        "risk_free_rate":       5.0,
        # Results
        "backtest_result":      None,
        "metrics":              None,
        "last_run":             "never",
        "engine_status":        "idle",
        # Monte Carlo
        "mc_result":            None,
        "mc_config":            None,
        # Compile mode
        "compile_mode":         False,
        # Strategy type tracking
        "strategy_type":        "demo",   # "demo" | "cpp" | "python"
        "py_source":            "",
        "py_parse_report":      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
SS = st.session_state  # convenience alias


# ────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ────────────────────────────────────────────────────────────────────────────

SECTIONS = [
    "1. CONFIGURATION & INPUTS",
    "2. STRATEGY LOADER",
    "3. DATA LOADER",
    "4. BACKTEST ENGINE",
    "5. RESULTS DASHBOARD",
    "6. CANDLESTICK CHART",
    "7. MONTE CARLO SIMULATOR",
    "8. ANNUAL & PERIODIC BREAKDOWN",
    "9. SIGNAL HISTORY LOG",
    "10. RISK METRICS PANEL",
    "11. EXPORT & REPORTING",
    "12. CODE REMODELER",
]

with st.sidebar:
    st.markdown(
        f'<div style="font-family:monospace;color:{COLORS["green"]};'
        f'font-size:1.1rem;font-weight:900;letter-spacing:0.12em;'
        f'padding:8px 0 4px 0;">⬡ QUANT TERMINAL</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-family:monospace;font-size:0.65rem;'
        f'color:#444;margin-bottom:12px;">v{APP_VERSION} · Systematic Backtest Harness</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for sec in SECTIONS:
        is_active = SS["section"] == sec
        color = COLORS["green"] if is_active else COLORS["text_dim"]
        prefix = "▶ " if is_active else "  "
        if st.sidebar.button(
            f"{prefix}{sec}",
            key=f"nav_{sec}",
            use_container_width=True,
        ):
            SS["section"] = sec
            st.rerun()

    # Status bar
    result  = SS.get("backtest_result")
    n_trades = len(result.trades) if result and result.trades is not None else 0

    sidebar_status_bar(
        instrument    = SS["instrument"],
        strategy_file = SS["strategy_filename"],
        csv_file      = SS["data_filename"],
        last_run      = SS["last_run"],
        total_trades  = n_trades,
        engine_status = SS["engine_status"],
    )


# ────────────────────────────────────────────────────────────────────────────
# Helper: current backtest config dict
# ────────────────────────────────────────────────────────────────────────────

def _build_config_dict() -> dict:
    return {k: SS.get(k, DEFAULT_CONFIG.get(k)) for k in DEFAULT_CONFIG}


def _build_backtest_config() -> BacktestConfig:
    return BacktestConfig(
        instrument          = SS["instrument"],
        starting_balance    = SS["starting_balance"],
        num_contracts       = SS["num_contracts"],
        sizing_mode         = SS["sizing_mode"],
        risk_per_trade_pct  = SS["risk_per_trade_pct"],
        commission_per_side = SS["commission_per_side"],
        exchange_fee        = SS["exchange_fee"],
        nfa_fee             = SS["nfa_fee"],
        slippage_ticks      = SS["slippage_ticks"],
        profit_target_mode  = SS["profit_target_mode"],
        profit_target_value = SS["profit_target_value"],
        stop_loss_mode      = SS["stop_loss_mode"],
        stop_loss_value     = SS["stop_loss_value"],
        use_partial_exits   = SS["use_partial_exits"],
        partial_exit_pct    = SS["partial_exit_pct"],
        max_bars_in_trade   = SS["max_bars_in_trade"],
        eod_exit            = SS["eod_exit"],
        eod_exit_time       = SS["eod_exit_time"],
        session_filter      = SS["session_filter"],
        day_of_week_filter  = SS["day_of_week_filter"],
        daily_loss_limit    = SS["daily_loss_limit"],
    )


# ────────────────────────────────────────────────────────────────────────────
# Helper: filter data by trading period settings
# ────────────────────────────────────────────────────────────────────────────

def _apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    mode = SS.get("date_range_mode", "Full CSV History")
    if mode == "Full CSV History" or df is None or df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df
    if mode == "Custom Date Range":
        s = SS.get("custom_start")
        e = SS.get("custom_end")
        if s:
            df = df[idx >= pd.Timestamp(s, tz="UTC")]
        if e:
            df = df[df.index <= pd.Timestamp(e, tz="UTC")]
    elif mode == "Last N Trading Days":
        n = SS.get("last_n_days", 252)
        trading_days = df.index.normalize().unique()
        if len(trading_days) > n:
            cutoff = trading_days[-n]
            df = df[df.index >= cutoff]
    elif mode == "Last N Calendar Years":
        n = SS.get("last_n_years", 3)
        cutoff = pd.Timestamp(datetime.today()) - pd.DateOffset(years=n)
        df = df[df.index >= cutoff.tz_localize("UTC")]
    return df


def _render_parse_report(report):
    """Shared parse report display for both C++ and Python strategies."""
    if not report:
        return
    name = getattr(report.strategy, "name", "Unknown") if hasattr(report, "strategy") else "Unknown"
    st.markdown(
        f'<div style="font-family:monospace;font-size:0.80rem;'
        f'color:{COLORS["green"]};margin-bottom:4px;">'
        f'◈ STRATEGY PARSE REPORT — {name}</div>',
        unsafe_allow_html=True,
    )
    lines = report.summary or ["No summary available"]
    html = "<br>".join(
        f'<span style="color:{COLORS["green"] if l.startswith("✅") else COLORS["amber"] if l.startswith("⚠") else COLORS["red"] if l.startswith("❌") else COLORS["text"]}">{l}</span>'
        for l in lines
    )
    st.markdown(
        f'<div style="background:#050505;border:1px solid #2A2A2A;'
        f'border-radius:6px;padding:10px;font-family:monospace;'
        f'font-size:0.72rem;color:#CCC;max-height:360px;overflow-y:auto;">'
        + html +
        f'<br><br><span style="color:#555;">Lines: {report.line_count} '
        f'| Score: {report.parse_score:.0f}%</span>'
        '</div>',
        unsafe_allow_html=True,
    )



# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION & INPUTS
# ════════════════════════════════════════════════════════════════════════════

def render_section_1():
    section_header("CONFIGURATION & INPUTS",
                   "Master control panel — all backtest parameters")

    # ── 2.1 Instrument selector ──────────────────────────────────────────────
    with st.expander("⬡  INSTRUMENT SELECTOR", expanded=True):
        cols = st.columns(4)
        for i, sym in enumerate(["NQ", "ES", "MNQ", "MES"]):
            with cols[i]:
                spec = INSTRUMENTS[sym]
                active = SS["instrument"] == sym
                border = COLORS["green"] if active else COLORS["card_border"]
                if st.button(
                    f"**{sym}**\n{spec['name'][:20]}",
                    key=f"inst_{sym}",
                    use_container_width=True,
                ):
                    SS["instrument"] = sym
                    st.rerun()
        st.markdown("---")
        instrument_spec_card(SS["instrument"], INSTRUMENTS[SS["instrument"]])

    # ── 2.2 Account & capital ────────────────────────────────────────────────
    with st.expander("💰  ACCOUNT & CAPITAL SETTINGS", expanded=True):
        st.markdown("**Starting Account Balance**")
        bal_cols = st.columns(6)
        for i, amt in enumerate([10_000, 25_000, 50_000, 100_000, 150_000, 200_000]):
            with bal_cols[i]:
                if st.button(f"${amt//1000}k", key=f"bal_{amt}"):
                    SS["starting_balance"] = float(amt)
        SS["starting_balance"] = st.number_input(
            "Custom Balance ($)",
            min_value=100.0, max_value=10_000_000.0,
            value=float(SS["starting_balance"]),
            step=1_000.0, format="%.2f",
        )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Number of Contracts**")
            ctr_cols = st.columns(5)
            for i, n in enumerate([1, 2, 5, 10, 20]):
                with ctr_cols[i]:
                    if st.button(str(n), key=f"ctr_{n}"):
                        SS["num_contracts"] = n
            SS["num_contracts"] = st.number_input(
                "Custom contracts",
                min_value=1, max_value=500,
                value=int(SS["num_contracts"]), step=1,
            )
        with c2:
            SS["sizing_mode"] = st.radio(
                "Contract Sizing Mode",
                ["Fixed", "Dynamic / Volatility-Scaled", "State Machine Sizing"],
                index=["Fixed", "Dynamic / Volatility-Scaled",
                       "State Machine Sizing"].index(SS["sizing_mode"]),
                horizontal=False,
            )
            if SS["sizing_mode"] != "Fixed":
                SS["risk_per_trade_pct"] = st.number_input(
                    "Risk per trade (% of account)",
                    min_value=0.1, max_value=10.0,
                    value=float(SS["risk_per_trade_pct"]),
                    step=0.1, format="%.2f",
                )

    # ── 2.3 Commission & fees ────────────────────────────────────────────────
    with st.expander("💸  COMMISSION & FEES"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Commission / contract / side**")
            for amt in [0.0, 0.25, 0.50, 1.00, 2.50]:
                if st.button(f"${amt:.2f}", key=f"comm_{amt}"):
                    SS["commission_per_side"] = amt
            SS["commission_per_side"] = st.number_input(
                "Custom ($)", min_value=0.0, max_value=50.0,
                value=float(SS["commission_per_side"]), step=0.05, format="%.4f",
            )
        with c2:
            st.markdown("**Exchange Fee**")
            for amt in [0.0, 0.85, 1.18, 1.50]:
                if st.button(f"${amt:.2f}", key=f"exch_{amt}"):
                    SS["exchange_fee"] = amt
            SS["exchange_fee"] = st.number_input(
                "Custom ($)", min_value=0.0, max_value=20.0,
                value=float(SS["exchange_fee"]), step=0.01, format="%.4f",
                key="exch_custom",
            )
        with c3:
            st.markdown("**NFA Fee**")
            SS["nfa_fee"] = st.number_input(
                "NFA ($/contract)", min_value=0.0, max_value=5.0,
                value=float(SS["nfa_fee"]), step=0.01, format="%.4f",
            )
            st.markdown("**Slippage (ticks)**")
            slip_cols = st.columns(4)
            for i, t in enumerate([0, 1, 2, 4]):
                with slip_cols[i]:
                    if st.button(str(t), key=f"slip_{t}"):
                        SS["slippage_ticks"] = t
            SS["slippage_ticks"] = st.number_input(
                "Custom ticks", min_value=0, max_value=20,
                value=int(SS["slippage_ticks"]), step=1,
            )

        spec = INSTRUMENTS[SS["instrument"]]
        slip_dollars = SS["slippage_ticks"] * spec["tick_value"] * 2
        rt_cost = (SS["commission_per_side"] + SS["exchange_fee"] + SS["nfa_fee"]) * 2
        total_cost = rt_cost + slip_dollars
        banner_info(
            f"Total estimated round-trip cost: "
            f"**${rt_cost:.4f}** commission + "
            f"**${slip_dollars:.4f}** slippage = "
            f"**${total_cost:.4f}** per contract"
        )

    # ── 2.4 Profit target & stop loss ────────────────────────────────────────
    with st.expander("🎯  PROFIT TARGET & STOP LOSS SETTINGS"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Profit Target Mode**")
            SS["profit_target_mode"] = st.radio(
                "Mode",
                ["Strategy", "Points", "Ticks", "Dollars", "R-Multiple"],
                index=["Strategy", "Points", "Ticks", "Dollars",
                       "R-Multiple"].index(SS["profit_target_mode"]),
                horizontal=True, label_visibility="collapsed",
            )
            for amt in [500, 1_000, 1_500, 2_000, 3_000, 5_000]:
                if st.button(f"${amt:,}", key=f"tp_{amt}"):
                    SS["profit_target_value"] = float(amt)
            SS["profit_target_value"] = st.number_input(
                "Custom TP value", min_value=0.0, max_value=1_000_000.0,
                value=float(SS["profit_target_value"]), step=50.0, format="%.2f",
            )
        with c2:
            st.markdown("**Stop Loss Mode**")
            SS["stop_loss_mode"] = st.radio(
                "Mode",
                ["Strategy", "Points", "Ticks", "Dollars", "R-Multiple"],
                index=["Strategy", "Points", "Ticks", "Dollars",
                       "R-Multiple"].index(SS["stop_loss_mode"]),
                horizontal=True, label_visibility="collapsed",
            )
            for amt in [100, 250, 500, 1_000, 2_000]:
                if st.button(f"${amt:,}", key=f"sl_{amt}"):
                    SS["stop_loss_value"] = float(amt)
            SS["stop_loss_value"] = st.number_input(
                "Custom SL value", min_value=0.0, max_value=1_000_000.0,
                value=float(SS["stop_loss_value"]), step=50.0, format="%.2f",
            )

        # R:R display
        if SS["stop_loss_value"] > 0:
            rr = SS["profit_target_value"] / SS["stop_loss_value"]
            st.markdown(
                f'<div style="font-family:monospace;color:{COLORS["amber"]};">'
                f"RR Ratio: 1:{rr:.2f}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            SS["use_partial_exits"] = st.toggle("Enable Partial Exits", SS["use_partial_exits"])
            if SS["use_partial_exits"]:
                SS["partial_exit_pct"] = st.slider(
                    "Exit % at first target", 10, 90,
                    int(SS["partial_exit_pct"]), 5,
                )
        with c2:
            SS["max_bars_in_trade"] = st.number_input(
                "Max bars in trade (0 = disabled)",
                min_value=0, max_value=10_000,
                value=int(SS["max_bars_in_trade"]), step=1,
            )
            SS["eod_exit"] = st.toggle("End-of-day forced exit", SS["eod_exit"])
            if SS["eod_exit"]:
                SS["eod_exit_time"] = st.text_input("EOD exit time (HH:MM CT)",
                                                     SS["eod_exit_time"])

    # ── 2.5 Trading period ───────────────────────────────────────────────────
    with st.expander("📅  TRADING DAYS / TESTING PERIOD"):
        SS["date_range_mode"] = st.radio(
            "Date Range Mode",
            ["Full CSV History", "Custom Date Range",
             "Last N Trading Days", "Last N Calendar Years"],
            index=["Full CSV History", "Custom Date Range",
                   "Last N Trading Days", "Last N Calendar Years"
                   ].index(SS["date_range_mode"]),
            horizontal=True,
        )
        if SS["date_range_mode"] == "Custom Date Range":
            c1, c2 = st.columns(2)
            with c1:
                d = st.date_input("Start date",
                                  value=date(2020, 1, 1) if not SS["custom_start"]
                                  else SS["custom_start"])
                SS["custom_start"] = d
            with c2:
                d = st.date_input("End date",
                                  value=date.today() if not SS["custom_end"]
                                  else SS["custom_end"])
                SS["custom_end"] = d
        elif SS["date_range_mode"] == "Last N Trading Days":
            SS["last_n_days"] = st.number_input("Trading days back",
                                                  min_value=1, max_value=5_000,
                                                  value=SS.get("last_n_days", 252))
        elif SS["date_range_mode"] == "Last N Calendar Years":
            n_cols = st.columns(5)
            for i, yr in enumerate([1, 2, 3, 5, 10]):
                with n_cols[i]:
                    if st.button(f"{yr}yr", key=f"yr_{yr}"):
                        SS["last_n_years"] = yr
            SS["last_n_years"] = st.number_input(
                "Custom years", min_value=1, max_value=30,
                value=SS.get("last_n_years", 3))

        st.markdown("---")
        SS["session_filter"] = st.selectbox(
            "Session Filter",
            ["Full Session", "Regular Trading Hours Only",
             "Overnight Only", "Pre-Market"],
            index=["Full Session", "Regular Trading Hours Only",
                   "Overnight Only", "Pre-Market"].index(SS["session_filter"]),
        )

        days_avail = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_sel = st.multiselect(
            "Day-of-Week Filter",
            days_avail,
            default=[days_avail[i] for i in SS["day_of_week_filter"]],
        )
        SS["day_of_week_filter"] = [days_avail.index(d) for d in day_sel]

    # ── 2.6 Display scaling ──────────────────────────────────────────────────
    with st.expander("📐  SIZING & DISPLAY SCALING"):
        scale_labels = ["1 MNQ", "2 MNQ", "5 MNQ", "10 MNQ", "1 NQ", "2 NQ", "5 NQ"]
        scale_values = [1, 2, 5, 10, 20, 40, 100]
        sc_cols = st.columns(len(scale_labels))
        for i, (lbl, val) in enumerate(zip(scale_labels, scale_values)):
            with sc_cols[i]:
                if st.button(lbl, key=f"scale_{lbl}"):
                    SS["display_scale"] = val
        SS["display_scale"] = st.number_input(
            "Custom scale multiplier (× base contracts)",
            min_value=1, max_value=1000,
            value=int(SS["display_scale"]), step=1,
        )

    # ── Risk-free rate ────────────────────────────────────────────────────────
    with st.expander("📊  RISK-FREE RATE (Sharpe)"):
        SS["risk_free_rate"] = st.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0, max_value=20.0,
            value=float(SS["risk_free_rate"]), step=0.25, format="%.2f",
        )
        SS["daily_loss_limit"] = st.number_input(
            "Daily loss limit $ (0 = disabled)",
            min_value=0.0, max_value=1_000_000.0,
            value=float(SS["daily_loss_limit"]), step=100.0,
        )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — STRATEGY LOADER
# ════════════════════════════════════════════════════════════════════════════

def render_section_2():
    section_header("STRATEGY LOADER",
                   "Upload .cpp OR .py strategy file — logic parser + native compile support")

    # ── Strategy type tabs ────────────────────────────────────────────────────
    lang_tab1, lang_tab2, lang_tab3 = st.tabs([
        "📄 C++ Strategy (.cpp)",
        "🐍 Python Strategy (.py)",
        "🔧 Demo Strategy",
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: C++ STRATEGY
    # ════════════════════════════════════════════════════════════════
    with lang_tab1:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("**Upload C++ Strategy File(s)**")
            st.caption("Accepts .cpp / .h / .hpp — up to 10 MB each. Drop multiple files for multi-file strategies.")
            uploaded_cpp = st.file_uploader(
                "Drop .cpp files here",
                type=["cpp", "h", "hpp", "cxx"],
                accept_multiple_files=True,
                key="cpp_uploader",
            )

            if uploaded_cpp:
                combined_source = ""
                for f in uploaded_cpp:
                    size_kb = len(f.getvalue()) / 1024
                    st.markdown(
                        f'<span class="pill pill-green">{f.name}</span>'
                        f'<span class="pill">{size_kb:.1f} KB</span>'
                        f'<span class="pill">{datetime.now().strftime("%H:%M:%S")}</span>',
                        unsafe_allow_html=True,
                    )
                    combined_source += f.read().decode("utf-8", errors="replace") + "\n\n"

                SS["cpp_source"]        = combined_source
                SS["strategy_filename"] = uploaded_cpp[0].name
                SS["strategy_type"]     = "cpp"
                SS["use_demo_strategy"] = False

                with st.spinner("Parsing C++ strategy logic…"):
                    try:
                        report = parse_cpp_strategy(combined_source, uploaded_cpp[0].name)
                        SS["parse_report"] = report
                        score = report.parse_score
                        if score >= 60:
                            banner_success(
                                f"**{report.strategy.name}** parsed — confidence: {score:.0f}%  "
                                f"| Signals: {len(report.strategy.signals)}  "
                                f"| Stop: {report.strategy.stop.mode if report.strategy.stop else '?'}  "
                                f"| Target: {report.strategy.target.mode if report.strategy.target else '?'}"
                            )
                        else:
                            banner_warning(
                                f"Partial parse — confidence {score:.0f}%. "
                                "Override any missing elements in Section 1 before running."
                            )
                    except Exception as exc:
                        banner_error(f"Parse error: {exc}\n\n{traceback.format_exc()[-400:]}")
                        SS["use_demo_strategy"] = True

            st.markdown("---")
            st.markdown("**Compiled C++ Execution (Advanced)**")
            SS["compile_mode"] = st.toggle(
                "Compile & Execute C++ Natively (requires g++ on PATH)",
                SS.get("compile_mode", False),
            )
            if SS["compile_mode"]:
                banner_warning(
                    "Native g++ compilation mode enabled. "
                    "The terminal will fall back to the Python engine if compilation fails."
                )

        with col2:
            # Parse report panel
            report = SS.get("parse_report")
            if report and SS.get("strategy_type") == "cpp":
                _render_parse_report(report)
                if hasattr(report, 'strategy') and report.strategy.raw_constants:
                    with st.expander(f"📐 Parsed Constants ({len(report.strategy.raw_constants)})"):
                        cdf = pd.DataFrame(
                            list(report.strategy.raw_constants.items()),
                            columns=["Name", "Value"],
                        )
                        st.dataframe(cdf, use_container_width=True, hide_index=True)
            else:
                banner_info("Upload a .cpp file to see the parse report here.")

        # Raw source preview
        if SS.get("cpp_source") and SS.get("strategy_type") == "cpp":
            with st.expander("📄 Raw C++ Source Preview (first 200 lines)"):
                lines = SS["cpp_source"].splitlines()[:200]
                st.code("\n".join(lines), language="cpp")

    # ════════════════════════════════════════════════════════════════
    # TAB 2: PYTHON STRATEGY
    # ════════════════════════════════════════════════════════════════
    with lang_tab2:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("**Upload Python Strategy File (.py)**")
            st.caption(
                "Your .py file must define `class Strategy(PyStrategyBase)` "
                "with `on_bar()`, `stop_distance()`, and `target_distance()` methods."
            )
            uploaded_py = st.file_uploader(
                "Drop .py strategy file here",
                type=["py"],
                accept_multiple_files=False,
                key="py_uploader",
            )

            # Example download
            st.download_button(
                "📥 Download Example .py Strategy Template",
                data=EXAMPLE_PY_STRATEGY.encode(),
                file_name="example_strategy.py",
                mime="text/plain",
                help="Download a working example Python strategy to edit and re-upload.",
            )

            if uploaded_py:
                size_kb = len(uploaded_py.getvalue()) / 1024
                st.markdown(
                    f'<span class="pill pill-blue">{uploaded_py.name}</span>'
                    f'<span class="pill">{size_kb:.1f} KB</span>'
                    f'<span class="pill">{datetime.now().strftime("%H:%M:%S")}</span>',
                    unsafe_allow_html=True,
                )
                py_source = uploaded_py.read().decode("utf-8", errors="replace")
                SS["py_source"]         = py_source
                SS["strategy_filename"] = uploaded_py.name
                SS["strategy_type"]     = "python"
                SS["use_demo_strategy"] = False

                with st.spinner("Loading and validating Python strategy…"):
                    try:
                        py_report = load_py_strategy(py_source, uploaded_py.name)
                        SS["py_parse_report"] = py_report

                        if py_report.is_valid:
                            # Build a parse_report wrapper compatible with the rest of the app
                            strat_def = py_report.strategy_def
                            SS["parse_report"] = type("R", (), {
                                "strategy":    strat_def,
                                "line_count":  py_report.line_count,
                                "parse_score": py_report.parse_score,
                                "summary":     py_report.summary,
                                "warnings":    py_report.warnings,
                            })()
                            banner_success(
                                f"**{py_report.strategy_name}** loaded — "
                                f"parse score: {py_report.parse_score:.0f}%  |  "
                                f"Signals: {len(strat_def.signals if strat_def else [])}  |  "
                                f"Cooldown: {strat_def.cooldown_bars if strat_def else 5} bars"
                            )
                        else:
                            for err in py_report.errors:
                                banner_error(err)
                    except Exception as exc:
                        banner_error(f"Python strategy load error: {exc}\n\n{traceback.format_exc()[-400:]}")

            # Inline editor
            st.markdown("---")
            st.markdown("**Or write/edit strategy directly in the terminal:**")
            if SS.get("strategy_type") == "python" and SS.get("py_source"):
                initial_code = SS["py_source"]
            else:
                initial_code = EXAMPLE_PY_STRATEGY

            edited_code = st.text_area(
                "Python Strategy Editor",
                value=initial_code,
                height=340,
                key="py_editor",
                help="Edit Python strategy code inline, then click Load.",
            )
            if st.button("⚡ Load Inline Strategy", key="load_inline_py"):
                with st.spinner("Validating inline Python strategy…"):
                    try:
                        py_report = load_py_strategy(edited_code, "inline_strategy.py")
                        SS["py_source"]         = edited_code
                        SS["py_parse_report"]   = py_report
                        SS["strategy_filename"] = "inline_strategy.py"
                        SS["strategy_type"]     = "python"
                        SS["use_demo_strategy"] = False
                        if py_report.is_valid:
                            strat_def = py_report.strategy_def
                            SS["parse_report"] = type("R", (), {
                                "strategy":    strat_def,
                                "line_count":  py_report.line_count,
                                "parse_score": py_report.parse_score,
                                "summary":     py_report.summary,
                                "warnings":    py_report.warnings,
                            })()
                            banner_success(f"**{py_report.strategy_name}** loaded from inline editor.")
                        else:
                            for err in py_report.errors:
                                banner_error(err)
                    except Exception as exc:
                        banner_error(f"Inline load error: {exc}")

        with col2:
            py_report = SS.get("py_parse_report")
            if py_report and SS.get("strategy_type") == "python":
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.80rem;'
                    f'color:{COLORS["blue"]};margin-bottom:4px;">'
                    f'◈ PYTHON STRATEGY PARSE REPORT — {py_report.strategy_name}</div>',
                    unsafe_allow_html=True,
                )
                all_lines = (py_report.summary or []) + (py_report.errors or [])
                html_lines = "<br>".join(
                    f'<span style="color:{COLORS["green"] if l.startswith("✅") else COLORS["amber"] if l.startswith("⚠") else COLORS["red"] if l.startswith("❌") else COLORS["text"]}">{l}</span>'
                    for l in all_lines
                )
                st.markdown(
                    f'<div style="background:#050505;border:1px solid #2A2A2A;'
                    f'border-radius:6px;padding:10px;font-family:monospace;'
                    f'font-size:0.72rem;color:#CCC;max-height:380px;overflow-y:auto;">'
                    + html_lines +
                    f'<br><br><span style="color:#555;">Lines: {py_report.line_count} '
                    f'| Valid: {py_report.is_valid} | Score: {py_report.parse_score:.0f}%</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                if py_report.strategy_def and py_report.strategy_def.raw_constants:
                    with st.expander(f"📐 Detected Constants ({len(py_report.strategy_def.raw_constants)})"):
                        cdf = pd.DataFrame(
                            list(py_report.strategy_def.raw_constants.items()),
                            columns=["Name", "Value"],
                        )
                        st.dataframe(cdf, use_container_width=True, hide_index=True)
            else:
                banner_info("Upload or write a .py strategy to see its parse report here.")

    # ════════════════════════════════════════════════════════════════
    # TAB 3: DEMO STRATEGY
    # ════════════════════════════════════════════════════════════════
    with lang_tab3:
        st.markdown("**Built-in Demo Strategy: MOM_LONG / MOM_SHORT**")
        banner_info(
            "The demo strategy uses z-return momentum + VWAP regime filter. "
            "It fires MOM_LONG when price is above VWAP with strong positive z-return, "
            "and MOM_SHORT when price is below VWAP with strong negative z-return. "
            "ATR × 1.5 stop, 2R target. Use this to test the terminal with any CSV data."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅  Activate Demo Strategy", type="primary", use_container_width=True):
                demo_def = demo_strategy_definition()
                SS["parse_report"] = type("R", (), {
                    "strategy":    demo_def,
                    "line_count":  0,
                    "parse_score": 100.0,
                    "summary":     [
                        "✅ Entry Signals: MOM_LONG, MOM_SHORT",
                        "✅ Stop Loss: ATR × 1.5 (dynamic)",
                        "✅ Take Profit: 2.0R with 50% partial at 1.0R",
                        "✅ Sizing: Fixed 1 contract",
                        "✅ Cooldown: 5 bars between signals",
                        "✅ EOD Exit: 15:45 CT",
                        "✅ Indicators: VWAP, ATR, z-return, z-momentum",
                    ],
                    "warnings": [],
                })()
                SS["strategy_filename"] = "<demo>"
                SS["strategy_type"]     = "demo"
                SS["use_demo_strategy"] = True
                banner_success("Demo strategy activated. Go to Section 4 to run the backtest.")

        with col2:
            if SS.get("strategy_type") == "demo":
                st.markdown(
                    f'<div style="background:{COLORS["green_dark"]};border:1px solid {COLORS["green"]};'
                    f'border-radius:8px;padding:12px;font-family:monospace;font-size:0.78rem;'
                    f'color:{COLORS["green"]};">✅ DEMO STRATEGY ACTIVE</div>',
                    unsafe_allow_html=True,
                )

        # Show demo parse report if active
        if SS.get("strategy_type") == "demo" and SS.get("parse_report"):
            report = SS["parse_report"]
            _render_parse_report(report)



# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATA LOADER
# ════════════════════════════════════════════════════════════════════════════

def render_section_3():
    section_header("DATA LOADER",
                   "CSV historical data import, validation & free API data")

    tab1, tab2 = st.tabs(["📁 CSV Upload", "🌐 Free API Data (yfinance)"])

    with tab1:
        _render_csv_upload()

    with tab2:
        _render_api_data()


def _render_csv_upload():
    uploaded = st.file_uploader(
        "Upload OHLCV CSV file(s) — TradeStation / NinjaTrader / Sierra Chart / Rithmic",
        type=["csv", "txt"],
        accept_multiple_files=True,
        help="Multiple files will be concatenated in chronological order.",
    )

    if uploaded:
        all_frames = []
        all_reports = []
        for f in uploaded:
            try:
                df, report = load_csv_upload(io.BytesIO(f.read()))
                all_frames.append(df)
                all_reports.append((f.name, report))
            except Exception as exc:
                banner_error(f"**{f.name}**: {exc}")
                return

        if all_frames:
            raw_df = pd.concat(all_frames).sort_index()
            raw_df = raw_df[~raw_df.index.duplicated(keep="last")]
            SS["raw_df"] = raw_df
            SS["data_filename"] = uploaded[0].name
            combined_report = all_reports[0][1]
            combined_report["final_rows"] = len(raw_df)
            SS["data_report"] = combined_report

            # Instrument auto-detection
            det = auto_detect_instrument(raw_df, SS["data_filename"])
            if det["confidence"] in ("high", "medium"):
                if det["detected"]:
                    SS["instrument"] = det["detected"]
                if det["confidence"] == "high":
                    banner_success(det["message"])
                else:
                    banner_warning(det["message"])
            else:
                banner_warning(det["message"])

    # Display report and preview if data loaded
    if SS.get("raw_df") is not None and SS.get("data_report"):
        raw_df = SS["raw_df"]
        rep    = SS["data_report"]

        st.markdown("---")
        st.markdown(
            f'<div class="section-header">◈ DATA QUALITY REPORT</div>',
            unsafe_allow_html=True,
        )

        q_items = [
            (True,  "File",          SS["data_filename"]),
            (True,  "Date Range",    f"{rep.get('date_start','?')} → {rep.get('date_end','?')}"),
            (True,  "Total Bars",    f"{rep.get('final_rows', len(raw_df)):,}"),
            (True,  "Bar Interval",  rep.get("interval", "unknown")),
            (rep.get("duplicate_rows", 0) == 0, "Duplicates Removed",
             f"{rep.get('duplicate_rows', 0)}"),
            (len(rep.get("anomalies", [])) == 0, "Price Anomalies",
             f"{len(rep.get('anomalies', []))} detected"),
            (rep.get("gaps_detected", 0) == 0, "Gaps Detected",
             f"{rep.get('gaps_detected', 0)} gaps > 2× normal interval"),
            (not rep.get("has_weekends", False), "Weekend Data",
             "None" if not rep.get("has_weekends") else "Present — verify session filter"),
        ]
        checklist(q_items)

        if rep.get("missing_bars"):
            with st.expander(f"⚠️ Gap Details ({len(rep['missing_bars'])} shown)"):
                st.code("\n".join(rep["missing_bars"]))

        # Preview
        st.markdown("---")
        st.markdown('<div class="section-header">◈ DATA PREVIEW</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**First 20 rows**")
            st.dataframe(raw_df.head(20), use_container_width=True, height=300)
        with c2:
            st.markdown("**Last 20 rows**")
            st.dataframe(raw_df.tail(20), use_container_width=True, height=300)

        # Mini price chart
        st.markdown("**Close Price Overview**")
        close = raw_df["close"].resample("1D").last().dropna() if len(raw_df) > 5000 \
            else raw_df["close"]
        fig = go.Figure(go.Scatter(
            x=close.index, y=close.values,
            mode="lines",
            line=dict(color=COLORS["green"], width=1),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.05)",
        ))
        fig.update_layout(
            height=220, margin=dict(l=50, r=10, t=10, b=30),
            paper_bgcolor=COLORS["bg"], plot_bgcolor="#0F0F0F",
            font=dict(color=COLORS["text"]),
            yaxis=dict(gridcolor=COLORS["grid"]),
            xaxis=dict(gridcolor=COLORS["grid"]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Column mapping
        if raw_df is not None:
            needed = {"open", "high", "low", "close"}
            missing = needed - set(raw_df.columns)
            if missing:
                banner_warning(
                    f"Missing columns: {missing}. "
                    "Use the mapper below to assign them."
                )
                with st.expander("🔧 Column Mapper"):
                    for col in missing:
                        st.selectbox(f"Map to '{col}'", options=["—"] + list(raw_df.columns),
                                     key=f"col_map_{col}")


def _render_api_data():
    """Section 3 — API Data Tab: Native Pre-loaded Data + Live API fetching."""
    COLORS_REF = COLORS

    # ══════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    ntab, apitab = st.tabs([
        "⚡ Pre-Loaded Native Data  (ES · NQ · SPX — No API needed)",
        "📡 Live API Fetch  (Tastytrade · Polygon · Alpha Vantage · Finnhub · yfinance)",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — NATIVE PRE-LOADED DATA
    # ══════════════════════════════════════════════════════════════════════
    with ntab:
        st.markdown(
            f'''<div style="background:#050A05;border:2px solid {COLORS["green"]};border-radius:10px;
            padding:16px 20px;margin-bottom:16px;">
            <div style="font-family:monospace;font-size:0.90rem;color:{COLORS["green"]};font-weight:700;">
            ⚡ NATIVE PRE-LOADED HISTORICAL DATA</div>
            <div style="font-family:monospace;font-size:0.74rem;color:#666;margin-top:6px;">
            ES · NQ · SPX datasets are permanently embedded in the terminal and available
            instantly — no API calls, no internet required, no manual imports.<br>
            <b>Total: 11,096,654 bars</b> across all instruments and timeframes.
            </div></div>''',
            unsafe_allow_html=True,
        )

        # Dataset catalogue table
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.78rem;color:{COLORS["green"]};margin-bottom:6px;">'
            f'◈ AVAILABLE DATASETS</div>', unsafe_allow_html=True)

        cat_rows = []
        for stem, symbol, tf, desc in CATALOGUE:
            from native_data import get_preload_path
            p = get_preload_path(stem)
            size_mb = round(p.stat().st_size / 1024 / 1024, 1) if p.exists() else 0
            bars_approx = {
                "ES_1m":4_234_977,"NQ_1m":4_174_598,
                "ES_5m":850_923,"NQ_5m":850_374,
                "ES_15m":283_715,"NQ_15m":283_644,
                "SPX_5m":380_948,
            }.get(stem, "~15K")
            cat_rows.append({
                "Symbol": symbol, "TF": tf,
                "Bars": f"{bars_approx:,}" if isinstance(bars_approx,int) else bars_approx,
                "Size (MB)": size_mb,
                "Description": desc[:55],
                "Status": "✅ Ready" if p.exists() else "❌ Missing",
            })
        import pandas as pd
        cat_df = pd.DataFrame(cat_rows)
        st.dataframe(cat_df, use_container_width=True, hide_index=True,
                     column_config={
                         "Status": st.column_config.TextColumn("Status", width="small"),
                         "Bars":   st.column_config.TextColumn("Bars"),
                     })

        # ── Selector ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{COLORS["amber"]};">'
                    f'◈ LOAD INTO TERMINAL</div>', unsafe_allow_html=True)

        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            n_sym = st.selectbox("Instrument", ["ES","NQ","SPX","MES","MNQ"], key="nd_sym")
        with nc2:
            # Show only available timeframes for selected symbol
            avail_tfs = list({row[2] for row in CATALOGUE
                              if row[1].upper() == n_sym.upper()})
            avail_tfs_sorted = sorted(avail_tfs, key=lambda x: {
                "1m":1,"5m":2,"15m":3,"30m":4,"1h":5,"4h":6,"1d":7}.get(x,99))
            n_tf = st.selectbox("Timeframe", avail_tfs_sorted, key="nd_tf")
        with nc3:
            prefer_full = st.toggle("Full history (2014–2026)", True, key="nd_full")

        # Date range filter
        ndc1, ndc2 = st.columns(2)
        with ndc1:
            from datetime import date as dt_date
            nd_start = st.date_input("From (optional)", value=dt_date(2020,1,1), key="nd_start")
        with ndc2:
            nd_end = st.date_input("To (optional)", value=dt_date.today(), key="nd_end")

        if st.button("⚡  LOAD NATIVE DATA", type="primary", use_container_width=True, key="nd_load"):
            with st.spinner(f"Loading {n_sym} {n_tf} from pre-loaded dataset…"):
                df_native, desc_native = get_best_available(
                    n_sym, n_tf,
                    str(nd_start), str(nd_end),
                )
            if df_native.empty:
                banner_warning(f"No native data found for {n_sym} {n_tf}. "
                               "Try a different timeframe or use the Live API tab.")
            else:
                SS["raw_df"]       = df_native
                SS["data_filename"]= f"{n_sym}_{n_tf}_native"
                SS["instrument"]   = n_sym if n_sym in INSTRUMENTS else "ES"
                banner_success(
                    f"✅ Loaded **{len(df_native):,}** bars of {n_sym} ({n_tf}) — "
                    f"{desc_native} | "
                    f"{str(df_native.index.min().date())} → {str(df_native.index.max().date())}"
                )

        # Show preview if data already loaded from native
        if SS.get("raw_df") is not None and "native" in SS.get("data_filename",""):
            raw = SS["raw_df"]
            st.markdown("---")
            st.markdown(f'<div style="font-family:monospace;font-size:0.76rem;color:{COLORS["text_dim"]};">'
                        f'Currently loaded: <b>{SS["data_filename"]}</b> · {len(raw):,} bars · '
                        f'{str(raw.index.min().date())} → {str(raw.index.max().date())}</div>',
                        unsafe_allow_html=True)
            import plotly.graph_objects as go
            daily = raw["close"].resample("1D").last().dropna()
            fig_prev = go.Figure(go.Scatter(
                x=daily.index, y=daily.values, mode="lines",
                line=dict(color=COLORS["green"], width=1.2),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.05)",
            ))
            fig_prev.update_layout(
                height=200, margin=dict(l=50,r=10,t=10,b=30),
                paper_bgcolor=COLORS["bg"], plot_bgcolor="#0F0F0F",
                font=dict(color=COLORS["text"]),
                xaxis=dict(gridcolor=COLORS["grid"]),
                yaxis=dict(gridcolor=COLORS["grid"], tickformat=","),
            )
            st.plotly_chart(fig_prev, use_container_width=True, key="nd_preview")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — LIVE API FETCH
    # ══════════════════════════════════════════════════════════════════════
    with apitab:
        st.markdown("**Live API pipeline — priority chain: Tastytrade → Polygon → Alpha Vantage → Finnhub → yfinance**")

        # Tastytrade connection status
        try:
            from tastytrade_loader import get_client as tt_client
            tt = tt_client()
            tt_ok = tt.ping()
        except Exception:
            tt_ok = False

        status_color = COLORS["green"] if tt_ok else COLORS["red"]
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.76rem;">' +
            f'<span style="color:{status_color};">● Tastytrade</span> ' +
            f'({"connected" if tt_ok else "not connected"}) &nbsp;&nbsp;' +
            f'<span style="color:{COLORS["text_dim"]};">Polygon · Alpha Vantage · Finnhub · yfinance also available</span>' +
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            api_asset = st.selectbox("Asset Class", ["Futures","Options"], key="api_asset")
        with c2:
            if api_asset == "Futures":
                api_ticker = st.selectbox("Instrument", ["NQ","ES","MNQ","MES","SPX"], key="api_tick")
            else:
                api_ticker = st.selectbox("Underlying", ["SPY","QQQ"], key="api_tick_opt")
        with c3:
            api_tf = st.selectbox("Timeframe", ["1d","1h","30m","15m","5m","1m"], key="api_tf")
        with c4:
            api_start = st.date_input("Start", value=date(2024,1,1), key="api_start")
            api_end   = st.date_input("End",   value=date.today(),   key="api_end")

        source_order = st.multiselect(
            "Source priority order",
            ["tastytrade","polygon","alpha_vantage","finnhub","yfinance"],
            default=["tastytrade","polygon","alpha_vantage","finnhub","yfinance"],
            key="api_sources",
        )

        if api_asset == "Options":
            dte_filter = st.multiselect("DTE Filter",
                [0,1,2,3,5,7,14,30], default=[0,1,2], key="api_dte")

        if st.button("📡  FETCH FROM API", type="primary", key="api_fetch"):
            try:
                if api_asset == "Futures":
                    from multi_api_loader import fetch_ohlcv
                    with st.spinner(f"Fetching {api_ticker} ({api_tf}) via {source_order[0] if source_order else 'yfinance'}…"):
                        df_api, report_api = fetch_ohlcv(
                            api_ticker, api_tf,
                            str(api_start), str(api_end),
                            sources=source_order or None,
                        )
                    if df_api.empty:
                        errors_str = " | ".join(f"{k}: {v}" for k,v in report_api.get("errors",{}).items())
                        banner_error(f"All sources failed: {errors_str}")
                    else:
                        SS["raw_df"]        = df_api
                        SS["data_filename"] = f"{api_ticker}_{api_tf}_{report_api.get('source','api')}"
                        SS["instrument"]    = api_ticker if api_ticker in INSTRUMENTS else "ES"
                        banner_success(
                            f"✅ **{len(df_api):,}** bars of {api_ticker} ({api_tf}) "
                            f"from **{report_api.get('source','?')}** | "
                            f"tried: {report_api.get('tried',[])}"
                        )
                else:
                    df_opt = get_options_chain(api_ticker, dte_filter=dte_filter)
                    st.dataframe(df_opt.head(100), use_container_width=True)
                    banner_success(f"Fetched **{len(df_opt):,}** options contracts for {api_ticker}.")
            except Exception as exc:
                banner_error(f"Fetch failed: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════════════════

def render_section_4():
    section_header("BACKTEST ENGINE", "Run controls, pre-flight checklist, live progress")

    # ── Pre-run checklist ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">◈ PRE-RUN CHECKLIST</div>',
                unsafe_allow_html=True)

    raw_df   = SS.get("raw_df")
    rep      = SS.get("parse_report")
    strategy = rep.strategy if rep else None
    cfg      = _build_backtest_config()
    spec     = INSTRUMENTS[SS["instrument"]]

    slip_dol = SS["slippage_ticks"] * spec["tick_value"]
    rt_cost  = (SS["commission_per_side"] + SS["exchange_fee"] + SS["nfa_fee"]) * 2

    items = [
        (True,                      "Instrument",        f'{SS["instrument"]} — {spec["name"]}'),
        (strategy is not None,      "Strategy",          getattr(strategy, "name", "None") if strategy else "Not loaded"),
        (raw_df is not None,        "CSV Data",          f'{SS["data_filename"]} ({len(raw_df):,} bars)' if raw_df is not None else "Not loaded"),
        (SS["starting_balance"] > 0,"Account Balance",   f'${SS["starting_balance"]:,.2f}'),
        (True,                      "Commission",        f'${SS["commission_per_side"]:.4f}/contract/side'),
        (True,                      "Slippage",          f'{SS["slippage_ticks"]} tick(s) = ${slip_dol:.4f}/fill'),
        (True,                      "Date Range Mode",   SS["date_range_mode"]),
        (True,                      "Profit Target",     f'{SS["profit_target_mode"]} (${SS["profit_target_value"]:,.2f})'),
        (True,                      "Stop Loss",         f'{SS["stop_loss_mode"]} (${SS["stop_loss_value"]:,.2f})'),
        (True,                      "EOD Exit",          f'{SS["eod_exit_time"]}' if SS["eod_exit"] else "Disabled"),
    ]

    all_ok = checklist(items)

    if not all_ok:
        banner_warning("Complete all checklist items before running the backtest.")
        return

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        dry_run = st.toggle("Dry Run (validate config only, no simulation)", value=False)
    with c2:
        run_label = "🔍 VALIDATE CONFIG" if dry_run else "▶  RUN BACKTEST"
        run_btn = st.button(run_label, type="primary", use_container_width=True)
    with c3:
        reset_btn = st.button("🔄  RESET RESULTS", use_container_width=True)

    if reset_btn:
        SS["backtest_result"] = None
        SS["metrics"]         = None
        SS["engine_status"]   = "idle"
        SS["mc_result"]       = None
        st.rerun()

    if run_btn:
        if dry_run:
            banner_success(
                "Dry run complete — configuration is valid. "
                f"Will process {len(raw_df):,} bars with "
                f"round-trip cost ${rt_cost + slip_dol * 2:.4f}/contract."
            )
            return

        _run_backtest(raw_df, strategy, cfg)

    # ── Live status (after a run) ─────────────────────────────────────────────
    result = SS.get("backtest_result")
    if result:
        st.markdown("---")
        st.markdown('<div class="section-header">◈ ENGINE OUTPUT LOG</div>',
                    unsafe_allow_html=True)
        n_trades = len(result.trades) if result.trades is not None else 0
        log_text = (
            f"[COMPLETE] strategy    : {result.strategy_name}\n"
            f"[COMPLETE] period      : {result.start_date} → {result.end_date}\n"
            f"[COMPLETE] bars proc.  : {result.total_bars:,}\n"
            f"[COMPLETE] trades gen. : {n_trades:,}\n"
            f"[COMPLETE] runtime     : {result.runtime_secs:.2f}s\n"
        )
        if result.stderr:
            log_text += f"\n[STDERR]\n{result.stderr}"
        if result.compile_mode:
            log_text += "\n[MODE] C++ native compilation"

        terminal_log(log_text, height=260)


def _run_backtest(raw_df, strategy, cfg: BacktestConfig):
    """Execute the backtest and store results in session state."""
    SS["engine_status"] = "running"
    progress_bar = st.progress(0, text="Initialising backtest engine…")
    status_text  = st.empty()

    def progress_cb(pct: float, msg: str):
        p = min(int(pct), 99)
        progress_bar.progress(p, text=msg)
        status_text.markdown(
            f'<div style="font-family:monospace;font-size:0.76rem;'
            f'color:{COLORS["text_dim"]};">{msg}</div>',
            unsafe_allow_html=True,
        )

    try:
        df = _apply_date_filter(raw_df)
        if df is None or df.empty:
            banner_error("No data in the selected date range.")
            SS["engine_status"] = "error"
            return

        strategy_type = SS.get("strategy_type", "demo")

        # ── C++ native compilation path ────────────────────────────────────
        if strategy_type == "cpp" and SS["compile_mode"] and SS.get("cpp_source"):
            data_path = write_normalised_for_cpp(df, "data/processed/backtest_input.csv")
            result = run_cpp_backtest(SS["cpp_source"], data_path, cfg)
            if result.stderr and ("COMPILE ERROR" in result.stderr or "RUNTIME ERROR" in result.stderr):
                banner_warning("C++ compile/runtime error — falling back to Python engine…")
                result = BacktestEngine().run(df, strategy, cfg, progress_cb)
            elif result.stderr:
                banner_warning(f"C++ stderr: {result.stderr[:300]}")

        # ── Python strategy path (.py upload) ─────────────────────────────
        elif strategy_type == "python" and SS.get("py_source"):
            py_report = SS.get("py_parse_report")
            if py_report and py_report.is_valid and py_report.strategy_class:
                adapter = PyStrategyAdapter(py_report.strategy_class, _build_config_dict())
                strat_def = py_report.strategy_def or demo_strategy_definition()
                # Inject adapter into engine via monkey-patch on strategy_def
                strat_def._py_adapter = adapter
                result = BacktestEngine().run(df, strat_def, cfg, progress_cb,
                                              py_adapter=adapter)
            else:
                banner_warning("Python strategy not fully loaded — using demo fallback.")
                result = BacktestEngine().run(df, strategy, cfg, progress_cb)

        # ── Python-parsed C++ or demo path ────────────────────────────────
        else:
            result = BacktestEngine().run(df, strategy, cfg, progress_cb)

        progress_bar.progress(100, text="Computing performance metrics…")
        m = compute_all_metrics(
            result.trades,
            result.equity_curve,
            {**_build_config_dict(), **DEFAULT_COMBINE},
            risk_free_rate=SS["risk_free_rate"] / 100,
        )

        # Tag trades with volatility regime
        try:
            raw_df = SS.get("raw_df")
            if raw_df is not None and not result.trades.empty:
                result.trades = tag_trades_with_regime(result.trades, raw_df)
                regime_analytics = compute_regime_analytics(result.trades)
                SS["regime_analytics"] = regime_analytics
        except Exception as _re:
            SS["regime_analytics"] = None

        SS["backtest_result"] = result
        SS["metrics"]         = m
        SS["engine_status"]   = "complete"
        SS["last_run"]        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        progress_bar.empty()
        status_text.empty()
        n = len(result.trades)
        banner_success(
            f"Backtest complete — **{n:,} trades** | "
            f"Net P&L: **${m.get('net_pnl', 0):+,.2f}** | "
            f"Win Rate: **{m.get('win_rate_pct', 0):.1f}%** | "
            f"Sharpe: **{m.get('sharpe', 0):.2f}** | "
            f"Runtime: {result.runtime_secs:.2f}s"
        )

    except Exception as exc:
        SS["engine_status"] = "error"
        progress_bar.empty()
        banner_error(
            f"Backtest engine error: **{exc}**\n\n"
            "What to check:\n"
            "- Ensure CSV data is loaded and contains OHLCV columns\n"
            "- Verify the date range includes valid trading days\n"
            "- Check instrument selection matches your data\n\n"
            f"Technical detail: `{traceback.format_exc()[-500:]}`"
        )


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def render_section_5():
    section_header("RESULTS DASHBOARD",
                   "Primary output hub — equity curve, KPIs, trade statistics")

    result = SS.get("backtest_result")
    m      = SS.get("metrics")

    if result is None or m is None:
        banner_info("No backtest results yet. Go to **Section 4 — Backtest Engine** and run a backtest.")
        return

    scale  = SS.get("display_scale", 1)
    trades = result.trades
    equity = result.equity_curve

    # Scale P&L if display scale ≠ 1
    if scale != 1:
        m_scaled = {
            k: v * scale
            if isinstance(v, float) and any(x in k for x in
                                             ("pnl", "win", "loss", "dd", "fee",
                                              "comm", "slip", "value", "revenue",
                                              "mean", "std", "gross"))
            else v
            for k, v in m.items()
        }
        banner_info(f"Display scale: ×{scale} — all P&L figures scaled accordingly.")
    else:
        m_scaled = m

    # ── 6.1 KPI Header Bar ────────────────────────────────────────────────────
    kpi_card_row(m_scaled)

    # Strategy subtitle
    rep   = SS.get("parse_report")
    sname = rep.strategy.name if rep else "—"
    cfg   = _build_backtest_config()
    spec  = INSTRUMENTS[SS["instrument"]]
    dr    = f"{m.get('start_date','?')} → {m.get('end_date','?')}"
    strategy_subtitle(
        instrument     = SS["instrument"],
        strategy_name  = sname,
        config_label   = f"Fixed {SS['num_contracts']}×",
        date_range     = dr,
        pills=[
            (f"Stop: {SS['stop_loss_mode']}", "red"),
            (f"Target: {SS['profit_target_mode']}", "green"),
            (f"Size: {SS['sizing_mode']}", "blue"),
            (f"EOD: {'On' if SS['eod_exit'] else 'Off'}", "amber"),
        ],
    )

    # ── 6.2 Equity curve & drawdown ───────────────────────────────────────────
    st.markdown("---")
    if scale != 1:
        eq_scaled = equity.copy()
        eq_scaled["equity"]   = eq_scaled["equity"]   + (equity["equity"] - equity["equity"].iloc[0]) * (scale - 1)
        eq_scaled["drawdown"] = eq_scaled.get("drawdown", 0) * scale
    else:
        eq_scaled = equity

    show_ma = st.toggle("Overlay equity MA", value=False)
    ma_win  = st.slider("MA window (bars)", 10, 200, 50, 5) if show_ma else 50
    eq_fig  = equity_drawdown_chart(eq_scaled, show_ma=show_ma, ma_window=ma_win)
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── 6.3 Annual P&L bar chart + table ─────────────────────────────────────
    st.markdown("---")
    section_header("ANNUAL P&L BREAKDOWN")
    c1, c2 = st.columns([1.4, 1])
    annual = m.get("annual_table", pd.DataFrame())
    if not annual.empty and scale != 1:
        annual = annual.copy()
        annual["TOTAL P&L"] = annual["TOTAL P&L"] * scale
    with c1:
        if not annual.empty:
            st.plotly_chart(annual_pnl_chart(annual), use_container_width=True)
        else:
            st.info("No annual data available.")
    with c2:
        if not annual.empty:
            st.dataframe(
                annual.style.applymap(
                    lambda v: f"color: {COLORS['green']}" if isinstance(v, (int, float)) and v > 0
                    else f"color: {COLORS['red']}" if isinstance(v, (int, float)) and v < 0
                    else "",
                    subset=["TOTAL P&L"],
                ),
                use_container_width=True, height=340,
            )
            totals = {
                "YEAR": "TOTAL",
                "TRADES":  annual["TRADES"].sum(),
                "WIN RATE": f"{trades['net_pnl'].gt(0).mean()*100:.1f}%",
                "EXP/TRADE": f"${m_scaled.get('expected_value',0):+,.0f}",
                "TOTAL P&L": annual["TOTAL P&L"].sum() * scale,
            }
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.78rem;'
                f'color:{COLORS["green"]};padding:6px 0;">'
                f'CUMULATIVE: ${totals["TOTAL P&L"]:+,.0f} | '
                f'{totals["TRADES"]} trades | {totals["WIN RATE"]} win rate</div>',
                unsafe_allow_html=True,
            )

    # ── 6.4 Trade-level statistics ────────────────────────────────────────────
    st.markdown("---")
    section_header("TRADE-LEVEL STATISTICS")
    sc = scale
    rows = [
        ("Total Trades",             f"{int(m.get('total_trades',0)):,}",              "white"),
        ("Avg Trades / Active Day",  f"{m.get('avg_trades_per_day',0):.1f}",           "white"),
        ("Avg Trades / Session",     f"{m.get('avg_trades_per_session',0):.1f}",       "white"),
        ("Win Rate",                 f"{m.get('win_rate_pct',0):.2f}%",               "green"),
        ("Avg Winning Trade",        f"${m.get('avg_win',0)*sc:+,.2f}",               "green"),
        ("Avg Losing Trade",         f"${m.get('avg_loss',0)*sc:+,.2f}",              "red"),
        ("Payoff Ratio",             f"{m.get('payoff_ratio',0):.2f}",                "amber"),
        ("Profit Factor",            f"{m.get('profit_factor',0):.3f}",               "blue"),
        ("Expected Value / Trade",   f"${m.get('expected_value',0)*sc:+,.2f}",        "amber"),
        ("Largest Win",              f"${m.get('largest_win',0)*sc:+,.2f}",           "green"),
        ("Largest Loss",             f"${m.get('largest_loss',0)*sc:+,.2f}",          "red"),
        ("Max Consec. Wins",         f"{int(m.get('max_consec_wins',0))}",             "green"),
        ("Max Consec. Losses",       f"{int(m.get('max_consec_losses',0))}",           "red"),
        ("Avg Hold Time (all)",      f"{m.get('avg_hold_all',0):.1f} min",            "white"),
        ("Avg Hold Time (wins)",     f"{m.get('avg_hold_wins',0):.1f} min",           "green"),
        ("Avg Hold Time (losses)",   f"{m.get('avg_hold_losses',0):.1f} min",         "red"),
        ("Avg MAE",                  f"${m.get('avg_mae',0)*sc:+,.2f}",               "red"),
        ("Avg MFE",                  f"${m.get('avg_mfe',0)*sc:+,.2f}",               "green"),
        ("Time Stop Exit Rate",      f"{m.get('time_stop_pct',0):.1f}%",             "amber"),
        ("Total Gross Revenue",      f"${m.get('gross_pnl',0)*sc:+,.2f}",             "green"),
        ("Total Commissions Paid",   f"${m.get('total_commission',0)*sc:,.2f}",        "red"),
        ("Total Slippage Cost",      f"${m.get('total_slippage',0)*sc:,.2f}",          "red"),
        ("Net P&L After All Costs",  f"${m.get('net_pnl',0)*sc:+,.2f}",              "green" if m.get("net_pnl",0) >= 0 else "red"),
    ]
    c1, c2 = st.columns(2)
    with c1:
        metric_grid(rows[:12])
    with c2:
        metric_grid(rows[12:])


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — CANDLESTICK CHART
# ════════════════════════════════════════════════════════════════════════════

def render_section_6():
    section_header("CANDLESTICK CHART",
                   "TradingView-style chart with trade execution overlays")

    raw_df = SS.get("raw_df")
    result = SS.get("backtest_result")

    if raw_df is None:
        banner_info("Load CSV data first (Section 3).")
        return

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tf = st.selectbox("Timeframe", ["1m","5m","15m","30m","1h","4h","1d"], index=6)
    with c2:
        outcome_f = st.selectbox("Trade Filter", ["ALL","WIN","LOSS","TIME","EOD","GAP"])
    with c3:
        show_stops   = st.toggle("Show Stops",    True)
        show_targets = st.toggle("Show Targets",  True)
    with c4:
        show_shading = st.toggle("Show Shading",  True)
        show_volume  = st.toggle("Show Volume",   True)
        show_vwap    = st.toggle("Show VWAP",     False)

    # Resample
    try:
        resample_map = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min",
                        "1h":"1h","4h":"4h","1d":"1D"}
        rule = resample_map.get(tf, "1D")
        if rule != "1min" and len(raw_df) > 1000:
            chart_df = raw_df.resample(rule).agg({
                "open": "first","high": "max","low": "min",
                "close": "last","volume": "sum",
            }).dropna(subset=["close"])
        else:
            chart_df = raw_df.copy()
    except Exception:
        chart_df = raw_df.copy()

    # Signal filter
    trades_df = result.trades if result else None
    sig_names = []
    if trades_df is not None and not trades_df.empty:
        sig_names = list(trades_df["signal"].unique())
    sig_filter = st.multiselect("Signal Type Filter", sig_names, default=sig_names)

    # Jump to trade
    if trades_df is not None and not trades_df.empty:
        trade_nums = ["(none)"] + [f"Trade #{t}: {r['signal']} {r['entry_time']}"
                                   for t, (_, r) in
                                   enumerate(trades_df.iterrows())]
        jump = st.selectbox("Jump to Trade", trade_nums)
        if jump != "(none)":
            t_idx = trade_nums.index(jump) - 1
            t_row = trades_df.iloc[t_idx]
            et = pd.Timestamp(t_row["entry_time"])
            window = pd.Timedelta(hours=4)
            chart_df = chart_df[
                (chart_df.index >= et - window) &
                (chart_df.index <= et + window)
            ]

    try:
        fig = candlestick_chart(
            chart_df, trades_df,
            show_stops=show_stops, show_targets=show_targets,
            show_shading=show_shading, show_volume=show_volume,
            show_vwap=show_vwap, outcome_filter=outcome_f,
            signal_filter=sig_filter if sig_filter else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        banner_error(f"Chart error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MONTE CARLO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

def render_section_7():
    section_header("MONTE CARLO SIMULATOR",
                   "Forward-projection robustness analysis with stress testing")

    result = SS.get("backtest_result")
    if result is None or result.trades is None or result.trades.empty:
        banner_info("Run a backtest first (Section 4).")
        return

    # ── Configuration ─────────────────────────────────────────────────────────
    with st.expander("⚙️  SIMULATION CONFIGURATION", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Simulation Paths**")
            for n in [100, 500, 1_000, 5_000]:
                if st.button(str(n), key=f"mc_paths_{n}"):
                    SS["mc_num_paths"] = n
            n_paths = st.number_input("Custom paths", 10, 50_000,
                                       SS.get("mc_num_paths", 1_000), 100)
            SS["mc_num_paths"] = int(n_paths)
        with c2:
            st.markdown("**Horizon (Trading Days)**")
            for h in [30, 63, 126, 252, 504]:
                if st.button(f"{h}d", key=f"mc_hor_{h}"):
                    SS["mc_horizon"] = h
            horizon = st.number_input("Custom days", 5, 2520,
                                       SS.get("mc_horizon", 252), 10)
            SS["mc_horizon"] = int(horizon)
        with c3:
            method = st.selectbox("Sampling Method", METHODS,
                                   index=METHODS.index(SS.get("mc_method", METHODS[0])))
            SS["mc_method"] = method

        st.markdown("---")
        st.markdown("**Walk-Forward Variance Stressors**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            noise = st.slider("P&L Noise ±%", 0, 50, int(SS.get("mc_noise", 0)), 1)
            SS["mc_noise"] = noise
            wr_cut = st.slider("Win Rate Haircut %", 0, 50, int(SS.get("mc_wr_cut", 0)), 1)
            SS["mc_wr_cut"] = wr_cut
        with sc2:
            stop_inc = st.slider("Stop Size Increase %", 0, 100, int(SS.get("mc_stop_inc", 0)), 5)
            SS["mc_stop_inc"] = stop_inc
            size_red = st.slider("Size Reduction %", 0, 90, int(SS.get("mc_size_red", 0)), 5)
            SS["mc_size_red"] = size_red
        with sc3:
            trade_rem = st.slider("Random Trade Removal %", 0, 50, int(SS.get("mc_t_rem", 0)), 2)
            SS["mc_t_rem"] = trade_rem
            block_size = st.number_input("Block Bootstrap Size", 2, 50,
                                          int(SS.get("mc_block", 10))) if method == "Block Bootstrap" else 10
            SS["mc_block"] = block_size

    # ── Combine mode ──────────────────────────────────────────────────────────
    with st.expander("🎯  COMBINE / PROP FIRM CHALLENGE MODE"):
        combine_on = st.toggle("Simulate Prop Firm Combine Rules", SS.get("mc_combine", False))
        SS["mc_combine"] = combine_on
        if combine_on:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                SS["mc_daily_loss"]  = st.number_input("Daily Loss Limit $", 0.0, 1e6,
                                                         float(SS.get("mc_daily_loss", 3_000)), 100.0)
                SS["mc_max_dd"]      = st.number_input("Max Drawdown Limit $", 0.0, 1e6,
                                                         float(SS.get("mc_max_dd", 6_000)), 100.0)
            with cc2:
                SS["mc_profit_tgt"]  = st.number_input("Profit Target $", 0.0, 1e6,
                                                         float(SS.get("mc_profit_tgt", 12_000)), 100.0)
            with cc3:
                SS["mc_min_days"]    = st.number_input("Min Trading Days", 1, 90,
                                                         int(SS.get("mc_min_days", 5)))
                SS["mc_max_days"]    = st.number_input("Max Trading Days", 1, 90,
                                                         int(SS.get("mc_max_days", 30)))

    show_paths = st.toggle("Show individual sample paths (up to 100)", False)

    if st.button("🎲  RUN MONTE CARLO SIMULATION", type="primary"):
        mc_cfg = MonteCarloConfig(
            num_paths        = SS["mc_num_paths"],
            horizon_days     = SS["mc_horizon"],
            sampling_method  = SS["mc_method"],
            noise_pct        = SS.get("mc_noise", 0),
            win_rate_haircut = SS.get("mc_wr_cut", 0),
            stop_size_pct    = SS.get("mc_stop_inc", 0),
            size_reduction   = SS.get("mc_size_red", 0),
            trade_removal    = SS.get("mc_t_rem", 0),
            block_size       = SS.get("mc_block", 10),
            combine_mode     = SS.get("mc_combine", False),
            daily_loss_limit = SS.get("mc_daily_loss", 3_000),
            max_drawdown_limit= SS.get("mc_max_dd", 6_000),
            profit_target    = SS.get("mc_profit_tgt", 12_000),
            min_trading_days = SS.get("mc_min_days", 5),
            max_trading_days = SS.get("mc_max_days", 30),
            show_individual_paths = show_paths,
        )
        with st.spinner(f"Running {mc_cfg.num_paths:,} paths × {mc_cfg.horizon_days}d…"):
            t0 = time.perf_counter()
            try:
                mc_res = run_monte_carlo(result.trades, mc_cfg)
                elapsed = time.perf_counter() - t0
                SS["mc_result"] = mc_res
                SS["mc_config"] = mc_cfg
                banner_success(
                    f"Monte Carlo complete — {mc_cfg.num_paths:,} paths in {elapsed:.2f}s | "
                    f"Median final: ${mc_res.median_final:+,.0f} | "
                    f"Prob(profit): {mc_res.prob_profit:.1f}%"
                )
            except Exception as exc:
                banner_error(f"Monte Carlo error: {exc}")

    # ── Results ───────────────────────────────────────────────────────────────
    mc_res = SS.get("mc_result")
    if mc_res and len(mc_res.pct_50) > 0:
        combine_tgts = {
            "profit_target":     SS.get("mc_profit_tgt", 12_000),
            "max_drawdown_limit":SS.get("mc_max_dd", 6_000),
        } if SS.get("mc_combine") else None

        st.plotly_chart(
            monte_carlo_fan_chart(mc_res, show_paths, combine_tgts),
            use_container_width=True,
        )
        st.plotly_chart(monte_carlo_histogram(mc_res), use_container_width=True)

        # Stats table
        section_header("SIMULATION STATISTICS")
        mc_rows = [
            ("Median Final P&L",    f"${mc_res.median_final:+,.0f}",          "white"),
            ("Mean Final P&L",      f"${mc_res.mean_final:+,.0f}",            "white"),
            ("5th Pct Final",       f"${mc_res.p5_final:+,.0f}",              "red"),
            ("95th Pct Final",      f"${mc_res.p95_final:+,.0f}",             "green"),
            ("Probability Profit",  f"{mc_res.prob_profit:.1f}%",             "green"),
            ("Probability Ruin",    f"{mc_res.prob_ruin:.1f}%",               "red"),
            ("Median Max Drawdown", f"${mc_res.max_dd_median:,.0f}",          "amber"),
            ("95th Pct Max DD",     f"${mc_res.max_dd_p95:,.0f}",             "red"),
        ]
        if SS.get("mc_combine"):
            mc_rows += [
                ("Combine Pass Rate",   f"{mc_res.combine_pass_rate:.1f}%",   "green"),
                ("Combine Fail Rate",   f"{mc_res.combine_fail_rate:.1f}%",   "red"),
            ]
        metric_grid(mc_rows, title="MONTE CARLO STATISTICS")


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — ANNUAL & PERIODIC BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

def render_section_8():
    section_header("ANNUAL & PERIODIC BREAKDOWN",
                   "Year / month / quarter / DOW / time-of-day analytics")

    m = SS.get("metrics")
    if m is None:
        banner_info("Run a backtest first.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Yearly", "🗓️ Monthly Heatmap", "📊 Quarterly", "📆 Day-of-Week", "⏱️ Time-of-Day"
    ])

    with tab1:
        annual = m.get("annual_table", pd.DataFrame())
        if not annual.empty:
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.plotly_chart(annual_pnl_chart(annual), use_container_width=True)
            with c2:
                st.dataframe(annual, use_container_width=True, height=500)
        else:
            st.info("Not enough data for yearly breakdown.")

    with tab2:
        monthly = m.get("monthly_table", pd.DataFrame())
        if not monthly.empty:
            st.plotly_chart(monthly_heatmap(monthly), use_container_width=True)
            st.dataframe(monthly.style.background_gradient(
                cmap="RdYlGn", axis=None,
            ), use_container_width=True)
        else:
            st.info("Not enough data for monthly heatmap.")

    with tab3:
        trades_df = SS.get("backtest_result")
        if trades_df and not trades_df.trades.empty:
            st.plotly_chart(quarterly_chart(trades_df.trades), use_container_width=True)

    with tab4:
        dow = m.get("dow_table", pd.DataFrame())
        if not dow.empty:
            st.plotly_chart(dow_chart(dow), use_container_width=True)
            st.dataframe(dow, use_container_width=True, hide_index=True)

    with tab5:
        tod = m.get("tod_table", pd.DataFrame())
        if not tod.empty:
            st.plotly_chart(tod_chart(tod), use_container_width=True)
            st.dataframe(tod, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — SIGNAL HISTORY LOG
# ════════════════════════════════════════════════════════════════════════════

def render_section_9():
    section_header("SIGNAL HISTORY LOG",
                   "Trade log · Session P&L · Volatility Regime Profitability")

    result = SS.get("backtest_result")
    if result is None or result.trades is None or result.trades.empty:
        banner_info("No trades yet. Run a backtest first.")
        return

    trades = result.trades.copy()
    equity_df = result.equity_curve if result.equity_curve is not None else None
    regime_analytics = SS.get("regime_analytics", None)
    C = COLORS

    # ═══════════════════════════════════════════════════════
    # SUB-TABS
    # ═══════════════════════════════════════════════════════
    tab_trades, tab_session, tab_regime, tab_dist = st.tabs([
        "📋 Trade Log",
        "📅 Session P&L",
        "🌊 Volatility Regime Analytics",
        "📊 P&L Distribution",
    ])

    # ─── TAB 1: Trade Log ────────────────────────────────────
    with tab_trades:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sig_search = st.text_input("Filter by Signal", "", key="s9_sigsearch")
        with c2:
            outcome_search = st.selectbox("Filter by Outcome",
                ["ALL","WIN","LOSS","TIME","EOD","GAP","BLOCKED"], key="s9_outsearch")
        with c3:
            per_page = st.selectbox("Rows per page", [25, 50, 100, 250], index=1, key="s9_perpage")
        with c4:
            if "sig_page" not in SS: SS["sig_page"] = 0
            df_filt = trades.copy()
            if sig_search: df_filt = df_filt[df_filt["signal"].str.contains(sig_search, na=False)]
            if outcome_search != "ALL": df_filt = df_filt[df_filt["outcome"] == outcome_search]
            max_page = max(0, (len(df_filt) - 1) // per_page)
            SS["sig_page"] = st.number_input("Page", 0, max_page, min(int(SS["sig_page"]), max_page), 1, key="s9_page")

        signal_table(df_filt, SS["sig_page"], per_page, sig_search, outcome_search)

        st.markdown("---")
        csv_bytes = export_trade_log_csv(trades)
        st.download_button("📊  Download Full Trade Log (CSV)", csv_bytes,
            file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

        # Blocked signals
        st.markdown("---")
        section_header("BLOCKED SIGNALS LOG")
        blocked = trades[trades["outcome"] == "BLOCKED"] if "outcome" in trades.columns else pd.DataFrame()
        if not blocked.empty:
            lines = [f"[BLOCKED {r.get('entry_time','?')}] {r.get('signal','?')}  Q={r.get('q_score',0):.1f}"
                     for _,r in blocked.iterrows()]
            terminal_log("\n".join(lines), height=180)
        else:
            st.markdown('<div style="font-family:monospace;font-size:0.76rem;color:#555;">No blocked signals in this backtest.</div>',
                        unsafe_allow_html=True)

    # ─── TAB 2: Session P&L ──────────────────────────────────
    with tab_session:
        section_header("SESSION P&L BREAKDOWN", "Daily trading session results")

        if "entry_time" not in trades.columns:
            banner_info("No entry_time data available for session breakdown.")
        else:
            trades_copy = trades.copy()
            trades_copy["date"] = pd.to_datetime(trades_copy["entry_time"]).dt.date

            session_df = trades_copy.groupby("date").agg(
                trades   =("net_pnl","count"),
                wins     =("net_pnl", lambda x: (x>0).sum()),
                losses   =("net_pnl", lambda x: (x<0).sum()),
                session_pnl=("net_pnl","sum"),
                avg_pnl  =("net_pnl","mean"),
                best_trade=("net_pnl","max"),
                worst_trade=("net_pnl","min"),
            ).reset_index()
            session_df["win_rate"]   = (session_df["wins"] / session_df["trades"] * 100).round(1)
            session_df["cumulative"] = session_df["session_pnl"].cumsum().round(2)
            session_df["result"]     = session_df["session_pnl"].apply(lambda v: "✅ GREEN" if v>=0 else "❌ RED")

            # Regime column if available
            if "regime" in trades_copy.columns:
                top_regime = trades_copy.groupby(["date","regime"])["net_pnl"].sum().reset_index()
                top_regime = top_regime.loc[top_regime.groupby("date")["net_pnl"].idxmax()]
                top_regime = top_regime.rename(columns={"regime":"top_regime"})[["date","top_regime"]]
                session_df = session_df.merge(top_regime, on="date", how="left")

            # Summary KPIs
            green_days = (session_df["session_pnl"] >= 0).sum()
            red_days   = (session_df["session_pnl"] < 0).sum()
            total_days = len(session_df)
            best_day   = float(session_df["session_pnl"].max())
            worst_day  = float(session_df["session_pnl"].min())

            kc = st.columns(5)
            with kc[0]: st.metric("Total Sessions", f"{total_days}")
            with kc[1]: st.metric("Green Days",  f"{green_days}", delta=f"{green_days/max(total_days,1)*100:.0f}%")
            with kc[2]: st.metric("Red Days",    f"{red_days}",   delta=f"-{red_days/max(total_days,1)*100:.0f}%")
            with kc[3]: st.metric("Best Day",    f"${best_day:+,.0f}")
            with kc[4]: st.metric("Worst Day",   f"${worst_day:+,.0f}")

            st.markdown("---")

            # Session table
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.78rem;color:{C["green"]};'
                f'margin-bottom:6px;">◈ DAILY SESSION TABLE</div>',
                unsafe_allow_html=True)

            # Build styled HTML table
            header_cols = ["DATE","TRADES","WIN RATE","SESSION P&L","AVG P&L",
                           "BEST","WORST","CUMULATIVE","RESULT"]
            if "top_regime" in session_df.columns:
                header_cols.append("TOP REGIME")

            html = '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:0.74rem;">' 
            html += '<thead><tr style="background:#1A1A1A;border-bottom:1px solid #333;">' 
            for col in header_cols:
                html += f'<th style="padding:6px 10px;color:#888;text-align:left;">{col}</th>'
            html += '</tr></thead><tbody>'

            for i, row in session_df.iterrows():
                bg = "background:#0D0D0D;" if i%2==0 else ""
                pnl_color = C["green"] if row["session_pnl"] >= 0 else C["red"]
                cum_color  = C["green"] if row["cumulative"] >= 0 else C["red"]
                cells = [
                    (str(row["date"]), "#AAA"),
                    (str(int(row["trades"])), "#DDD"),
                    (f"{row['win_rate']:.1f}%", C["green"] if row["win_rate"]>=50 else C["red"]),
                    (f"${row['session_pnl']:+,.2f}", pnl_color),
                    (f"${row['avg_pnl']:+,.2f}", pnl_color),
                    (f"${row['best_trade']:+,.2f}", C["green"]),
                    (f"${row['worst_trade']:+,.2f}", C["red"]),
                    (f"${row['cumulative']:+,.2f}", cum_color),
                    (row["result"], pnl_color),
                ]
                if "top_regime" in session_df.columns:
                    reg = str(row.get("top_regime","—"))
                    rc  = REGIME_COLORS.get(reg, "#888")
                    cells.append((reg, rc))

                html += f'<tr style="{bg}border-bottom:1px solid #1E1E1E;">'
                for val, col_val in cells:
                    html += f'<td style="padding:5px 10px;color:{col_val};">{val}</td>'
                html += '</tr>'

            html += '</tbody></table></div>'
            st.markdown(html, unsafe_allow_html=True)

            # Session P&L bar chart
            st.markdown("---")
            import plotly.graph_objects as go
            bar_colors = [C["green"] if v>=0 else C["red"] for v in session_df["session_pnl"]]
            fig_sess = go.Figure(go.Bar(
                x=session_df["date"].astype(str),
                y=session_df["session_pnl"],
                marker_color=bar_colors,
                text=[f"${v:+,.0f}" for v in session_df["session_pnl"]],
                textposition="outside",
                textfont=dict(size=8),
            ))
            fig_sess.update_layout(
                height=280, title=dict(text="Daily Session P&L", font=dict(color=C["text"],size=12)),
                xaxis=dict(tickangle=-45, gridcolor=C["grid"]),
                yaxis=dict(tickformat="$,.0f", gridcolor=C["grid"]),
                margin=dict(l=55,r=10,t=40,b=80),
                paper_bgcolor=C["bg"], plot_bgcolor="#0F0F0F",
                font=dict(color=C["text"]),
            )
            st.plotly_chart(fig_sess, use_container_width=True, key="s9_sess_bar")

            # Download session data
            sess_bytes = session_df.to_csv(index=False).encode()
            st.download_button("📥 Download Session P&L (CSV)", sess_bytes,
                               file_name=f"session_pnl_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv")

    # ─── TAB 3: Volatility Regime Analytics ──────────────────
    with tab_regime:
        section_header("VOLATILITY REGIME PROFITABILITY",
                       "Most & least profitable market conditions for this strategy")

        if "regime" not in trades.columns:
            banner_info(
                "Regime data not yet attached. Re-run the backtest to auto-tag trades "
                "with volatility regime (TRENDING / ROTATIONAL / MEAN_REVERT / VOLATILE / CHOPPY)."
            )
        else:
            ra = regime_analytics or compute_regime_analytics(trades)
            breakdown = ra.get("breakdown", pd.DataFrame())
            best_reg  = ra.get("best_regime", "N/A")
            worst_reg = ra.get("worst_regime", "N/A")

            # Hero banners
            bc = st.columns(2)
            with bc[0]:
                bc_color = REGIME_COLORS.get(best_reg, COLORS["green"])
                st.markdown(
                    f'<div style="background:#0A1A0A;border:2px solid {bc_color};'
                    f'border-radius:10px;padding:14px 20px;text-align:center;">' 
                    f'<div style="font-family:monospace;font-size:0.78rem;color:#888;">MOST PROFITABLE REGIME</div>'
                    f'<div style="font-family:monospace;font-size:1.6rem;font-weight:900;color:{bc_color};">{best_reg}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            with bc[1]:
                wc_color = REGIME_COLORS.get(worst_reg, COLORS["red"])
                st.markdown(
                    f'<div style="background:#1A0A0A;border:2px solid {wc_color};'
                    f'border-radius:10px;padding:14px 20px;text-align:center;">'
                    f'<div style="font-family:monospace;font-size:0.78rem;color:#888;">LEAST PROFITABLE REGIME</div>'
                    f'<div style="font-family:monospace;font-size:1.6rem;font-weight:900;color:{wc_color};">{worst_reg}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            if not breakdown.empty:
                section_header("REGIME BREAKDOWN TABLE")
                # Styled regime table
                html2 = '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:0.76rem;">'
                html2 += '<thead><tr style="background:#1A1A1A;border-bottom:1px solid #333;">'
                for col in ["REGIME","TRADES","WIN RATE","AVG P&L","TOTAL P&L","PROFIT FACTOR","SHARE %"]:
                    html2 += f'<th style="padding:7px 12px;color:#888;text-align:left;">{col}</th>'
                html2 += '</tr></thead><tbody>'

                for _, row in breakdown.iterrows():
                    reg   = str(row["REGIME"])
                    rc    = REGIME_COLORS.get(reg, "#888")
                    tp    = float(str(row["TOTAL P&L"]).replace("$","").replace(",","").replace("+","")) if isinstance(row["TOTAL P&L"],(int,float)) else row["TOTAL P&L"]
                    tp_c  = COLORS["green"] if (isinstance(tp,float) and tp>=0) else COLORS["red"]
                    ap    = float(str(row["AVG P&L"]).replace("$","").replace(",","").replace("+","")) if isinstance(row["AVG P&L"],(int,float)) else 0
                    ap_c  = COLORS["green"] if (isinstance(ap,float) and ap>=0) else COLORS["red"]
                    wr    = str(row["WIN RATE"])
                    wr_c  = COLORS["green"] if float(wr.replace("%",""))>=50 else COLORS["red"]

                    html2 += f'<tr style="border-bottom:1px solid #1E1E1E;">'
                    html2 += f'<td style="padding:6px 12px;"><span style="background:{rc}22;border:1px solid {rc};border-radius:4px;padding:2px 8px;color:{rc};font-weight:700;">{reg}</span></td>'
                    html2 += f'<td style="padding:6px 12px;color:#DDD;">{row["TRADES"]}</td>'
                    html2 += f'<td style="padding:6px 12px;color:{wr_c};">{wr}</td>'
                    html2 += f'<td style="padding:6px 12px;color:{ap_c};">${ap:+,.2f}</td>' if isinstance(ap,float) else f'<td style="padding:6px 12px;color:{ap_c};">{row["AVG P&L"]}</td>'
                    html2 += f'<td style="padding:6px 12px;color:{tp_c};font-weight:700;">${tp:+,.2f}</td>' if isinstance(tp,float) else f'<td style="padding:6px 12px;color:{tp_c};">{row["TOTAL P&L"]}</td>'
                    html2 += f'<td style="padding:6px 12px;color:#00BFFF;">{row["PROFIT FACTOR"]}</td>'
                    html2 += f'<td style="padding:6px 12px;color:#888;">{row["SHARE %"]}%</td>'
                    html2 += '</tr>'

                html2 += '</tbody></table></div>'
                st.markdown(html2, unsafe_allow_html=True)

                # Regime P&L bar chart
                st.markdown("---")
                import plotly.graph_objects as go
                reg_names = breakdown["REGIME"].tolist()
                reg_pnl   = breakdown["TOTAL P&L"].tolist()
                reg_clrs  = [REGIME_COLORS.get(r, "#888") for r in reg_names]
                fig_reg = go.Figure(go.Bar(
                    x=reg_names, y=reg_pnl, marker_color=reg_clrs,
                    text=[f"${v:+,.0f}" for v in reg_pnl],
                    textposition="outside",
                    textfont=dict(color=COLORS["text"],size=10),
                ))
                fig_reg.update_layout(
                    height=340,
                    title=dict(text="Total P&L by Volatility Regime", font=dict(color=COLORS["text"],size=12)),
                    xaxis=dict(gridcolor=COLORS["grid"]),
                    yaxis=dict(tickformat="$,.0f", gridcolor=COLORS["grid"]),
                    margin=dict(l=55,r=10,t=45,b=40),
                    paper_bgcolor=COLORS["bg"], plot_bgcolor="#0F0F0F",
                    font=dict(color=COLORS["text"]),
                )
                st.plotly_chart(fig_reg, use_container_width=True, key="s9_reg_bar")

                # Win rate by regime
                wr_vals = [float(str(row["WIN RATE"]).replace("%","")) for _,row in breakdown.iterrows()]
                fig_wr = go.Figure(go.Bar(
                    x=reg_names, y=wr_vals,
                    marker_color=[COLORS["green"] if w>=50 else COLORS["red"] for w in wr_vals],
                    text=[f"{w:.1f}%" for w in wr_vals], textposition="outside",
                    textfont=dict(color=COLORS["text"],size=10),
                ))
                fig_wr.add_hline(y=50, line=dict(color=COLORS["amber"],dash="dash",width=1.5),
                                 annotation_text="50% breakeven",annotation_font_color=COLORS["amber"])
                fig_wr.update_layout(
                    height=300,
                    title=dict(text="Win Rate by Volatility Regime", font=dict(color=COLORS["text"],size=12)),
                    yaxis=dict(tickformat=".0f", ticksuffix="%", gridcolor=COLORS["grid"]),
                    margin=dict(l=55,r=10,t=45,b=40),
                    paper_bgcolor=COLORS["bg"], plot_bgcolor="#0F0F0F",
                    font=dict(color=COLORS["text"]),
                )
                st.plotly_chart(fig_wr, use_container_width=True, key="s9_wr_bar")

                # Download
                reg_bytes = breakdown.to_csv(index=False).encode()
                st.download_button("📥 Download Regime Analytics (CSV)", reg_bytes,
                                   file_name="regime_analytics.csv", mime="text/csv")

    # ─── TAB 4: P&L Distribution ─────────────────────────────
    with tab_dist:
        section_header("P&L DISTRIBUTION", "Return distribution shape, skewness and tail risk")
        pd1,pd2,pd3 = st.columns(3)
        with pd1: dsplit = st.selectbox("Split by",["signal","direction","regime","none"],key="s9_dsplit")
        with pd2: ddaily = st.toggle("Daily returns",equity_df is not None,key="s9_ddaily")
        with pd3: dbins  = st.slider("Bins",20,120,60,5,key="s9_dbins")

        fig_dist = build_pnl_distribution(
            trades, equity_df, dsplit,
            ddaily and equity_df is not None,
            dbins, "P&L Distribution — Signal Log"
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="s9_distfig")

        st.markdown("---")
        dist_stats = build_return_stats_table(trades, equity_df)
        if not dist_stats.empty:
            section_header("DISTRIBUTION STATISTICS")
            st.dataframe(dist_stats, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — RISK METRICS PANEL
# ════════════════════════════════════════════════════════════════════════════

def render_section_10():
    section_header("RISK METRICS PANEL",
                   "Full suite of institutional-grade risk & return statistics")

    m = SS.get("metrics")
    if m is None:
        banner_info("Run a backtest first.")
        return

    t1, t2, t3, t4, t5 = st.tabs([
        "📈 Return", "⚖️ Risk-Adjusted", "📉 Drawdown",
        "🎯 Trade Quality", "💸 Costs"
    ])

    with t1:
        section_header("RETURN METRICS")
        metric_grid([
            ("Total Net P&L ($)",        f"${m.get('net_pnl',0):+,.2f}",           "green" if m.get("net_pnl",0) >= 0 else "red"),
            ("Total Net P&L (%)",        f"{m.get('net_pnl_pct',0):+.2f}%",        "green" if m.get("net_pnl_pct",0) >= 0 else "red"),
            ("Annualised Return (%)",    f"{m.get('annualised_return_pct',0):.2f}%","green" if m.get("annualised_return_pct",0) >= 0 else "red"),
            ("CAGR (%)",                 f"{m.get('cagr_pct',0):.2f}%",            "green" if m.get("cagr_pct",0) >= 0 else "red"),
            ("Backtest Period",          f"{m.get('years',0):.2f} years",           "white"),
            ("Date Range",               f"{m.get('start_date','?')} → {m.get('end_date','?')}", "white"),
            ("Daily P&L Mean ($)",       f"${m.get('daily_pnl_mean',0):+,.2f}",    "white"),
            ("Daily P&L Std Dev ($)",    f"${m.get('daily_pnl_std',0):,.2f}",      "white"),
            ("Gross P&L ($)",            f"${m.get('gross_pnl',0):+,.2f}",         "white"),
            ("Total Fees Paid ($)",      f"${m.get('total_fees',0):,.2f}",          "red"),
        ])

    with t2:
        section_header("RISK-ADJUSTED RETURN METRICS")
        rfr = SS.get("risk_free_rate", 5.0)
        SS["risk_free_rate"] = st.number_input(
            "Risk-free rate (%)", 0.0, 20.0, float(rfr), 0.25
        )
        metric_grid([
            ("Sharpe Ratio (annualised)",   f"{m.get('sharpe',0):.4f}",      "blue"),
            ("Sortino Ratio (annualised)",  f"{m.get('sortino',0):.4f}",     "blue"),
            ("3-Year Sharpe",               f"{m.get('sharpe_3yr',0):.4f}",  "blue"),
            ("10-Year Sharpe",              f"{m.get('sharpe_10yr',0):.4f}", "blue"),
            ("Calmar Ratio",                f"{m.get('calmar',0):.4f}",      "amber"),
            ("MAR Ratio",                   f"{m.get('mar',0):.4f}",         "amber"),
            ("Omega Ratio",                 f"{m.get('omega',0):.4f}",       "green"),
            ("Sterling Ratio",              f"{m.get('sterling',0):.4f}",    "amber"),
        ])

    with t3:
        section_header("DRAWDOWN METRICS")
        metric_grid([
            ("Maximum Drawdown ($)",         f"${m.get('max_drawdown_dollars',0):,.2f}",  "red"),
            ("Maximum Drawdown (%)",         f"{m.get('max_drawdown_pct',0):.2f}%",       "red"),
            ("Max DD Duration (days)",       f"{m.get('max_dd_duration_days',0):,}",      "red"),
            ("Average Drawdown ($)",         f"${m.get('avg_drawdown_dollars',0):,.2f}",  "amber"),
            ("Recovery Factor",              f"{m.get('recovery_factor',0):.4f}",         "green"),
            ("Ulcer Index",                  f"{m.get('ulcer_index',0):.4f}",             "amber"),
        ])

    with t4:
        section_header("TRADE QUALITY METRICS")
        metric_grid([
            ("Win Rate (%)",                f"{m.get('win_rate_pct',0):.2f}%",           "green"),
            ("Profit Factor",               f"{m.get('profit_factor',0):.4f}",           "blue"),
            ("Payoff Ratio",                f"{m.get('payoff_ratio',0):.4f}",            "amber"),
            ("Expected Value ($)",          f"${m.get('expected_value',0):+,.2f}",       "amber"),
            ("Kelly Criterion (full %)",    f"{m.get('kelly_full_pct',0):.2f}%",         "amber"),
            ("Half-Kelly (%)",              f"{m.get('kelly_half_pct',0):.2f}%",         "amber"),
            ("Van Tharp SQN",              f"{m.get('sqn',0):.4f}",                     "blue"),
            ("Z-Score",                     f"{m.get('z_score',0):.4f}",                "white"),
            ("Breakeven Win Rate",          f"{m.get('breakeven_win_rate',0):.2f}%",     "white"),
        ])

    with t5:
        section_header("COST ANALYSIS")
        metric_grid([
            ("Total Commissions ($)",       f"${m.get('total_commission',0):,.2f}",      "red"),
            ("Total Slippage ($)",          f"${m.get('total_slippage',0):,.2f}",        "red"),
            ("Total Fees ($)",              f"${m.get('total_fees',0):,.2f}",            "red"),
            ("Costs as % of Gross P&L",     f"{m.get('costs_pct_of_gross',0):.2f}%",    "amber"),
            ("Breakeven Win Rate",          f"{m.get('breakeven_win_rate',0):.2f}%",     "amber"),
        ])


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — EXPORT & REPORTING
# ════════════════════════════════════════════════════════════════════════════

def render_section_11():
    section_header("EXPORT & REPORTING",
                   "Download results as PDF, CSV, JSON for offline analysis")

    result   = SS.get("backtest_result")
    m        = SS.get("metrics", {})
    mc_res   = SS.get("mc_result")
    cfg_dict = _build_config_dict()

    rep  = SS.get("parse_report")
    sname = rep.strategy.name if rep else "Strategy"

    if result is None:
        banner_info("Run a backtest first to generate export data.")
        return

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Trade log
        if result.trades is not None and not result.trades.empty:
            trade_bytes = export_trade_log_csv(result.trades)
            st.download_button(
                "📊  Download Trade Log (CSV)",
                trade_bytes,
                file_name=f"trade_log_{sname}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📊  Trade Log (CSV) — No trades", disabled=True,
                      use_container_width=True)

        # Equity curve
        if result.equity_curve is not None and not result.equity_curve.empty:
            eq_bytes = export_equity_curve_csv(result.equity_curve)
            st.download_button(
                "📈  Download Equity Curve (CSV)",
                eq_bytes,
                file_name=f"equity_{sname}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Monte Carlo
        if mc_res is not None:
            mc_bytes = export_monte_carlo_csv(mc_res)
            st.download_button(
                "🎲  Download Monte Carlo Results (CSV)",
                mc_bytes,
                file_name=f"montecarlo_{sname}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col2:
        # Config JSON
        cfg_bytes = export_config_json(cfg_dict, sname)
        st.download_button(
            "📋  Download Config Snapshot (JSON)",
            cfg_bytes,
            file_name=f"config_{sname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

        # PDF report
        st.markdown("---")
        st.markdown("**📄 Full PDF Report**")
        include_charts = st.toggle("Include charts in PDF (requires kaleido)", True)
        if st.button("🖨️  Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Building PDF report…"):
                try:
                    figs = {}
                    if include_charts and result.equity_curve is not None:
                        try:
                            figs["Equity Curve"] = equity_drawdown_chart(result.equity_curve)
                            annual = m.get("annual_table", pd.DataFrame())
                            if not annual.empty:
                                figs["Annual P&L"] = annual_pnl_chart(annual)
                            if mc_res:
                                figs["Monte Carlo"] = monte_carlo_fan_chart(mc_res)
                        except Exception:
                            figs = {}

                    pdf_bytes = export_pdf_report(
                        metrics       = m,
                        trades_df     = result.trades,
                        equity_df     = result.equity_curve,
                        mc_result     = mc_res,
                        config_dict   = cfg_dict,
                        strategy_name = sname,
                        figures       = figs if include_charts else None,
                    )
                    st.download_button(
                        "⬇️  Download PDF Report",
                        pdf_bytes,
                        file_name=f"backtest_report_{sname}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    banner_success("PDF report generated successfully.")
                except Exception as exc:
                    banner_error(f"PDF generation failed: {exc}\n\n"
                                 "Install `reportlab` and `kaleido` for full PDF support.")

    # ── Options data snapshot export ──────────────────────────────────────────
    st.markdown("---")
    section_header("OPTIONS DATA EXPORT")
    banner_info(
        "Options chain data is fetched live from yfinance. "
        "For historical EOD options backtesting, download free datasets from "
        "**github.com/optionstrat** or **Kaggle** and place .parquet files in "
        "`data/processed/options_eod_spy.parquet` / `options_eod_qqq.parquet`."
    )
    c1, c2 = st.columns(2)
    with c1:
        opt_und = st.selectbox("Options Underlying", ["SPY", "QQQ"])
        opt_dte = st.multiselect("DTE Filter", [0, 1, 2, 3, 5, 7], default=[0, 1, 2])
    with c2:
        if st.button("📡 Fetch Current Options Chain"):
            try:
                with st.spinner(f"Fetching {opt_und} options…"):
                    chain_df = get_options_chain(opt_und, dte_filter=opt_dte or None)
                if chain_df.empty:
                    banner_warning("No options data returned. Try without a DTE filter.")
                else:
                    st.dataframe(chain_df.head(200), use_container_width=True)
                    csv_opt = chain_df.to_csv(index=False).encode()
                    st.download_button(
                        f"⬇️  Download {opt_und} Chain (CSV)",
                        csv_opt,
                        file_name=f"{opt_und}_chain_{date.today()}.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                banner_error(f"Options fetch error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════

def main():
    section = SS["section"]
    try:
        if section == "1. CONFIGURATION & INPUTS":
            render_section_1()
        elif section == "2. STRATEGY LOADER":
            render_section_2()
        elif section == "3. DATA LOADER":
            render_section_3()
        elif section == "4. BACKTEST ENGINE":
            render_section_4()
        elif section == "5. RESULTS DASHBOARD":
            render_section_5()
        elif section == "6. CANDLESTICK CHART":
            render_section_candlestick(SS)
        elif section == "7. MONTE CARLO SIMULATOR":
            render_section_montecarlo(SS)
        elif section == "8. ANNUAL & PERIODIC BREAKDOWN":
            render_section_8()
        elif section == "9. SIGNAL HISTORY LOG":
            render_section_9()
        elif section == "10. RISK METRICS PANEL":
            render_section_10()
        elif section == "11. EXPORT & REPORTING":
            render_section_11()
        elif section == "12. CODE REMODELER":
            render_section_remodeler(SS)
    except Exception as exc:
        banner_error(
            f"**Unhandled error in {section}**\n\n"
            f"Error: `{exc}`\n\n"
            "What to do:\n"
            "- Check your data and strategy files are valid\n"
            "- Try resetting results in Section 4\n"
            "- Reload the app if the issue persists\n\n"
            f"Detail: `{traceback.format_exc()[-800:]}`"
        )


if __name__ == "__main__":
    main()
else:
    main()
