"""
section_montecarlo.py  v2
══════════════════════════
Section 7: Monte Carlo Simulator

Tabs
────
1. Simulation Config  — sampling method, paths, horizon, stressors
2. EVALUATION Mode    — prop firm challenge pass/fail simulation
3. PHASE TWO RUN-UP   — post-evaluation live account: balance growth,
                         payout milestones, drawdown management
4. Risk Metrics       — overlay of backtest metrics on simulation
5. Date Range         — filter trade sample window
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import COLORS, PLOTLY_DARK
from monte_carlo import MonteCarloConfig, run_monte_carlo, METHODS, MCResult
from ui_components import (
    section_header, banner_info, banner_warning,
    banner_error, banner_success, metric_grid,
)

_G  = COLORS["green"]
_R  = COLORS["red"]
_A  = COLORS["amber"]
_B  = COLORS["blue"]
_W  = COLORS["text"]
_C  = COLORS["card_bg"]
_BG = COLORS["bg"]


# ─────────────────────────────────────────────────────────────────────────────
# Fan chart builder
# ─────────────────────────────────────────────────────────────────────────────

def _fan_frame(paths, step, combine_targets=None):
    vis  = paths[:, :step + 1]
    days = np.arange(step + 1)
    p5   = np.percentile(vis,  5, axis=0)
    p25  = np.percentile(vis, 25, axis=0)
    p50  = np.percentile(vis, 50, axis=0)
    p75  = np.percentile(vis, 75, axis=0)
    p95  = np.percentile(vis, 95, axis=0)

    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(0,200,80,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(0,255,136,0.16)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=p50, mode="lines", name="Median",
        line=dict(color=_W, width=2.5),
    ))
    rng   = np.random.default_rng(0)
    n_vis = min(60, len(paths))
    for i in rng.choice(len(paths), n_vis, replace=False):
        fig.add_trace(go.Scatter(
            x=days, y=vis[i], mode="lines",
            line=dict(color="rgba(0,255,136,0.05)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_hline(y=0, line=dict(color="#333", width=1))
    if combine_targets:
        fig.add_hline(y=combine_targets.get("profit_target", 0),
                      line=dict(color=_G, dash="dash", width=1.5),
                      annotation_text="✓ Profit Target", annotation_font_color=_G)
        fig.add_hline(y=-abs(combine_targets.get("max_drawdown_limit", 0)),
                      line=dict(color=_R, dash="dash", width=1.5),
                      annotation_text="✗ Max DD", annotation_font_color=_R)
    fin = vis[:, -1]
    fig.update_layout(
        height=440,
        title=dict(
            text=(f"Monte Carlo — Day {step}  |  {len(paths):,} paths  |  "
                  f"Median ${np.median(fin):+,.0f}  |  "
                  f"P(profit) {(fin>0).mean()*100:.1f}%"),
            font=dict(color=_W, size=12),
        ),
        xaxis_title="Trading Day",
        yaxis=dict(tickformat="$,.0f", gridcolor=COLORS["grid"]),
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        **PLOTLY_DARK,
    )
    return fig


def _histogram_frame(final, label=""):
    profit = final[final >= 0]
    loss   = final[final  < 0]
    fig    = go.Figure()
    if len(profit): fig.add_trace(go.Histogram(x=profit, marker_color=_G, opacity=0.72, nbinsx=50, name="Profit"))
    if len(loss):   fig.add_trace(go.Histogram(x=loss,   marker_color=_R, opacity=0.72, nbinsx=30, name="Loss"))
    for pv, lbl, col in [(np.percentile(final,5),"5th",_R),(np.median(final),"Med",_W),(np.percentile(final,95),"95th",_G)]:
        fig.add_vline(x=pv, line=dict(color=col, dash="dash", width=1.5),
                      annotation_text=lbl, annotation_font_color=col)
    fig.update_layout(height=260, title=dict(text=f"Final Equity Distribution {label}", font=dict(color=_W,size=11)),
                      xaxis=dict(tickformat="$,.0f"), barmode="overlay",
                      margin=dict(l=55,r=15,t=40,b=30), **PLOTLY_DARK)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION stats
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_paths(paths, daily_loss_lim, max_dd_lim, profit_target, min_days, max_days):
    n = len(paths)
    passes = fail_dd = fail_dl = fail_time = 0
    pass_days = []
    for i in range(n):
        path   = paths[i]
        peak   = 0.0
        failed = won = False
        for day in range(1, min(paths.shape[1], max_days + 1)):
            dpnl = path[day] - path[day-1]
            cum  = path[day]
            peak = max(peak, cum)
            if dpnl < -daily_loss_lim:         fail_dl += 1; failed = True; break
            if (peak - cum) >= max_dd_lim:     fail_dd += 1; failed = True; break
            if cum >= profit_target and day >= min_days:
                won = True; pass_days.append(day); break
        if not failed and not won: fail_time += 1
        elif not failed and won:   passes    += 1
    return {
        "pass_rate":      passes   / n * 100,
        "fail_dd_rate":   fail_dd  / n * 100,
        "fail_dl_rate":   fail_dl  / n * 100,
        "fail_time_rate": fail_time/ n * 100,
        "avg_pass_day":   float(np.mean(pass_days)) if pass_days else 0.0,
        "passes": passes, "total_paths": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE TWO: Run-Up simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_phase_two(
    trade_pnl:     np.ndarray,
    start_balance: float,
    trailing_dd_limit: float,
    daily_loss_limit:  float,
    payout_levels:     list[float],
    payout_withdrawal: float,
    target_balance:    float,
    n_paths:    int,
    horizon:    int,
    seed:       int = 42,
) -> dict:
    """
    Simulate post-evaluation live account trading.

    Rules
    -----
    - Account starts at start_balance (e.g. $50,000 funded)
    - Trailing drawdown: if account drops trailing_dd_limit from peak → blown
    - Daily loss limit: daily loss > daily_loss_limit → trading halted for day
    - Payout milestones: when balance reaches a payout_level → withdraw payout_withdrawal
    - Target: reach target_balance (next milestone / double the account)
    """
    rng    = np.random.default_rng(seed)
    tpd    = max(1.0, len(trade_pnl) / max(horizon // 5, 1))

    paths_eq   = np.zeros((n_paths, horizon + 1))  # equity relative to start
    blown      = np.zeros(n_paths, dtype=bool)
    payout_counts = np.zeros(n_paths, dtype=int)
    reached_target= np.zeros(n_paths, dtype=bool)
    days_to_target= np.full(n_paths, horizon + 1)

    for i in range(n_paths):
        balance      = start_balance
        peak_balance = start_balance
        total_pnl    = 0.0
        payouts_hit  = set()
        paths_eq[i, 0] = 0.0

        n_trades = int(round(horizon * tpd))
        sample   = rng.choice(trade_pnl, size=n_trades, replace=True)
        daily    = np.array_split(sample, horizon)

        for day, block in enumerate(daily, 1):
            if blown[i]: break

            # Simulate day's trading
            day_pnl   = float(block.sum())

            # Daily loss circuit breaker
            if day_pnl < -daily_loss_limit:
                day_pnl = -daily_loss_limit  # halted at limit

            balance      += day_pnl
            total_pnl    += day_pnl
            peak_balance  = max(peak_balance, balance)

            # Trailing drawdown check
            if (peak_balance - balance) >= trailing_dd_limit:
                blown[i] = True
                paths_eq[i, day:] = total_pnl
                break

            # Payout milestone check
            for level in payout_levels:
                if level not in payouts_hit and balance >= level:
                    balance          -= payout_withdrawal
                    total_pnl        -= payout_withdrawal
                    payout_counts[i] += 1
                    payouts_hit.add(level)

            # Target check
            if not reached_target[i] and balance >= target_balance:
                reached_target[i]  = True
                days_to_target[i]  = day

            paths_eq[i, day] = total_pnl

    final = paths_eq[:, -1]
    active = ~blown

    return {
        "paths":           paths_eq,
        "final":           final,
        "blown_rate":      blown.mean() * 100,
        "survival_rate":   active.mean() * 100,
        "target_rate":     reached_target.mean() * 100,
        "avg_payouts":     float(payout_counts.mean()),
        "total_withdrawn": float(payout_counts.mean() * payout_withdrawal),
        "median_final":    float(np.median(final)),
        "p5_final":        float(np.percentile(final, 5)),
        "p95_final":       float(np.percentile(final, 95)),
        "avg_days_target": float(np.mean(days_to_target[reached_target]) if reached_target.any() else 0),
        "n_paths":         n_paths,
    }


def _phase_two_chart(result: dict, start_balance: float, payout_levels: list,
                     trailing_dd: float, target: float) -> go.Figure:
    paths = result["paths"]
    n     = len(paths)
    days  = np.arange(paths.shape[1])
    p5    = np.percentile(paths,  5, axis=0)
    p25   = np.percentile(paths, 25, axis=0)
    p50   = np.percentile(paths, 50, axis=0)
    p75   = np.percentile(paths, 75, axis=0)
    p95   = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(0,191,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(0,191,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75th pct", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=p50, mode="lines", name="Median Balance Growth",
        line=dict(color=_B, width=2.5),
    ))
    rng = np.random.default_rng(1)
    for i in rng.choice(n, min(50, n), replace=False):
        fig.add_trace(go.Scatter(
            x=days, y=paths[i], mode="lines",
            line=dict(color="rgba(0,191,255,0.05)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))
    # Threshold lines
    fig.add_hline(y=0, line=dict(color="#333", width=1))
    fig.add_hline(y=-trailing_dd, line=dict(color=_R, dash="dash", width=1.5),
                  annotation_text="✗ Account Blown (Trailing DD)", annotation_font_color=_R)
    fig.add_hline(y=target - start_balance, line=dict(color=_G, dash="dot", width=1.5),
                  annotation_text="🎯 Balance Target", annotation_font_color=_G)
    for lvl in payout_levels:
        fig.add_hline(y=lvl - start_balance, line=dict(color=_A, dash="dot", width=1),
                      annotation_text=f"💰 Payout ${lvl:,.0f}", annotation_font_color=_A)
    fig.update_layout(
        height=460,
        title=dict(text=f"PHASE TWO RUN-UP — {n:,} paths  |  "
                        f"Survival {result['survival_rate']:.1f}%  |  "
                        f"Target Hit {result['target_rate']:.1f}%",
                   font=dict(color=_B, size=12)),
        xaxis_title="Trading Day",
        yaxis=dict(tickformat="$,.0f", title="P&L from Start", gridcolor=COLORS["grid"]),
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        **PLOTLY_DARK,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main section render
# ─────────────────────────────────────────────────────────────────────────────

def render_section_montecarlo(SS: dict):
    section_header(
        "MONTE CARLO SIMULATOR",
        "Animated walk-paths · EVALUATION · PHASE TWO RUN-UP · Stress testing",
    )

    result = SS.get("backtest_result")
    if result is None or result.trades is None or result.trades.empty:
        banner_info("Run a backtest first (Section 4).")
        return

    trades = result.trades.copy()

    # ══════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════
    tab_sim, tab_eval, tab_p2, tab_risk, tab_date = st.tabs([
        "📊 Simulation Config",
        "🎯 EVALUATION Mode",
        "🚀 PHASE TWO — RUN-UP",
        "📉 Risk Metrics",
        "📅 Date Range",
    ])

    # ─── TAB 1: Simulation Config ─────────────────────────────────
    with tab_sim:
        _card_open("⬡ SIMULATION PARAMETERS", _G)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Paths**")
            for n in [100, 500, 1_000, 5_000]:
                if st.button(f"{n:,}", key=f"mc_n_{n}"): SS["mc_num_paths"] = n
            SS["mc_num_paths"] = int(st.number_input("Custom", 10, 50_000,
                int(SS.get("mc_num_paths", 1_000)), 100, key="mc_paths_inp"))
        with c2:
            st.markdown("**Horizon (days)**")
            for h in [21, 63, 126, 252, 504]:
                if st.button(f"{h}d", key=f"mc_h_{h}"): SS["mc_horizon"] = h
            SS["mc_horizon"] = int(st.number_input("Custom", 5, 2520,
                int(SS.get("mc_horizon", 252)), 5, key="mc_hor_inp"))
        with c3:
            SS["mc_method"] = st.selectbox("Sampling Method", METHODS,
                index=METHODS.index(SS.get("mc_method", METHODS[0])), key="mc_method_sel")
        _card_close()

        _card_open("⬡ VARIANCE STRESSORS", _A)
        s1, s2, s3 = st.columns(3)
        with s1:
            SS["mc_noise"]    = st.slider("P&L Noise ±%",       0, 50, int(SS.get("mc_noise",   0)), 1, key="mc_ns")
            SS["mc_wr_cut"]   = st.slider("Win-Rate Haircut %",  0, 50, int(SS.get("mc_wr_cut",  0)), 1, key="mc_wc")
        with s2:
            SS["mc_stop_inc"] = st.slider("Stop Increase %",     0,100, int(SS.get("mc_stop_inc",0)), 5, key="mc_si")
            SS["mc_size_red"] = st.slider("Size Reduction %",    0, 90, int(SS.get("mc_size_red",0)), 5, key="mc_sr")
        with s3:
            SS["mc_t_rem"]    = st.slider("Trade Removal %",     0, 50, int(SS.get("mc_t_rem",  0)), 2, key="mc_tr")
        _card_close()

    # ─── TAB 2: EVALUATION ────────────────────────────────────────
    with tab_eval:
        _card_open("🎯 PROP FIRM EVALUATION — CHALLENGE RULES", _B)
        eval_on = st.toggle("Activate EVALUATION Mode", SS.get("mc_eval_on", False), key="mc_eval_tog")
        SS["mc_eval_on"] = eval_on
        if eval_on:
            _card_close()
            _card_open("◈ ACCOUNT & CAPITAL", _G)
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Starting Account Balance**")
                for amt in [25_000, 50_000, 100_000, 150_000, 200_000]:
                    if st.button(f"${amt//1000}k", key=f"ev_b_{amt}"): SS["mc_eval_balance"] = float(amt)
                SS["mc_eval_balance"] = st.number_input("Custom ($)", 1_000.0, 5e6,
                    float(SS.get("mc_eval_balance", 50_000)), 1_000.0, key="ev_bal")
                mc1, mc2 = st.columns(2)
                with mc1: SS["mc_min_days"] = st.number_input("Min days", 1, 90, int(SS.get("mc_min_days",5)), key="ev_md")
                with mc2: SS["mc_max_days"] = st.number_input("Max days", 1, 90, int(SS.get("mc_max_days",30)), key="ev_xd")
            with e2:
                st.markdown("**Profit Target & Limits**")
                SS["mc_eval_profit_tgt"] = st.number_input("Profit Target ($)", 0.0, 1e6,
                    float(SS.get("mc_eval_profit_tgt", 6_000)), 100.0, key="ev_pt")
                SS["mc_eval_daily_loss"] = st.number_input("Daily Loss Limit ($)", 0.0, 5e5,
                    float(SS.get("mc_eval_daily_loss", 3_000)), 100.0, key="ev_dl")
                SS["mc_eval_max_dd"]     = st.number_input("Max Drawdown Limit ($)", 0.0, 5e5,
                    float(SS.get("mc_eval_max_dd", 6_000)), 100.0, key="ev_dd")
            _card_close()
        else:
            _card_close()

    # ─── TAB 3: PHASE TWO — RUN-UP ────────────────────────────────
    with tab_p2:
        _card_open("🚀 PHASE TWO — POST-EVALUATION LIVE ACCOUNT RUN-UP", _B)
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.74rem;color:#666;">'
            f'Simulate trading a funded live account after passing evaluation. '
            f'Tracks balance growth, payout milestones, drawdown risk, and account longevity.</div>',
            unsafe_allow_html=True,
        )
        p2_on = st.toggle("Activate PHASE TWO Simulation", SS.get("mc_p2_on", False), key="mc_p2_tog")
        SS["mc_p2_on"] = p2_on
        _card_close()

        if p2_on:
            _card_open("◈ LIVE ACCOUNT — ACCOUNT & CAPITAL SETTINGS", _G)
            p2a, p2b = st.columns(2)
            with p2a:
                st.markdown("**Funded Account Starting Balance**")
                for amt in [10_000, 25_000, 50_000, 100_000, 150_000, 200_000]:
                    if st.button(f"${amt//1000}k", key=f"p2_b_{amt}"): SS["mc_p2_balance"] = float(amt)
                SS["mc_p2_balance"] = st.number_input("Custom ($)", 1_000.0, 2e6,
                    float(SS.get("mc_p2_balance", 50_000)), 1_000.0, key="p2_bal")

                st.markdown("**Trailing Drawdown Limit (account blown if breached)**")
                for pct in [3, 5, 6, 8, 10]:
                    bal = SS.get("mc_p2_balance", 50_000)
                    if st.button(f"-{pct}% (${bal*pct//100:,.0f})", key=f"p2_td_{pct}"):
                        SS["mc_p2_trail_dd"] = bal * pct / 100
                SS["mc_p2_trail_dd"] = st.number_input("Trailing DD Limit ($)", 0.0, 1e6,
                    float(SS.get("mc_p2_trail_dd", 3_000)), 100.0, key="p2_tdd")

                st.markdown("**Daily Loss Limit**")
                SS["mc_p2_daily_loss"] = st.number_input("Daily Loss Limit ($)", 0.0, 1e6,
                    float(SS.get("mc_p2_daily_loss", 2_000)), 100.0, key="p2_dl")

            with p2b:
                st.markdown("**Payout Milestone Levels (when to withdraw)**")
                st.caption("Add balance levels at which a payout is triggered.")
                bal = SS.get("mc_p2_balance", 50_000)
                default_levels = [bal * 1.06, bal * 1.12, bal * 1.20, bal * 1.30]
                levels_str = st.text_area(
                    "Payout Trigger Balances (one per line)",
                    value="\n".join(f"{l:,.0f}" for l in
                                    SS.get("mc_p2_payout_levels", default_levels)),
                    height=120, key="p2_levels",
                )
                try:
                    payout_levels = [float(x.replace(",","").strip())
                                     for x in levels_str.splitlines() if x.strip()]
                except Exception:
                    payout_levels = default_levels
                SS["mc_p2_payout_levels"] = payout_levels

                SS["mc_p2_payout_amt"] = st.number_input(
                    "Withdrawal per payout event ($)", 0.0, 1e6,
                    float(SS.get("mc_p2_payout_amt", 500)), 50.0, key="p2_wa",
                )
                st.markdown("**Balance Target (goal)**")
                SS["mc_p2_target"] = st.number_input("Target balance ($)", 0.0, 1e7,
                    float(SS.get("mc_p2_target", bal * 2)), 500.0, key="p2_tgt")

            _card_close()

            _card_open("◈ SIMULATION PARAMETERS", _A)
            pp1, pp2, pp3 = st.columns(3)
            with pp1:
                SS["mc_p2_paths"] = int(st.number_input("Paths", 100, 20_000,
                    int(SS.get("mc_p2_paths", 1_000)), 100, key="p2_np"))
            with pp2:
                SS["mc_p2_horizon"] = int(st.number_input("Horizon (trading days)", 5, 2520,
                    int(SS.get("mc_p2_horizon", 252)), 5, key="p2_hor"))
            with pp3:
                animate_p2 = st.toggle("Animate paths", True, key="p2_anim")
            _card_close()

    # ─── TAB 4: Risk Metrics ──────────────────────────────────────
    with tab_risk:
        _card_open("◈ RISK METRIC DISPLAY", _G)
        SS["mc_selected_metrics"] = st.multiselect(
            "Metrics to show",
            ["Sharpe Ratio","Sortino Ratio","Calmar Ratio","SQN",
             "Kelly %","Max Drawdown","Win Rate","Profit Factor"],
            default=SS.get("mc_selected_metrics",["Sharpe Ratio","Max Drawdown","Win Rate"]),
            key="mc_rmet",
        )
        m = SS.get("metrics", {})
        if m:
            mk = {"Sharpe Ratio":("sharpe","blue"),"Sortino Ratio":("sortino","blue"),
                  "Calmar Ratio":("calmar","amber"),"SQN":("sqn","blue"),
                  "Kelly %":("kelly_half_pct","amber"),"Max Drawdown":("max_drawdown_dollars","red"),
                  "Win Rate":("win_rate_pct","green"),"Profit Factor":("profit_factor","blue")}
            rows = []
            for lbl in SS.get("mc_selected_metrics", []):
                if lbl in mk:
                    k, col = mk[lbl]
                    v = m.get(k, 0)
                    rows.append((lbl,
                                 f"${v:,.2f}" if "drawdown" in k else f"{v:.2f}%" if "pct" in k or "rate" in k else f"{v:.3f}",
                                 col))
            if rows: metric_grid(rows, "BACKTEST RISK METRICS")
        _card_close()

    # ─── TAB 5: Date Range ────────────────────────────────────────
    with tab_date:
        _card_open("◈ MC DATE WINDOW", _G)
        mc_date_mode = st.radio("Sample window", ["All trades","Custom range","Last N months"],
                                horizontal=True, key="mc_dm")
        mc_trades = trades.copy()
        if mc_date_mode == "Custom range" and "entry_time" in trades.columns:
            from datetime import date as dt_date
            cd1, cd2 = st.columns(2)
            with cd1: mc_s = st.date_input("From", dt_date(2023,1,1), key="mc_dr_s")
            with cd2: mc_e = st.date_input("To",   dt_date.today(),   key="mc_dr_e")
            mc_trades = trades[(pd.to_datetime(trades["entry_time"]).dt.date >= mc_s) &
                               (pd.to_datetime(trades["entry_time"]).dt.date <= mc_e)]
        elif mc_date_mode == "Last N months" and "entry_time" in trades.columns:
            n_mo = st.slider("Months", 1, 36, 12, 1, key="mc_nmo")
            cutoff = pd.Timestamp.now() - pd.DateOffset(months=n_mo)
            mc_trades = trades[pd.to_datetime(trades["entry_time"]) >= cutoff]
        st.markdown(f'<div style="font-family:monospace;font-size:0.76rem;color:#666;">'
                    f'Trades in window: <span style="color:{_G};">{len(mc_trades):,}</span></div>',
                    unsafe_allow_html=True)
        SS["mc_filtered_trades"] = mc_trades
        _card_close()

    # ══════════════════════════════════════════════════════════════
    # RUN CONTROLS
    # ══════════════════════════════════════════════════════════════
    st.markdown("---")
    rc1, rc2, rc3, rc4 = st.columns([1.5, 1, 1, 1])
    with rc1:
        animate = st.toggle("Animate walk-paths", True, key="mc_anim")
        anim_fps = st.select_slider("Speed", [1,2,5,10,20,50], 10, key="mc_fps",
                                    format_func=lambda v: f"{v}d/frame") if animate else 50
    with rc2:
        run_btn = st.button("▶  RUN / BACKTEST", type="primary",
                            use_container_width=True, key="mc_run")
    with rc3:
        p2_run  = st.button("🚀 RUN PHASE TWO", use_container_width=True, key="mc_p2_run",
                            disabled=not SS.get("mc_p2_on", False))
    with rc4:
        stop_btn = st.button("⏹  STOP", use_container_width=True, key="mc_stop")
        if stop_btn: SS["mc_running"] = False

    # ══════════════════════════════════════════════════════════════
    # MAIN MC SIMULATION
    # ══════════════════════════════════════════════════════════════
    if run_btn:
        mc_trades = SS.get("mc_filtered_trades", trades)
        if mc_trades is None or mc_trades.empty:
            banner_warning("No trades in selected window."); return

        mc_cfg = MonteCarloConfig(
            num_paths=int(SS.get("mc_num_paths",1_000)),
            horizon_days=int(SS.get("mc_horizon",252)),
            sampling_method=SS.get("mc_method",METHODS[0]),
            noise_pct=float(SS.get("mc_noise",0)),
            win_rate_haircut=float(SS.get("mc_wr_cut",0)),
            stop_size_pct=float(SS.get("mc_stop_inc",0)),
            size_reduction=float(SS.get("mc_size_red",0)),
            trade_removal=float(SS.get("mc_t_rem",0)),
            combine_mode=bool(SS.get("mc_eval_on",False)),
            daily_loss_limit=float(SS.get("mc_eval_daily_loss",3_000)),
            max_drawdown_limit=float(SS.get("mc_eval_max_dd",6_000)),
            profit_target=float(SS.get("mc_eval_profit_tgt",12_000)),
            min_trading_days=int(SS.get("mc_min_days",5)),
            max_trading_days=int(SS.get("mc_max_days",30)),
        )
        prog = st.progress(0, text="Running Monte Carlo…")
        t0   = time.perf_counter()
        mc_res = run_monte_carlo(mc_trades, mc_cfg)
        elapsed = time.perf_counter() - t0
        prog.progress(100, text=f"Done — {elapsed:.2f}s")
        SS["mc_result"] = mc_res; SS["mc_config"] = mc_cfg; SS["mc_running"] = True
        banner_success(f"MC complete — {mc_cfg.num_paths:,} paths | "
                       f"Median ${mc_res.median_final:+,.0f} | "
                       f"P(profit) {mc_res.prob_profit:.1f}%")

    mc_res = SS.get("mc_result")
    mc_cfg = SS.get("mc_config")

    if mc_res and len(mc_res.pct_50) > 0:
        combine_tgts = ({"profit_target": float(SS.get("mc_eval_profit_tgt",12_000)),
                         "max_drawdown_limit": float(SS.get("mc_eval_max_dd",6_000)),
                         "daily_loss_limit": float(SS.get("mc_eval_daily_loss",3_000))}
                        if SS.get("mc_eval_on") else None)
        horizon = len(mc_res.pct_50) - 1
        chart_ph = st.empty(); hist_ph = st.empty(); stats_ph = st.empty()

        if animate and SS.get("mc_running", False):
            for day in range(max(1, horizon//60), horizon+1, max(1, anim_fps)):
                if not SS.get("mc_running", True): break
                chart_ph.plotly_chart(_fan_frame(mc_res.paths, min(day,horizon), combine_tgts),
                                      use_container_width=True, key=f"mc_fan_{day}")
                time.sleep(0.04)
            SS["mc_running"] = False

        chart_ph.plotly_chart(_fan_frame(mc_res.paths, horizon, combine_tgts),
                              use_container_width=True, key="mc_fan_fin")
        hist_ph.plotly_chart(_histogram_frame(mc_res.final_equity, f"{horizon}d"),
                             use_container_width=True, key="mc_hist")

        with stats_ph.container():
            _card_open("◈ SIMULATION STATISTICS", _G)
            sc1, sc2 = st.columns(2)
            with sc1:
                metric_grid([
                    ("Median Final P&L",   f"${mc_res.median_final:+,.0f}",       "white"),
                    ("Mean Final P&L",     f"${mc_res.mean_final:+,.0f}",         "white"),
                    ("5th Pct Final",      f"${mc_res.p5_final:+,.0f}",           "red"),
                    ("95th Pct Final",     f"${mc_res.p95_final:+,.0f}",          "green"),
                    ("P(Profit)",          f"{mc_res.prob_profit:.2f}%",          "green"),
                    ("P(Ruin)",            f"{mc_res.prob_ruin:.2f}%",            "red"),
                    ("Median Max DD",      f"${mc_res.max_dd_median:,.0f}",       "amber"),
                    ("95th Pct Max DD",    f"${mc_res.max_dd_p95:,.0f}",          "red"),
                ])
            with sc2:
                extra = []
                if SS.get("mc_eval_on"):
                    ev = _evaluate_paths(mc_res.paths,
                                         float(SS.get("mc_eval_daily_loss",3_000)),
                                         float(SS.get("mc_eval_max_dd",6_000)),
                                         float(SS.get("mc_eval_profit_tgt",12_000)),
                                         int(SS.get("mc_min_days",5)),
                                         int(SS.get("mc_max_days",30)))
                    extra = [("─ EVALUATION ─","────────────","dim"),
                             ("PASS Rate",       f"{ev['pass_rate']:.2f}%",       "green"),
                             ("FAIL Max DD",     f"{ev['fail_dd_rate']:.2f}%",    "red"),
                             ("FAIL Daily Loss", f"{ev['fail_dl_rate']:.2f}%",    "red"),
                             ("FAIL Timeout",    f"{ev['fail_time_rate']:.2f}%",  "amber"),
                             ("Avg Pass Day",    f"Day {ev['avg_pass_day']:.1f}", "blue")]
                metric_grid([("Paths",   f"{len(mc_res.paths):,}",              "white"),
                             ("Horizon", f"{horizon}d",                          "white"),
                             ("Method",  mc_cfg.sampling_method if mc_cfg else "—","blue")] + extra)
            _card_close()

            from export import export_monte_carlo_csv
            st.download_button("📥 Download MC CSV", export_monte_carlo_csv(mc_res),
                               file_name=f"mc_{horizon}d.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════════════
    # PHASE TWO RUN
    # ══════════════════════════════════════════════════════════════
    if p2_run and SS.get("mc_p2_on"):
        mc_trades = SS.get("mc_filtered_trades", trades)
        if mc_trades is None or mc_trades.empty:
            banner_warning("No trades in window."); return

        pnl_arr = mc_trades["net_pnl"].values.astype(float)
        bal     = float(SS.get("mc_p2_balance", 50_000))

        with st.spinner("Running Phase Two simulation…"):
            t0  = time.perf_counter()
            p2r = _run_phase_two(
                trade_pnl=pnl_arr,
                start_balance=bal,
                trailing_dd_limit=float(SS.get("mc_p2_trail_dd", 3_000)),
                daily_loss_limit=float(SS.get("mc_p2_daily_loss", 2_000)),
                payout_levels=SS.get("mc_p2_payout_levels", [bal*1.06, bal*1.12]),
                payout_withdrawal=float(SS.get("mc_p2_payout_amt", 500)),
                target_balance=float(SS.get("mc_p2_target", bal*2)),
                n_paths=int(SS.get("mc_p2_paths", 1_000)),
                horizon=int(SS.get("mc_p2_horizon", 252)),
            )
            elapsed = time.perf_counter() - t0
        SS["mc_p2_result"] = p2r

        banner_success(
            f"Phase Two complete — {int(SS.get('mc_p2_paths',1_000)):,} paths | "
            f"Survival rate: **{p2r['survival_rate']:.1f}%** | "
            f"Target hit: **{p2r['target_rate']:.1f}%** | "
            f"Runtime: {elapsed:.2f}s"
        )

    p2r = SS.get("mc_p2_result")
    if p2r:
        st.markdown("---")
        _card_open("🚀 PHASE TWO — RUN-UP RESULTS", _B)

        bal    = float(SS.get("mc_p2_balance", 50_000))
        levels = SS.get("mc_p2_payout_levels", [bal*1.06, bal*1.12])
        tdd    = float(SS.get("mc_p2_trail_dd", 3_000))
        tgt    = float(SS.get("mc_p2_target", bal*2))

        st.plotly_chart(
            _phase_two_chart(p2r, bal, levels, tdd, tgt),
            use_container_width=True, key="p2_chart",
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            metric_grid([
                ("Account Survival Rate",   f"{p2r['survival_rate']:.2f}%",           "green"),
                ("Account Blown Rate",      f"{p2r['blown_rate']:.2f}%",              "red"),
                ("Target Balance Hit Rate", f"{p2r['target_rate']:.2f}%",             "blue"),
                ("Avg Days to Target",      f"{p2r['avg_days_target']:.1f}d",         "amber"),
            ])
        with pc2:
            metric_grid([
                ("Median Final P&L",        f"${p2r['median_final']:+,.0f}",          "white"),
                ("5th Pct Final",           f"${p2r['p5_final']:+,.0f}",              "red"),
                ("95th Pct Final",          f"${p2r['p95_final']:+,.0f}",             "green"),
                ("Avg Payouts Taken",       f"{p2r['avg_payouts']:.1f}",              "amber"),
                ("Avg Total Withdrawn",     f"${p2r['total_withdrawn']:,.0f}",         "green"),
            ])
        _card_close()


    # ══════════════════════════════════════════════════════════════
    # P&L DISTRIBUTION — shown after simulation completes
    # ══════════════════════════════════════════════════════════════
    mc_res = SS.get("mc_result")
    if mc_res is not None and len(mc_res.pct_50) > 0:
        trades_for_dist = SS.get("mc_filtered_trades", None)
        if trades_for_dist is None: trades_for_dist = SS.get("backtest_result", None)
        if hasattr(trades_for_dist, "trades"): trades_for_dist = trades_for_dist.trades
        equity_for_dist = None
        br = SS.get("backtest_result")
        if br and br.equity_curve is not None: equity_for_dist = br.equity_curve

        if trades_for_dist is not None and not trades_for_dist.empty:
            st.markdown("---")
            _card_open("◈ P&L DISTRIBUTION MODEL", _G)
            from pnl_distribution import build_pnl_distribution, build_return_stats_table
            pd1, pd2, pd3 = st.columns(3)
            with pd1: mc_dsplit = st.selectbox("Split by", ["signal","direction","regime","none"], key="mc_dsplit")
            with pd2: mc_ddaily = st.toggle("Daily returns view", equity_for_dist is not None, key="mc_ddaily")
            with pd3: mc_dbins  = st.slider("Bins", 20, 120, 60, 5, key="mc_dbins")

            fig_mcdist = build_pnl_distribution(
                trades_for_dist, equity_for_dist,
                split_by=mc_dsplit,
                show_daily=mc_ddaily and equity_for_dist is not None,
                n_bins=mc_dbins,
                title="P&L Distribution — Monte Carlo Sample",
            )
            st.plotly_chart(fig_mcdist, use_container_width=True, key="mc_distfig")

            dist_stats = build_return_stats_table(trades_for_dist, equity_for_dist)
            if not dist_stats.empty:
                with st.expander("📊 Distribution Statistics"):
                    st.dataframe(dist_stats, use_container_width=True, hide_index=True)
            _card_close()
