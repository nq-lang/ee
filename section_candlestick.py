"""
section_candlestick.py  v5
══════════════════════════
Section 6 — Candlestick Chart + Backtest Replay Terminal

REPLAY ANIMATION: Uses Plotly frames for smooth client-side animation.
All frames are built once server-side, then the animation runs
entirely in the browser — no server roundtrips, no stuttering.
Accurate OHLCV + trade markers appear progressively bar by bar.

EVALUATION  — reach net profit target without breaching max drawdown
PHASE TWO   — build balance without going negative (EOD + intraday DD tracking)
"""
from __future__ import annotations
import time
from datetime import date as dt_date
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import COLORS, PLOTLY_DARK, INSTRUMENTS
from ui_components import (section_header, banner_info,
                            banner_warning, banner_success, metric_grid)
from pnl_distribution import build_pnl_distribution, build_return_stats_table

_G=COLORS["green"]; _R=COLORS["red"]; _A=COLORS["amber"]

# TradingView-style chart config — scroll to zoom, drag axes to stretch/scroll
_CHART_CONFIG = {
    "scrollZoom":         True,   # mousewheel / pinch-to-zoom
    "displayModeBar":     True,
    "modeBarButtonsToAdd":["pan2d","zoom2d","resetScale2d"],
    "modeBarButtonsToRemove": ["lasso2d","select2d"],
    "doubleClick":        "reset+autosize",
    "showTips":           True,
    "responsive":         True,
}

_B=COLORS["blue"];  _W=COLORS["text"]; _C=COLORS["card_bg"]

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTH PLOTLY-FRAMES ANIMATION  (client-side, no server roundtrips)
# ─────────────────────────────────────────────────────────────────────────────

def build_animated_replay(
    ohlcv:      pd.DataFrame,
    trades:     pd.DataFrame,
    max_frames: int  = 300,
    show_vol:   bool = True,
    speed_ms:   int  = 80,
) -> go.Figure:
    """
    Build a single Plotly Figure with animation frames.

    The animation runs entirely in the browser after the figure is sent —
    smooth, accurate, and zero server roundtrips during playback.

    Parameters
    ----------
    ohlcv      : OHLCV DataFrame
    trades     : trade log (entry_time, exit_time, entry_price, stop_price,
                 target_price, direction, signal, net_pnl, outcome)
    max_frames : cap on number of animation frames (300 ≈ smooth, ≤10s build)
    show_vol   : include volume sub-panel
    speed_ms   : milliseconds between frames (lower = faster replay)
    """
    n = len(ohlcv)
    if n == 0:
        return go.Figure()

    # Build frame indices: evenly spaced across the full dataset
    min_start = min(50, n // 10)
    frame_idx = np.linspace(min_start, n, min(max_frames, n - min_start),
                            dtype=int)
    frame_idx = np.unique(np.clip(frame_idx, min_start, n))

    rows    = 2 if show_vol else 1
    heights = [0.78, 0.22] if show_vol else [1.0]

    # ── Precompute trade visibility per frame ─────────────────────────────────
    # For each frame index, determine which trades have entered
    trade_entry_bar: list[int] = []
    if trades is not None and not trades.empty and "entry_time" in trades.columns:
        for _, t in trades.iterrows():
            et = pd.Timestamp(t["entry_time"])
            # Find bar index where this trade entered
            try:
                bar_pos = ohlcv.index.searchsorted(et)
                trade_entry_bar.append(int(bar_pos))
            except Exception:
                trade_entry_bar.append(n + 1)
    else:
        trade_entry_bar = []

    # ── Build base traces (empty — filled per frame) ──────────────────────────
    ts = ohlcv.index
    op = ohlcv["open"].values
    hi = ohlcv["high"].values
    lo = ohlcv["low"].values
    cl = ohlcv["close"].values
    vol= ohlcv["volume"].values if "volume" in ohlcv.columns else np.zeros(n)

    def _make_candle_trace(end_i):
        return go.Candlestick(
            x=ts[:end_i], open=op[:end_i], high=hi[:end_i],
            low=lo[:end_i], close=cl[:end_i],
            name="Price",
            increasing=dict(line=dict(color=_G, width=1), fillcolor="#00AA44"),
            decreasing=dict(line=dict(color=_R, width=1), fillcolor="#CC2222"),
            hoverinfo="x+y", xaxis="x", yaxis="y",
        )

    def _make_vol_trace(end_i):
        vc = [_G if cl[j] >= op[j] else _R for j in range(end_i)]
        return go.Bar(
            x=ts[:end_i], y=vol[:end_i],
            marker_color=vc, opacity=0.45, name="Volume",
            xaxis="x", yaxis="y2" if show_vol else "y",
        )

    def _make_trade_traces(end_i):
        """Return entry/SL/TP traces for trades active at bar end_i."""
        traces_out = []
        if not trade_entry_bar or trades is None or trades.empty:
            return traces_out
        visible_trades = [
            (bi, ti) for ti, bi in enumerate(trade_entry_bar)
            if bi <= end_i
        ]
        for bi, ti in visible_trades[-50:]:  # cap at 50 visible trades
            t = trades.iloc[ti]
            is_long  = str(t.get("direction","LONG")).upper() == "LONG"
            ep       = float(t.get("entry_price", 0))
            sp       = float(t.get("stop_price", 0))
            tp       = float(t.get("target_price", 0))
            xp       = float(t.get("exit_price", 0))
            et_ts    = t.get("entry_time")
            xt_ts    = t.get("exit_time")
            outcome  = str(t.get("outcome","?"))
            pnl      = float(t.get("net_pnl", 0))
            sig      = str(t.get("signal",""))
            if pd.isna(et_ts) or ep == 0:
                continue
            col  = _G if is_long else _R
            sym  = "triangle-up" if is_long else "triangle-down"
            xt2  = xt_ts if (xt_ts and not pd.isna(xt_ts) and
                              pd.Timestamp(xt_ts) <= ts[min(end_i, n-1)]) else et_ts
            # Entry arrow + label
            traces_out.append(go.Scatter(
                x=[et_ts], y=[ep], mode="markers+text",
                marker=dict(symbol=sym, size=13, color=col,
                            line=dict(color="#FFF", width=1.2)),
                text=[f"{'▲' if is_long else '▼'} {sig}"],
                textposition="top center" if is_long else "bottom center",
                textfont=dict(color=col, size=13),
                showlegend=False, hoverinfo="skip",
                xaxis="x", yaxis="y",
            ))
            # SL line + price label
            if sp > 0:
                traces_out.append(go.Scatter(
                    x=[et_ts, xt2], y=[sp, sp], mode="lines+text",
                    line=dict(color=_R, width=1.2, dash="dash"),
                    text=["", f"SL {sp:,.2f}"],
                    textposition="middle right",
                    textfont=dict(color=_R, size=12),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x", yaxis="y",
                ))
            # TP line + price label
            if tp > 0:
                traces_out.append(go.Scatter(
                    x=[et_ts, xt2], y=[tp, tp], mode="lines+text",
                    line=dict(color=_G, width=1.2, dash="dot"),
                    text=["", f"TP {tp:,.2f}"],
                    textposition="middle right",
                    textfont=dict(color=_G, size=12),
                    showlegend=False, hoverinfo="skip",
                    xaxis="x", yaxis="y",
                ))
            # Exit marker (only if trade has closed within visible window)
            if xp > 0 and xt_ts and not pd.isna(xt_ts):
                if pd.Timestamp(xt_ts) <= ts[min(end_i, n-1)]:
                    ec = _G if outcome=="WIN" else _R if outcome=="LOSS" else _A
                    icon = {"WIN":"✓","LOSS":"✗","TIME":"⏱","EOD":"⊗","GAP":"◈"}.get(outcome,"•")
                    traces_out.append(go.Scatter(
                        x=[xt_ts], y=[xp], mode="markers+text",
                        marker=dict(symbol="square", size=9, color=ec,
                                    line=dict(color=_W, width=0.8)),
                        text=[f"{icon} ${pnl:+,.0f}"],
                        textposition="top center",
                        textfont=dict(color=ec, size=12),
                        showlegend=False, hoverinfo="skip",
                        xaxis="x", yaxis="y",
                    ))
        return traces_out

    # ── Build frame 0 (initial state) ─────────────────────────────────────────
    f0_end   = int(frame_idx[0])
    base_data= [_make_candle_trace(f0_end)]
    if show_vol: base_data.append(_make_vol_trace(f0_end))
    base_data += _make_trade_traces(f0_end)

    # ── Build all animation frames ────────────────────────────────────────────
    frames = []
    cumulative_pnls = []

    for fi, end_i in enumerate(frame_idx):
        end_i = int(end_i)
        # Running P&L at this frame
        if trades is not None and not trades.empty and "net_pnl" in trades.columns:
            visible_pnl = sum(
                float(trades.iloc[ti]["net_pnl"])
                for ti, bi in enumerate(trade_entry_bar)
                if bi <= end_i and (
                    trades.iloc[ti].get("exit_time") and
                    not pd.isna(trades.iloc[ti]["exit_time"]) and
                    pd.Timestamp(trades.iloc[ti]["exit_time"]) <= ts[min(end_i, n-1)]
                )
            )
        else:
            visible_pnl = 0.0
        cumulative_pnls.append(visible_pnl)

        frame_data = [_make_candle_trace(end_i)]
        if show_vol:
            frame_data.append(_make_vol_trace(end_i))
        frame_data += _make_trade_traces(end_i)

        bar_date = str(ts[min(end_i-1, n-1)])[:16] if end_i > 0 else ""
        frames.append(go.Frame(
            data=frame_data,
            name=str(fi),
            layout=go.Layout(
                title_text=(
                    f"Replay  Bar {end_i:,}/{n:,}  |  "
                    f"{bar_date}  |  "
                    f"Price {cl[min(end_i-1,n-1)]:,.2f}  |  "
                    f"P&L ${visible_pnl:+,.0f}"
                )
            ),
        ))

    # ── Assemble figure ───────────────────────────────────────────────────────
    specs = [[{"secondary_y": False}]] * rows
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
        specs=specs,
    )
    for trace in base_data:
        row = 2 if (show_vol and hasattr(trace,'yaxis') and
                    getattr(trace,'yaxis','y') == 'y2') else 1
        # All plotly traces go to row 1 except volume
        if isinstance(trace, go.Bar) and show_vol:
            fig.add_trace(trace, row=2, col=1)
        else:
            fig.add_trace(trace, row=1, col=1)

    fig.frames = frames

    # ── Slider steps ─────────────────────────────────────────────────────────
    steps = []
    for fi, end_i in enumerate(frame_idx):
        step = dict(
            args=[[str(fi)], {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate", "transition": {"duration": 0}}],
            label=str(ts[min(int(end_i)-1, n-1)])[:10],
            method="animate",
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Date: ", "font": {"color": _W, "size": 10}},
        pad={"b": 10, "t": 10},
        len=0.92, x=0.04,
        steps=steps,
        bgcolor=_C,
        bordercolor=_A,
        font=dict(color=_W, size=9),
    )]

    # ── Play / Pause buttons ──────────────────────────────────────────────────
    updatemenus = [dict(
        type="buttons",
        showactive=False,
        x=0.04, y=1.08, xanchor="left",
        bgcolor=_C, bordercolor=_G,
        font=dict(color=_W, size=11),
        buttons=[
            dict(
                label="▶  PLAY",
                method="animate",
                args=[None, {
                    "frame": {"duration": speed_ms, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                    "mode": "immediate",
                }],
            ),
            dict(
                label="⏸  PAUSE",
                method="animate",
                args=[[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
            ),
        ],
    )]

    fig.update_layout(
        height=620 if show_vol else 520,
        title=dict(
            text=f"Backtest Replay  |  {n:,} bars  |  {len(frames)} frames",
            font=dict(color=_W, size=12),
        ),
        xaxis_rangeslider_visible=False,
        updatemenus=updatemenus,
        sliders=sliders,
        showlegend=False,
        dragmode="pan",
        margin=dict(l=55, r=90, t=60, b=80),
        **PLOTLY_DARK,
    )
    fig.update_xaxes(fixedrange=False, row=1, col=1)
    fig.update_yaxes(fixedrange=False, tickformat=",", row=1, col=1)
    if show_vol:
        fig.update_xaxes(fixedrange=False, row=2, col=1)
        fig.update_yaxes(fixedrange=False, row=2, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Static chart (instant view — no animation)
# ─────────────────────────────────────────────────────────────────────────────

def _static_candle_fig(ohlcv, trades=None, show_vol=True, show_vwap=False):
    rows    = 2 if show_vol else 1
    heights = [0.78, 0.22] if show_vol else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=heights, vertical_spacing=0.03)
    ts=ohlcv.index; op=ohlcv["open"].values; hi=ohlcv["high"].values
    lo=ohlcv["low"].values; cl=ohlcv["close"].values
    fig.add_trace(go.Candlestick(
        x=ts, open=op, high=hi, low=lo, close=cl, name="Price",
        increasing=dict(line=dict(color=_G, width=1), fillcolor="#00AA44"),
        decreasing=dict(line=dict(color=_R, width=1), fillcolor="#CC2222"),
        hoverinfo="x+y",
    ), row=1, col=1)
    if show_vwap and "vwap" in ohlcv.columns:
        fig.add_trace(go.Scatter(x=ts, y=ohlcv["vwap"], mode="lines",
            name="VWAP", line=dict(color=_A, width=1, dash="dot")), row=1, col=1)
    if show_vol and "volume" in ohlcv.columns:
        vc=[_G if cl[i]>=op[i] else _R for i in range(len(cl))]
        fig.add_trace(go.Bar(x=ts, y=ohlcv["volume"], marker_color=vc,
                             opacity=0.45, name="Volume"), row=2, col=1)
    if trades is not None and not trades.empty:
        for tr in _static_trade_traces(trades):
            fig.add_trace(tr, row=1, col=1)
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        margin=dict(l=55, r=90, t=30, b=30),
        dragmode="pan",
        **PLOTLY_DARK,
    )
    # Y-axis: fixedrange=False lets users drag the price ladder to stretch/shrink
    # X-axis: fixedrange=False lets users drag the date axis to scroll/zoom timeline
    fig.update_xaxes(fixedrange=False, row=1, col=1)
    fig.update_yaxes(fixedrange=False, tickformat=",", row=1, col=1)
    if show_vol:
        fig.update_xaxes(fixedrange=False, row=2, col=1)
        fig.update_yaxes(fixedrange=False, row=2, col=1)
    return fig


def _static_trade_traces(trades):
    traces=[]
    for _,t in trades.iterrows():
        il=str(t.get("direction","LONG")).upper()=="LONG"
        ep=float(t.get("entry_price",0)); sp=float(t.get("stop_price",0))
        tp=float(t.get("target_price",0)); xp=float(t.get("exit_price",0))
        oc=str(t.get("outcome","?")); et=t.get("entry_time"); xt=t.get("exit_time")
        sig=str(t.get("signal","")); pnl=float(t.get("net_pnl",0))
        if pd.isna(et) or ep==0: continue
        ce=_G if il else _R; sym="triangle-up" if il else "triangle-down"
        xt2=xt if (xt and not pd.isna(xt)) else et
        traces.append(go.Scatter(x=[et],y=[ep],mode="markers+text",
            marker=dict(symbol=sym,size=13,color=ce,line=dict(color="#FFF",width=1.2)),
            text=[f"{'▲' if il else '▼'} {sig}"],
            textposition="top center" if il else "bottom center",
            textfont=dict(color=ce,size=13,family="monospace"),showlegend=False,hoverinfo="skip"))
        if sp>0:
            traces.append(go.Scatter(x=[et,xt2],y=[sp,sp],mode="lines+text",
                line=dict(color=_R,width=1.2,dash="dash"),
                text=["",f"SL {sp:,.2f}"],textposition="middle right",
                textfont=dict(color=_R,size=12),showlegend=False,hoverinfo="skip"))
        if tp>0:
            traces.append(go.Scatter(x=[et,xt2],y=[tp,tp],mode="lines+text",
                line=dict(color=_G,width=1.2,dash="dot"),
                text=["",f"TP {tp:,.2f}"],textposition="middle right",
                textfont=dict(color=_G,size=12),showlegend=False,hoverinfo="skip"))
        if xp>0 and xt and not pd.isna(xt):
            ec=_G if oc=="WIN" else _R if oc=="LOSS" else _A
            icon={"WIN":"✓","LOSS":"✗","TIME":"⏱","EOD":"⊗","GAP":"◈"}.get(oc,"•")
            traces.append(go.Scatter(x=[xt],y=[xp],mode="markers+text",
                marker=dict(symbol="square",size=9,color=ec,line=dict(color=_W,width=0.8)),
                text=[f"{icon} ${pnl:+,.0f}"],textposition="top center",
                textfont=dict(color=ec,size=12),showlegend=False,hoverinfo="skip"))
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Live tracker
# ─────────────────────────────────────────────────────────────────────────────
def _tracker_fig(equity_df):
    if equity_df is None or equity_df.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.6,0.4],
        vertical_spacing=0.04,subplot_titles=["Session P&L","Drawdown from Peak"])
    eq=equity_df["equity"].values; ts=equity_df.index
    pnl=eq-eq[0]; pk=np.maximum.accumulate(eq); dd=pk-eq
    col=_G if pnl[-1]>=0 else _R
    fig.add_trace(go.Scatter(x=ts,y=pnl,mode="lines",line=dict(color=col,width=2),
        fill="tozeroy",fillcolor=f"rgba(0,255,136,0.07)" if pnl[-1]>=0 else "rgba(255,59,59,0.07)"),row=1,col=1)
    fig.add_hline(y=0,line=dict(color="#333",width=1),row=1,col=1)
    fig.add_trace(go.Scatter(x=ts,y=-dd,mode="lines",line=dict(color=_R,width=1.5),
        fill="tozeroy",fillcolor="rgba(255,59,59,0.09)"),row=2,col=1)
    last_pnl=float(pnl[-1]); max_dd=float(dd.max())
    fig.add_annotation(x=0.01,y=0.97,xref="paper",yref="paper",
        text=f"P&L: ${last_pnl:+,.0f}  |  Max DD: -${max_dd:,.0f}",
        showarrow=False,font=dict(color=col,size=11,family="monospace"),
        bgcolor=_C,bordercolor=col,borderwidth=1)
    fig.update_layout(height=280,showlegend=False,margin=dict(l=55,r=15,t=30,b=25),**PLOTLY_DARK)
    fig.update_yaxes(tickformat="$,.0f",row=1,col=1)
    fig.update_yaxes(tickformat="$,.0f",row=2,col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION simulation
# ─────────────────────────────────────────────────────────────────────────────
def _run_evaluation(trade_pnl,start_bal,profit_target,max_dd_limit,
                    daily_loss_limit,min_days,max_days,n_paths,seed=42):
    rng=np.random.default_rng(seed)
    tpd=max(1.0,len(trade_pnl)/max(max_days//2,1))
    goal=start_bal+profit_target
    paths=np.full((n_paths,max_days+1),start_bal,dtype=float)
    res={"pass":0,"fail_dd":0,"fail_dl":0,"fail_time":0}
    pass_days=[]
    for i in range(n_paths):
        bal=start_bal; peak=start_bal; failed=won=False
        n_t=int(round(max_days*tpd))
        samp=rng.choice(trade_pnl,size=n_t,replace=True)
        daily=np.array_split(samp,max_days)
        for day,blk in enumerate(daily,1):
            if failed or won: break
            dp=float(blk.sum())
            if dp<-daily_loss_limit: res["fail_dl"]+=1; failed=True; paths[i,day:]=bal; break
            bal+=dp; peak=max(peak,bal); paths[i,day]=bal
            if (peak-bal)>=max_dd_limit: res["fail_dd"]+=1; failed=True; paths[i,day:]=bal; break
            if bal>=goal and day>=min_days: res["pass"]+=1; won=True; pass_days.append(day); break
        if not failed and not won: res["fail_time"]+=1
    n=n_paths
    return {"paths":paths,"pass_rate":res["pass"]/n*100,"fail_dd_rate":res["fail_dd"]/n*100,
            "fail_dl_rate":res["fail_dl"]/n*100,"fail_time_rate":res["fail_time"]/n*100,
            "avg_pass_day":float(np.mean(pass_days)) if pass_days else 0.0,
            "start_bal":start_bal,"goal":goal,"max_dd_limit":max_dd_limit,"n_paths":n}

def _eval_fig(r):
    paths=r["paths"]; n=len(paths); days=np.arange(paths.shape[1])
    p5=np.percentile(paths,5,axis=0); p25=np.percentile(paths,25,axis=0)
    p50=np.percentile(paths,50,axis=0); p75=np.percentile(paths,75,axis=0)
    p95=np.percentile(paths,95,axis=0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([days,days[::-1]]),y=np.concatenate([p95,p5[::-1]]),
        fill="toself",fillcolor="rgba(0,200,80,0.07)",line=dict(color="rgba(0,0,0,0)"),
        name="5–95th",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([days,days[::-1]]),y=np.concatenate([p75,p25[::-1]]),
        fill="toself",fillcolor="rgba(0,255,136,0.15)",line=dict(color="rgba(0,0,0,0)"),
        name="25–75th",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days,y=p50,mode="lines",name="Median",line=dict(color=_W,width=2.5)))
    rng2=np.random.default_rng(0)
    for i in rng2.choice(n,min(40,n),replace=False):
        fig.add_trace(go.Scatter(x=days,y=paths[i],mode="lines",
            line=dict(color="rgba(0,255,136,0.05)",width=0.7),showlegend=False,hoverinfo="skip"))
    fig.add_hline(y=r["goal"],line=dict(color=_G,dash="solid",width=2),
        annotation_text=f"✅ PROFIT TARGET  ${r['goal']:,.0f}",annotation_font_color=_G)
    floor=r["start_bal"]-r["max_dd_limit"]
    fig.add_hline(y=floor,line=dict(color=_R,dash="dash",width=2),
        annotation_text=f"❌ MAX DD FLOOR  ${floor:,.0f}",annotation_font_color=_R)
    fig.add_hline(y=r["start_bal"],line=dict(color="#444",dash="dot",width=1))
    fig.update_layout(height=440,
        title=dict(text=f"EVALUATION — PASS {r['pass_rate']:.1f}%  |  Avg pass day {r['avg_pass_day']:.1f}",
                   font=dict(color=_G,size=12)),
        xaxis_title="Trading Day",yaxis=dict(tickformat="$,.0f",gridcolor=COLORS["grid"]),
        margin=dict(l=65,r=20,t=55,b=40),showlegend=True,
        legend=dict(orientation="h",y=1.02,x=0),**PLOTLY_DARK)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PHASE TWO simulation
# ─────────────────────────────────────────────────────────────────────────────
def _run_phase_two(trade_pnl,start_bal,hard_floor,daily_loss_limit,payout_levels,
                   payout_amt,target_bal,n_paths,horizon,use_eod,use_intraday,
                   intraday_dd_limit,seed=42):
    rng=np.random.default_rng(seed)
    tpd=max(1.0,len(trade_pnl)/max(horizon//5,1))
    paths=np.full((n_paths,horizon+1),start_bal,dtype=float)
    blown=np.zeros(n_paths,dtype=bool)
    eod_neg=np.zeros(n_paths,dtype=int)
    payout_counts=np.zeros(n_paths,dtype=int)
    reached_target=np.zeros(n_paths,dtype=bool)
    days_to_target=np.full(n_paths,horizon+1)
    for i in range(n_paths):
        bal=start_bal; peak=start_bal; total=0.0; hits=set()
        samp=rng.choice(trade_pnl,size=int(round(horizon*tpd)),replace=True)
        daily=np.array_split(samp,horizon)
        for day,blk in enumerate(daily,1):
            if blown[i]: break
            if use_intraday and len(blk)>1:
                ip=0.0; halted=False
                for v in blk:
                    ip2=ip+v
                    if ip-(ip+v)>=intraday_dd_limit: halted=True; break
                    ip=ip2
                dp=0.0 if halted else float(blk.sum())
            else:
                dp=float(blk.sum())
            dp=max(dp,-daily_loss_limit)
            bal+=dp; total+=dp; paths[i,day]=bal
            if use_eod and bal<=hard_floor:
                eod_neg[i]+=1; blown[i]=True; paths[i,day:]=bal; break
            if bal<=hard_floor:
                blown[i]=True; paths[i,day:]=bal; break
            peak=max(peak,bal)
            for lvl in payout_levels:
                if lvl not in hits and bal>=lvl:
                    bal-=payout_amt; total-=payout_amt; paths[i,day]=bal
                    payout_counts[i]+=1; hits.add(lvl)
            if not reached_target[i] and bal>=target_bal:
                reached_target[i]=True; days_to_target[i]=day
    final=paths[:,-1]
    return {"paths":paths,"final":final,"blown_rate":blown.mean()*100,
            "survival_rate":(~blown).mean()*100,"target_rate":reached_target.mean()*100,
            "avg_payouts":float(payout_counts.mean()),"total_withdrawn":float(payout_counts.mean()*payout_amt),
            "median_final":float(np.median(final)),"mean_final":float(final.mean()),
            "p5_final":float(np.percentile(final,5)),"p95_final":float(np.percentile(final,95)),
            "avg_days_target":float(np.mean(days_to_target[reached_target]) if reached_target.any() else 0),
            "avg_eod_neg_days":float(eod_neg.mean()),"start_bal":start_bal,
            "hard_floor":hard_floor,"target_bal":target_bal,"n_paths":n_paths}

def _p2_fig(r,payout_levels):
    paths=r["paths"]; n=len(paths); days=np.arange(paths.shape[1])
    p5=np.percentile(paths,5,axis=0); p25=np.percentile(paths,25,axis=0)
    p50=np.percentile(paths,50,axis=0); p75=np.percentile(paths,75,axis=0)
    p95=np.percentile(paths,95,axis=0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([days,days[::-1]]),y=np.concatenate([p95,p5[::-1]]),
        fill="toself",fillcolor="rgba(0,191,255,0.07)",line=dict(color="rgba(0,0,0,0)"),name="5–95th",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([days,days[::-1]]),y=np.concatenate([p75,p25[::-1]]),
        fill="toself",fillcolor="rgba(0,191,255,0.15)",line=dict(color="rgba(0,0,0,0)"),name="25–75th",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=days,y=p50,mode="lines",name="Median Balance",line=dict(color=_B,width=2.5)))
    rng2=np.random.default_rng(1)
    for i in rng2.choice(n,min(35,n),replace=False):
        fig.add_trace(go.Scatter(x=days,y=paths[i],mode="lines",
            line=dict(color="rgba(0,191,255,0.05)",width=0.7),showlegend=False,hoverinfo="skip"))
    fig.add_hline(y=r["hard_floor"],line=dict(color=_R,dash="dash",width=2),
        annotation_text=f"💀 FLOOR  ${r['hard_floor']:,.0f}",annotation_font_color=_R)
    fig.add_hline(y=r["target_bal"],line=dict(color=_G,dash="dot",width=1.5),
        annotation_text=f"🎯 TARGET  ${r['target_bal']:,.0f}",annotation_font_color=_G)
    fig.add_hline(y=r["start_bal"],line=dict(color="#444",dash="dot",width=1))
    for lvl in payout_levels:
        fig.add_hline(y=lvl,line=dict(color=_A,dash="dot",width=1),
            annotation_text=f"💰 ${lvl:,.0f}",annotation_font_color=_A)
    fig.update_layout(height=440,
        title=dict(text=f"PHASE TWO — Survival {r['survival_rate']:.1f}%  |  Target Hit {r['target_rate']:.1f}%",
                   font=dict(color=_B,size=12)),
        xaxis_title="Trading Day",yaxis=dict(tickformat="$,.0f",gridcolor=COLORS["grid"]),
        margin=dict(l=65,r=20,t=55,b=40),showlegend=True,
        legend=dict(orientation="h",y=1.02,x=0),**PLOTLY_DARK)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Config widgets (collapsed panel mirroring Section 1)
# ─────────────────────────────────────────────────────────────────────────────
def _card(title,color=_G):
    st.markdown(f'<div style="background:{_C};border:1px solid #2A2A2A;border-radius:8px;'
                f'padding:14px 18px;margin-bottom:8px;"><div style="font-family:monospace;'
                f'font-size:0.78rem;color:{color};margin-bottom:10px;">{title}</div>',
                unsafe_allow_html=True)
def _card_end(): st.markdown("</div>",unsafe_allow_html=True)

def _cfg_account(SS):
    _card("💰 ACCOUNT & CAPITAL",_G)
    bc=st.columns(6)
    for i,amt in enumerate([10_000,25_000,50_000,100_000,150_000,200_000]):
        with bc[i]:
            if st.button(f"${amt//1000}k",key=f"cs_b{amt}"): SS["starting_balance"]=float(amt)
    SS["starting_balance"]=st.number_input("Balance ($)",100.0,1e7,float(SS.get("starting_balance",50_000)),1_000.0,key="cs_bci")
    cc=st.columns(2)
    with cc[0]:
        nc=st.columns(5)
        for i,n in enumerate([1,2,5,10,20]):
            with nc[i]:
                if st.button(str(n),key=f"cs_cn{n}"): SS["num_contracts"]=n
        SS["num_contracts"]=st.number_input("Contracts",1,500,int(SS.get("num_contracts",1)),key="cs_cni")
    with cc[1]:
        SS["sizing_mode"]=st.radio("Sizing",["Fixed","Dynamic / Volatility-Scaled"],
            index=["Fixed","Dynamic / Volatility-Scaled"].index(
                SS.get("sizing_mode","Fixed") if SS.get("sizing_mode","Fixed") in
                ["Fixed","Dynamic / Volatility-Scaled"] else "Fixed"),key="cs_szm")
        if SS["sizing_mode"]!="Fixed":
            SS["risk_per_trade_pct"]=st.number_input("Risk%",0.1,10.0,float(SS.get("risk_per_trade_pct",1.0)),0.1,key="cs_rpt")
    _card_end()

def _cfg_fees(SS):
    _card("💸 COMMISSION & FEES",_A)
    c1,c2,c3=st.columns(3)
    with c1:
        for v in [0.0,0.25,0.50,1.00]:
            if st.button(f"${v:.2f} comm",key=f"cs_cf{v}"): SS["commission_per_side"]=v
        SS["commission_per_side"]=st.number_input("Comm/side",0.0,50.0,float(SS.get("commission_per_side",0.50)),0.05,key="cs_cfi")
    with c2:
        for v in [0.0,0.85,1.18]:
            if st.button(f"${v:.2f} exch",key=f"cs_ef{v}"): SS["exchange_fee"]=v
        SS["exchange_fee"]=st.number_input("Exchange fee",0.0,20.0,float(SS.get("exchange_fee",0.85)),0.01,key="cs_efi")
    with c3:
        SS["nfa_fee"]=st.number_input("NFA fee",0.0,5.0,float(SS.get("nfa_fee",0.02)),0.01,key="cs_nfa")
        SS["slippage_ticks"]=st.number_input("Slippage (ticks)",0,20,int(SS.get("slippage_ticks",1)),key="cs_slip")
    spec=INSTRUMENTS.get(SS.get("instrument","NQ"),INSTRUMENTS["NQ"])
    rt=(SS.get("commission_per_side",0.5)+SS.get("exchange_fee",0.85)+SS.get("nfa_fee",0.02))*2
    sl=SS.get("slippage_ticks",1)*spec["tick_value"]*2
    st.markdown(f'<div style="font-family:monospace;font-size:0.73rem;color:{_B};">'
                f"RT cost: ${rt:.4f} + ${sl:.4f} slip = <b>${rt+sl:.4f}</b></div>",unsafe_allow_html=True)
    _card_end()

def _cfg_stop_target(SS):
    _card("🎯 PROFIT TARGET & STOP LOSS",_G)
    c1,c2=st.columns(2)
    with c1:
        SS["profit_target_mode"]=st.radio("Target",["Strategy","Points","Dollars","R-Multiple"],
            index=["Strategy","Points","Dollars","R-Multiple"].index(
                SS.get("profit_target_mode","Strategy") if SS.get("profit_target_mode","Strategy")
                in ["Strategy","Points","Dollars","R-Multiple"] else "Strategy"),
            horizontal=True,key="cs_ptm")
        for v in [500,1_000,2_000,3_000]:
            if st.button(f"${v:,}tp",key=f"cs_tp{v}"): SS["profit_target_value"]=float(v)
        SS["profit_target_value"]=st.number_input("Target $",0.0,1e6,float(SS.get("profit_target_value",1_000)),50.0,key="cs_tpv")
    with c2:
        SS["stop_loss_mode"]=st.radio("Stop",["Strategy","Points","Dollars","R-Multiple"],
            index=["Strategy","Points","Dollars","R-Multiple"].index(
                SS.get("stop_loss_mode","Strategy") if SS.get("stop_loss_mode","Strategy")
                in ["Strategy","Points","Dollars","R-Multiple"] else "Strategy"),
            horizontal=True,key="cs_slm")
        for v in [100,250,500,1_000]:
            if st.button(f"${v:,}sl",key=f"cs_sl{v}"): SS["stop_loss_value"]=float(v)
        SS["stop_loss_value"]=st.number_input("Stop $",0.0,1e6,float(SS.get("stop_loss_value",500)),50.0,key="cs_slv")
    if SS.get("stop_loss_value",0)>0:
        rr=SS.get("profit_target_value",1000)/SS.get("stop_loss_value",500)
        st.markdown(f'<div style="font-family:monospace;color:{_A};">RR: 1:{rr:.2f}</div>',unsafe_allow_html=True)
    _card_end()

def _cfg_sizing(SS):
    _card("📐 SIZING & DISPLAY SCALING",_B)
    lbls=["1 MNQ","2 MNQ","5 MNQ","10 MNQ","1 NQ","2 NQ","5 NQ"]
    vals=[1,2,5,10,20,40,100]
    sc=st.columns(len(lbls))
    for i,(l,v) in enumerate(zip(lbls,vals)):
        with sc[i]:
            if st.button(l,key=f"cs_sc{l}"): SS["display_scale"]=v
    SS["display_scale"]=st.number_input("Custom ×",1,1000,int(SS.get("display_scale",1)),key="cs_sci")
    _card_end()

def _cfg_period(SS):
    _card("📅 TESTING PERIOD",_A)
    modes=["Full CSV History","Custom Date Range","Last N Trading Days","Last N Calendar Years"]
    cur=SS.get("date_range_mode","Full CSV History")
    if cur not in modes: cur="Full CSV History"
    SS["date_range_mode"]=st.radio("Range",modes,index=modes.index(cur),horizontal=True,key="cs_drm")
    if SS["date_range_mode"]=="Custom Date Range":
        d1,d2=st.columns(2)
        with d1: SS["custom_start"]=st.date_input("Start",value=dt_date(2020,1,1),key="cs_ds")
        with d2: SS["custom_end"]=st.date_input("End",value=dt_date.today(),key="cs_de")
    elif SS["date_range_mode"]=="Last N Trading Days":
        SS["last_n_days"]=st.number_input("Days",1,5000,int(SS.get("last_n_days",252)),key="cs_nd")
    elif SS["date_range_mode"]=="Last N Calendar Years":
        yc=st.columns(5)
        for i,yr in enumerate([1,2,3,5,10]):
            with yc[i]:
                if st.button(f"{yr}yr",key=f"cs_yr{yr}"): SS["last_n_years"]=yr
        SS["last_n_years"]=st.number_input("Yrs",1,30,int(SS.get("last_n_years",3)),key="cs_ny")
    SS["session_filter"]=st.selectbox("Session",
        ["Full Session","Regular Trading Hours Only","Overnight Only","Pre-Market"],
        index=["Full Session","Regular Trading Hours Only","Overnight Only","Pre-Market"]
            .index(SS.get("session_filter","Full Session")),key="cs_sf")
    _card_end()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render_section_candlestick(SS:dict):
    section_header("CANDLESTICK CHART",
                   "Smooth Replay Backtest · Live P&L/DD · P&L Distribution · Evaluation · Phase Two")

    raw_df    = SS.get("raw_df")
    result    = SS.get("backtest_result")
    trades_df = result.trades    if result and result.trades is not None    else pd.DataFrame()
    equity_df = result.equity_curve if result and result.equity_curve is not None else None

    if raw_df is None:
        banner_info("Load data first — Section 3 (native pre-loaded data or live API).")
        return

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️  BACKTEST CONFIGURATION  (expand to edit)", expanded=False):
        ca,cb=st.columns(2)
        with ca: _cfg_account(SS); _cfg_fees(SS)
        with cb: _cfg_stop_target(SS); _cfg_sizing(SS); _cfg_period(SS)

    # ── Chart controls ────────────────────────────────────────────────────────
    st.markdown("---")
    ct1,ct2,ct3,ct4=st.columns([1.5,1,1,1])
    with ct1:
        tf=st.selectbox("Display Timeframe",
                        ["1m","5m","15m","30m","1h","4h","1d"],index=4,key="cs_tf")
    with ct2:
        out_f=st.selectbox("Trade Filter",
                            ["ALL","WIN","LOSS","TIME","EOD","GAP"],key="cs_of")
    with ct3:
        show_vol =st.toggle("Volume",True,key="cs_vol")
        show_vwap=st.toggle("VWAP",  False,key="cs_vw")
    with ct4:
        st.toggle("SL lines",True, key="cs_sl")
        st.toggle("TP lines",True, key="cs_tp")

    # Resample to display timeframe
    rmap={"1m":"1min","5m":"5min","15m":"15min","30m":"30min",
          "1h":"1h","4h":"4h","1d":"1D"}
    rule=rmap.get(tf,"1h")
    try:
        agg={"open":"first","high":"max","low":"min","close":"last"}
        if "volume" in raw_df.columns: agg["volume"]="sum"
        chart_df = (raw_df.resample(rule).agg(agg).dropna(subset=["close"])
                    if len(raw_df)>2000 else raw_df.copy())
    except Exception:
        chart_df=raw_df.copy()

    vis_t=trades_df.copy() if not trades_df.empty else pd.DataFrame()
    if out_f!="ALL" and not vis_t.empty:
        vis_t=vis_t[vis_t["outcome"]==out_f]

    # ── STATIC chart (full history overview) ──────────────────────────────────
    st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{_G};'
                f'margin-bottom:4px;">◈ STATIC CHART — FULL HISTORY VIEW</div>',
                unsafe_allow_html=True)
    st.plotly_chart(_static_candle_fig(chart_df,vis_t,show_vol,show_vwap),
                    use_container_width=True,config=_CHART_CONFIG,key="cs_static")

    # ── Live P&L + Drawdown tracker ───────────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{_G};">'
                f'◈ LIVE P&L & DRAWDOWN TRACKER</div>',unsafe_allow_html=True)
    if equity_df is not None and not equity_df.empty:
        eq=equity_df["equity"].values; s0=eq[0]
        curr_pnl=eq[-1]-s0; max_dd=float((np.maximum.accumulate(eq)-eq).max())
        kc=st.columns(4)
        with kc[0]: st.metric("Current P&L",f"${curr_pnl:+,.0f}")
        with kc[1]: st.metric("Max Drawdown",f"-${max_dd:,.0f}")
        with kc[2]: st.metric("Peak Balance",f"${np.maximum.accumulate(eq)[-1]:,.0f}")
        with kc[3]: st.metric("Current Balance",f"${eq[-1]:,.0f}")
        st.plotly_chart(_tracker_fig(equity_df),use_container_width=True,key="cs_tracker")
    else:
        st.markdown('<div style="font-family:monospace;color:#333;padding:10px;">'
                    'Run a backtest (Section 4) to activate the live P&L tracker.</div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SMOOTH ANIMATED REPLAY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        f'<div style="background:#050A05;border:1px solid {_G};border-radius:8px;'
        f'padding:12px 18px;margin-bottom:10px;">'
        f'<div style="font-family:monospace;font-size:0.84rem;color:{_G};font-weight:700;">'
        f'▶ REPLAY BACKTEST — SMOOTH ANIMATION</div>'
        f'<div style="font-family:monospace;font-size:0.72rem;color:#666;margin-top:4px;">'
        f'Client-side Plotly animation — runs entirely in your browser. '
        f'Press ▶ PLAY below the chart. Drag the slider to scrub manually. '
        f'Trade markers (▲▼), SL lines and TP lines appear as bars load.</div></div>',
        unsafe_allow_html=True)

    # Controls
    ac1,ac2,ac3,ac4=st.columns(4)
    with ac1:
        max_frames=st.select_slider(
            "Animation frames",
            options=[50,100,150,200,300,400,500],
            value=200, key="cs_frames",
            format_func=lambda v: f"{v} frames",
        )
    with ac2:
        speed_ms=st.select_slider(
            "Frame speed",
            options=[30,50,80,120,200,400],
            value=80, key="cs_speed",
            format_func=lambda v: f"{v}ms/frame",
        )
    with ac3:
        s_pct=st.slider("Start at %",0,80,0,5,key="cs_sp")
    with ac4:
        e_pct=st.slider("End at %",10,100,100,5,key="cs_ep")

    n_bars=len(chart_df)
    s_bar=int(n_bars*s_pct/100)
    e_bar=int(n_bars*e_pct/100)
    replay_df=chart_df.iloc[s_bar:e_bar] if e_bar>s_bar else chart_df.copy()

    # Filter trades to replay window
    replay_trades=pd.DataFrame()
    if not vis_t.empty and "entry_time" in vis_t.columns:
        if len(replay_df)>0:
            win_start=replay_df.index[0]; win_end=replay_df.index[-1]
            replay_trades=vis_t[
                (pd.to_datetime(vis_t["entry_time"])>=win_start) &
                (pd.to_datetime(vis_t["entry_time"])<=win_end)
            ]

    # Build animation button
    if st.button("🎬  BUILD & LAUNCH ANIMATION",type="primary",
                 use_container_width=True,key="cs_anim_build"):
        n_replay=len(replay_df)
        if n_replay<10:
            banner_warning("Not enough bars in selected range.")
        else:
            with st.spinner(
                f"Building {max_frames}-frame animation for {n_replay:,} bars "
                f"({s_pct}%→{e_pct}% of data)…"
            ):
                anim_fig=build_animated_replay(
                    replay_df, replay_trades,
                    max_frames=max_frames,
                    show_vol=show_vol,
                    speed_ms=speed_ms,
                )
            SS["cs_anim_fig"]=anim_fig
            banner_success(
                f"Animation ready — {max_frames} frames across {n_replay:,} bars. "
                f"Press ▶ PLAY in the chart controls below."
            )

    # Render animation if built
    if SS.get("cs_anim_fig") is not None:
        st.plotly_chart(SS["cs_anim_fig"],
                        use_container_width=True,
                        config=_CHART_CONFIG,
                        key="cs_anim_chart")
        st.markdown(
            f'<div style="font-family:monospace;font-size:0.72rem;color:#555;margin-top:4px;">'
            f'▶ PLAY / ⏸ PAUSE buttons appear at top-left of chart. '
            f'Use the slider at bottom to scrub to any point in time.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="font-family:monospace;color:#333;padding:20px;text-align:center;">'
            f'Click 🎬 BUILD & LAUNCH ANIMATION above to start.</div>',
            unsafe_allow_html=True)

    # Jump to trade
    if not vis_t.empty and "entry_time" in vis_t.columns:
        with st.expander("🔍 Jump to Trade (zoom static view)"):
            opts=["(none)"]+[
                f"#{int(r.get('trade_id',i+1))} · {r.get('signal','?')} · "
                f"{str(r.get('entry_time',''))[:16]} · ${r.get('net_pnl',0):+,.0f}"
                for i,(_,r) in enumerate(vis_t.iterrows())
            ]
            jump=st.selectbox("Trade",opts,key="cs_jump")
            if jump!="(none)":
                idx2=opts.index(jump)-1
                row=vis_t.iloc[idx2]
                et=pd.Timestamp(row["entry_time"])
                win=pd.Timedelta(hours=6)
                zoom=chart_df[(chart_df.index>=et-win)&(chart_df.index<=et+win)]
                if not zoom.empty:
                    zt=vis_t[abs((pd.to_datetime(vis_t["entry_time"])-et).dt.total_seconds())<86400*3]
                    fz=_static_candle_fig(zoom,zt if not zt.empty else None,show_vol,show_vwap)
                    fz.update_layout(title=dict(
                        text=f"Trade #{int(row.get('trade_id',idx2+1))} — "
                             f"{row.get('signal','?')} @ {row.get('entry_price',0):,.2f}"))
                    st.plotly_chart(fz,use_container_width=True,config=_CHART_CONFIG,key="cs_zoom")

    # ── P&L Distribution ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div style="font-family:monospace;font-size:0.80rem;color:{_G};">◈ P&L DISTRIBUTION MODEL</div>',unsafe_allow_html=True)
    pd1,pd2,pd3=st.columns(3)
    with pd1: dsplit=st.selectbox("Split by",["signal","direction","regime","none"],key="cs_dspl")
    with pd2: ddaily=st.toggle("Daily returns view",equity_df is not None,key="cs_dd")
    with pd3: dbins=st.slider("Bins",20,120,60,5,key="cs_dbins")
    if not trades_df.empty:
        st.plotly_chart(build_pnl_distribution(trades_df,equity_df,dsplit,
            ddaily and equity_df is not None,dbins,"P&L Distribution"),
            use_container_width=True,key="cs_dist")
        with st.expander("📊 Distribution Statistics"):
            st.dataframe(build_return_stats_table(trades_df,equity_df),use_container_width=True,hide_index=True)
    else:
        st.markdown('<div style="font-family:monospace;color:#333;padding:10px;">Run a backtest to see distribution.</div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUATION + PHASE TWO TABS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    ev_tab,p2_tab=st.tabs(["🎯 EVALUATION","🚀 PHASE TWO — RUN-UP"])

    with ev_tab:
        st.markdown(f"""<div style="background:#050510;border:2px solid {_G};border-radius:10px;
padding:16px 20px;margin-bottom:14px;">
<div style="font-family:monospace;font-size:0.92rem;color:{_G};font-weight:700;">🎯 EVALUATION</div>
<div style="font-family:monospace;font-size:0.74rem;color:#666;margin-top:6px;">
Goal: Reach the <span style="color:{_G};">net profit target</span> before the account
hits the <span style="color:{_R};">max drawdown floor</span>.<br>
Green line = profit target. Red line = drawdown limit. Each path = one evaluation attempt.
</div></div>""",unsafe_allow_html=True)
        _card("◈ EVALUATION SETTINGS",_G)
        ea,eb=st.columns(2)
        with ea:
            for amt in [25_000,50_000,100_000,150_000]:
                if st.button(f"${amt//1000}k",key=f"ev_b{amt}"): SS["cs_ev_bal"]=float(amt)
            SS["cs_ev_bal"]=st.number_input("Start balance ($)",1_000.0,5e6,float(SS.get("cs_ev_bal",50_000)),1_000.0,key="cs_evbali")
            mc1,mc2=st.columns(2)
            with mc1: SS["cs_ev_min"]=st.number_input("Min days",1,90,int(SS.get("cs_ev_min",5)),key="cs_evmd")
            with mc2: SS["cs_ev_max"]=st.number_input("Max days",1,90,int(SS.get("cs_ev_max",30)),key="cs_evxd")
        with eb:
            bal_ev=SS.get("cs_ev_bal",50_000)
            for pct in [6,8,10,12]:
                if st.button(f"+{pct}% (${bal_ev*pct//100:,.0f})",key=f"ev_pt{pct}"): SS["cs_ev_profit"]=bal_ev*pct/100
            SS["cs_ev_profit"]=st.number_input("Profit target ($)",0.0,1e6,float(SS.get("cs_ev_profit",bal_ev*0.10)),100.0,key="cs_evpti")
            for pct in [3,5,6,8]:
                if st.button(f"-{pct}% max DD",key=f"ev_dd{pct}"): SS["cs_ev_maxdd"]=bal_ev*pct/100
            SS["cs_ev_maxdd"]=st.number_input("Max DD limit ($)",0.0,5e5,float(SS.get("cs_ev_maxdd",bal_ev*0.06)),100.0,key="cs_evddi")
            SS["cs_ev_dl"]=st.number_input("Daily loss limit ($)",0.0,1e5,float(SS.get("cs_ev_dl",bal_ev*0.04)),100.0,key="cs_evdli")
        _card_end()
        SS["cs_ev_paths"]=int(st.number_input("Paths",100,50_000,int(SS.get("cs_ev_paths",1_000)),100,key="cs_evnp"))
        if st.button("▶  RUN EVALUATION",type="primary",key="cs_ev_run"):
            if trades_df.empty: banner_warning("Run a backtest first.")
            else:
                with st.spinner("Running evaluation…"):
                    er=_run_evaluation(trades_df["net_pnl"].values.astype(float),
                        SS.get("cs_ev_bal",50_000),SS.get("cs_ev_profit",5_000),
                        SS.get("cs_ev_maxdd",3_000),SS.get("cs_ev_dl",2_000),
                        SS.get("cs_ev_min",5),SS.get("cs_ev_max",30),SS.get("cs_ev_paths",1_000))
                SS["cs_ev_result"]=er
                banner_success(f"Evaluation — PASS {er['pass_rate']:.1f}%  |  Avg day {er['avg_pass_day']:.1f}")
        if SS.get("cs_ev_result"):
            er=SS["cs_ev_result"]
            st.plotly_chart(_eval_fig(er),use_container_width=True,key="cs_evfig")
            mc1,mc2=st.columns(2)
            with mc1:
                metric_grid([("✅ PASS Rate",f"{er['pass_rate']:.2f}%","green"),
                             ("❌ FAIL — Max DD",f"{er['fail_dd_rate']:.2f}%","red"),
                             ("❌ FAIL — Daily",f"{er['fail_dl_rate']:.2f}%","red"),
                             ("⏱ FAIL — Timeout",f"{er['fail_time_rate']:.2f}%","amber"),
                             ("Avg Days to PASS",f"Day {er['avg_pass_day']:.1f}","blue")],"EVALUATION RESULTS")
            with mc2:
                metric_grid([("Start Balance",f"${er['start_bal']:,.0f}","white"),
                             ("Profit Target",f"+${SS.get('cs_ev_profit',5000):,.0f} → ${er['goal']:,.0f}","green"),
                             ("Max DD Floor",f"${er['start_bal']-er['max_dd_limit']:,.0f}","red"),
                             ("Paths Simulated",f"{er['n_paths']:,}","white")],"PARAMETERS")

    with p2_tab:
        st.markdown(f"""<div style="background:#050510;border:2px solid {_B};border-radius:10px;
padding:16px 20px;margin-bottom:14px;">
<div style="font-family:monospace;font-size:0.92rem;color:{_B};font-weight:700;">🚀 PHASE TWO — LIVE ACCOUNT RUN-UP</div>
<div style="font-family:monospace;font-size:0.74rem;color:#666;margin-top:6px;">
Goal: <span style="color:{_B};">Build account balance as high as possible</span> without going 
<span style="color:{_R};">completely negative</span>.<br>
EOD Tracking monitors end-of-day balance. Intraday DD halts same-day trading if breached.
</div></div>""",unsafe_allow_html=True)
        _card("◈ PHASE TWO SETTINGS",_B)
        p2a,p2b=st.columns(2)
        with p2a:
            for amt in [10_000,25_000,50_000,100_000,150_000]:
                if st.button(f"${amt//1000}k",key=f"p2_b{amt}"): SS["cs_p2_bal"]=float(amt)
            SS["cs_p2_bal"]=st.number_input("Start balance ($)",1_000.0,2e6,float(SS.get("cs_p2_bal",50_000)),1_000.0,key="cs_p2bali")
            bal_p2=SS.get("cs_p2_bal",50_000)
            for f_pct in [0,2,5]:
                fval=0 if f_pct==0 else bal_p2*(1-f_pct/100)
                if st.button(f"Floor {'$0' if f_pct==0 else f'-{f_pct}%'}",key=f"p2_fl{f_pct}"): SS["cs_p2_floor"]=fval
            SS["cs_p2_floor"]=st.number_input("Hard floor ($)",-1e6,1e6,float(SS.get("cs_p2_floor",0)),100.0,key="cs_p2fli")
            SS["cs_p2_dl"]=st.number_input("Daily loss limit ($)",0.0,1e6,float(SS.get("cs_p2_dl",2_000)),100.0,key="cs_p2dli")
            for mult in [1.5,2.0,2.5,3.0]:
                if st.button(f"×{mult} target",key=f"p2_tg{mult}"): SS["cs_p2_target"]=bal_p2*mult
            SS["cs_p2_target"]=st.number_input("Balance target ($)",0.0,1e7,float(SS.get("cs_p2_target",bal_p2*2)),500.0,key="cs_p2tgti")
        with p2b:
            SS["cs_p2_eod"]=st.toggle("EOD Balance Tracking",SS.get("cs_p2_eod",True),key="cs_p2eod")
            if SS["cs_p2_eod"]: st.caption("Account blown if EOD balance ≤ hard floor.")
            SS["cs_p2_intraday"]=st.toggle("Intraday Drawdown Tracking",SS.get("cs_p2_intraday",False),key="cs_p2id")
            if SS["cs_p2_intraday"]:
                SS["cs_p2_id_limit"]=st.number_input("Intraday DD limit ($)",100.0,1e6,float(SS.get("cs_p2_id_limit",1_500)),100.0,key="cs_p2idli")
            else: SS["cs_p2_id_limit"]=float("inf")
            default_lvls=[bal_p2*1.06,bal_p2*1.12,bal_p2*1.20,bal_p2*1.30]
            lv_str=st.text_area("Payout trigger balances (one per line)",
                value="\n".join(f"{l:,.0f}" for l in SS.get("cs_p2_levels",default_lvls)),
                height=100,key="cs_p2lvs")
            try: p2_lvls=[float(x.replace(",","").strip()) for x in lv_str.splitlines() if x.strip()]
            except Exception: p2_lvls=default_lvls
            SS["cs_p2_levels"]=p2_lvls
            SS["cs_p2_payout"]=st.number_input("Withdrawal/payout ($)",0.0,1e6,float(SS.get("cs_p2_payout",500)),50.0,key="cs_p2pwi")
        _card_end()
        pp1,pp2=st.columns(2)
        with pp1: SS["cs_p2_paths"]=int(st.number_input("Paths",100,20_000,int(SS.get("cs_p2_paths",1_000)),100,key="cs_p2np"))
        with pp2: SS["cs_p2_hor"]=int(st.number_input("Horizon (days)",5,2520,int(SS.get("cs_p2_hor",252)),5,key="cs_p2hori"))
        if st.button("🚀 RUN PHASE TWO",type="primary",key="cs_p2_run"):
            if trades_df.empty: banner_warning("Run a backtest first.")
            else:
                with st.spinner("Running Phase Two…"):
                    p2r=_run_phase_two(trades_df["net_pnl"].values.astype(float),
                        float(SS.get("cs_p2_bal",50_000)),float(SS.get("cs_p2_floor",0)),
                        float(SS.get("cs_p2_dl",2_000)),SS.get("cs_p2_levels",[]),
                        float(SS.get("cs_p2_payout",500)),float(SS.get("cs_p2_target",100_000)),
                        int(SS.get("cs_p2_paths",1_000)),int(SS.get("cs_p2_hor",252)),
                        bool(SS.get("cs_p2_eod",True)),bool(SS.get("cs_p2_intraday",False)),
                        float(SS.get("cs_p2_id_limit",1_500)))
                SS["cs_p2_result"]=p2r
                banner_success(f"Phase Two — Survival {p2r['survival_rate']:.1f}%  |  Target hit {p2r['target_rate']:.1f}%")
        if SS.get("cs_p2_result"):
            p2r=SS["cs_p2_result"]
            st.plotly_chart(_p2_fig(p2r,SS.get("cs_p2_levels",[])),use_container_width=True,key="cs_p2fig")
            pc1,pc2=st.columns(2)
            with pc1:
                metric_grid([("✅ Survival Rate",f"{p2r['survival_rate']:.2f}%","green"),
                             ("💀 Blown Rate",f"{p2r['blown_rate']:.2f}%","red"),
                             ("🎯 Target Hit Rate",f"{p2r['target_rate']:.2f}%","blue"),
                             ("Avg Days to Target",f"{p2r['avg_days_target']:.1f}d","amber"),
                             ("Avg EOD Negative Days",f"{p2r['avg_eod_neg_days']:.1f}","red")],"PHASE TWO RESULTS")
            with pc2:
                metric_grid([("Median Final Balance",f"${p2r['median_final']:,.0f}","white"),
                             ("5th Pct",f"${p2r['p5_final']:,.0f}","red"),
                             ("95th Pct",f"${p2r['p95_final']:,.0f}","green"),
                             ("Avg Payouts",f"{p2r['avg_payouts']:.1f}","amber"),
                             ("Total Withdrawn",f"${p2r['total_withdrawn']:,.0f}","green"),
                             ("Paths",f"{p2r['n_paths']:,}","white")],"BALANCE STATS")

    st.markdown("---")
    st.markdown(f'<div style="font-family:monospace;font-size:0.73rem;color:#555;">'
                f'<span style="color:{_G};">▲ LONG  ···TP</span> &nbsp;'
                f'<span style="color:{_R};">▼ SHORT  ---SL</span> &nbsp;'
                f'<span style="color:{_G};">✓ WIN</span> &nbsp;'
                f'<span style="color:{_R};">✗ LOSS</span> &nbsp;'
                f'<span style="color:{_A};">⏱ TIME</span></div>',unsafe_allow_html=True)
