"""
pnl_distribution.py
════════════════════
Shared PNL distribution chart module.

Renders:
  - Daily return histogram (overlaid by signal/strategy type, matching Image 1)
  - Trade P&L distribution
  - Return percentile table
  - Skewness / kurtosis / tail-risk metrics
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLORS, PLOTLY_DARK

_G  = COLORS["green"]
_R  = COLORS["red"]
_A  = COLORS["amber"]
_B  = COLORS["blue"]
_W  = COLORS["text"]
_C  = COLORS["card_bg"]

# Palette matching the reference screenshot (blue/orange/green tones)
_PALETTE = [
    "rgba(100,149,237,0.72)",   # cornflower blue  — strategy 1 / long
    "rgba(255,160,100,0.72)",   # soft orange       — strategy 2 / short
    "rgba(100,200,120,0.72)",   # sage green        — strategy 3 / all
    "rgba(200,100,200,0.60)",   # purple            — carry/extra
    "rgba(255,220,80,0.60)",    # gold              — extra
]


def build_pnl_distribution(
    trades_df:   pd.DataFrame,
    equity_df:   Optional[pd.DataFrame] = None,
    split_by:    str = "signal",          # "signal" | "direction" | "regime" | "none"
    show_daily:  bool = True,             # daily returns or per-trade P&L
    n_bins:      int  = 60,
    title:       str  = "P&L Distribution",
) -> go.Figure:
    """
    Build a stacked/overlaid histogram of P&L distribution.

    Parameters
    ----------
    trades_df : trade log DataFrame
    equity_df : optional equity curve (for daily return calculation)
    split_by  : grouping key for multi-series overlay
    show_daily: if True and equity_df provided, use daily returns; else per-trade
    n_bins    : histogram bin count

    Returns
    -------
    plotly Figure matching the reference screenshot dark style
    """
    if trades_df is None or trades_df.empty:
        return _empty_fig(title)

    fig = go.Figure()

    # ── Compute series to plot ────────────────────────────────────────────────
    if show_daily and equity_df is not None and "equity" in equity_df.columns:
        daily_eq  = equity_df["equity"].resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()

        # Main distribution
        fig.add_trace(go.Histogram(
            x=daily_ret.values,
            name="daily returns (all)",
            marker_color=_PALETTE[0],
            nbinsx=n_bins,
            opacity=0.80,
        ))
        fig.add_vline(x=0, line=dict(color="#111111", width=2))
        fig.update_layout(
            xaxis_title="daily returns",
            yaxis_title="Count",
        )
    else:
        # Per-trade P&L, optionally split by grouping key
        pnl = trades_df["net_pnl"].values if "net_pnl" in trades_df.columns else np.array([])

        if split_by != "none" and split_by in trades_df.columns:
            groups = trades_df[split_by].unique()
            for gi, grp in enumerate(groups[:5]):  # cap at 5 series
                sub = trades_df[trades_df[split_by] == grp]["net_pnl"].values
                fig.add_trace(go.Histogram(
                    x=sub,
                    name=f"pnl {str(grp).lower()[:10]}",
                    marker_color=_PALETTE[gi % len(_PALETTE)],
                    nbinsx=n_bins,
                    opacity=0.80,
                ))
        else:
            wins  = pnl[pnl >= 0]
            loss  = pnl[pnl  < 0]
            if len(wins): fig.add_trace(go.Histogram(x=wins, name="wins",
                marker_color=_PALETTE[2], nbinsx=n_bins//2, opacity=0.80))
            if len(loss): fig.add_trace(go.Histogram(x=loss, name="losses",
                marker_color=_PALETTE[1], nbinsx=n_bins//2, opacity=0.80))

        fig.add_vline(x=0, line=dict(color="#111111", width=2))
        fig.update_layout(
            xaxis_title="trade P&L ($)",
            yaxis_title="Count",
        )

    fig.update_layout(
        height=340,
        title=dict(text=title, font=dict(color=_W, size=12)),
        barmode="overlay",
        bargap=0.01,
        margin=dict(l=55, r=20, t=42, b=40),
        showlegend=True,
        legend=dict(
            bgcolor=_C, bordercolor=COLORS["card_border"],
            x=0.78, y=0.97,
            font=dict(size=10, color=_W),
        ),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor="#F0F2F6" if False else "#0F0F0F",
        font=dict(color=_W, family="'JetBrains Mono','Courier New',monospace"),
        xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"],
                   tickformat=".2%"  if show_daily else "$,.0f"),
        yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"]),
    )
    return fig


def build_return_stats_table(
    trades_df:  pd.DataFrame,
    equity_df:  Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of distribution statistics for display.
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    pnl = trades_df["net_pnl"].values.astype(float) if "net_pnl" in trades_df.columns else np.array([0.0])

    from scipy import stats as scipy_stats
    sk  = float(scipy_stats.skew(pnl))    if len(pnl) > 3 else 0.0
    kt  = float(scipy_stats.kurtosis(pnl)) if len(pnl) > 3 else 0.0

    rows = [
        ("Mean P&L",            f"${pnl.mean():+,.2f}"),
        ("Median P&L",          f"${np.median(pnl):+,.2f}"),
        ("Std Dev",             f"${pnl.std():,.2f}"),
        ("Skewness",            f"{sk:+.3f}"),
        ("Excess Kurtosis",     f"{kt:+.3f}"),
        ("5th Pct (tail risk)", f"${np.percentile(pnl,5):+,.2f}"),
        ("95th Pct",            f"${np.percentile(pnl,95):+,.2f}"),
        ("Min (worst trade)",   f"${pnl.min():+,.2f}"),
        ("Max (best trade)",    f"${pnl.max():+,.2f}"),
    ]

    if equity_df is not None and "equity" in equity_df.columns:
        daily_eq = equity_df["equity"].resample("1D").last().dropna()
        dr = daily_eq.pct_change().dropna().values
        if len(dr) > 2:
            rows += [
                ("Daily Return Mean",  f"{dr.mean()*100:+.3f}%"),
                ("Daily Return Std",   f"{dr.std()*100:.3f}%"),
                ("Daily Skewness",     f"{float(scipy_stats.skew(dr)):+.3f}"),
                ("Daily Kurtosis",     f"{float(scipy_stats.kurtosis(dr)):+.3f}"),
            ]

    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=300, title=dict(text=title, font=dict(color=_W, size=12)),
                      paper_bgcolor=COLORS["bg"], plot_bgcolor="#0F0F0F",
                      font=dict(color=_W))
    fig.add_annotation(text="No data — run a backtest first",
                       xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color="#555", size=14))
    return fig
