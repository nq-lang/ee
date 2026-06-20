"""
charts.py — All Plotly chart construction for the terminal.

Every chart uses the dark institutional theme from config.PLOTLY_DARK.
Charts are returned as plotly.graph_objects.Figure objects ready for
st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLORS, PLOTLY_DARK
from monte_carlo import MCResult


def _apply_dark(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply global dark theme and sizing to any figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=40, b=40),
        **PLOTLY_DARK,
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# 1. Equity Curve + Drawdown (dual-panel)
# ────────────────────────────────────────────────────────────────────────────

def equity_drawdown_chart(
    equity_df: pd.DataFrame,
    show_ma: bool = False,
    ma_window: int = 20,
) -> go.Figure:
    """
    Dual-panel: cumulative P&L (top) + drawdown (bottom).
    Both panels share the same X-axis with a range slider.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
    )

    times  = equity_df.index
    equity = equity_df["equity"].values
    dd     = equity_df["drawdown"].values if "drawdown" in equity_df.columns else np.zeros_like(equity)
    dd_pct = equity_df["drawdown_pct"].values if "drawdown_pct" in equity_df.columns else np.zeros_like(equity)

    # Cumulative P&L (relative to start)
    start_bal = equity[0] if len(equity) > 0 else 0
    pnl_curve = equity - start_bal

    # ── Top panel: equity ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=times, y=pnl_curve,
            mode="lines",
            name="Cumulative P&L",
            line=dict(color=COLORS["green"], width=2),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.07)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "P&L: $%{y:,.0f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Peak annotation
    if len(pnl_curve) > 0:
        peak_idx = int(np.argmax(pnl_curve))
        fig.add_annotation(
            x=times[peak_idx], y=pnl_curve[peak_idx],
            text=f"Peak: ${pnl_curve[peak_idx]:,.0f}",
            showarrow=True, arrowhead=2,
            arrowcolor=COLORS["green"],
            font=dict(color=COLORS["green"], size=11),
            bgcolor=COLORS["card_bg"], bordercolor=COLORS["green"],
            row=1, col=1,
        )

    # Optional MA
    if show_ma and len(pnl_curve) > ma_window:
        ma = pd.Series(pnl_curve).rolling(ma_window).mean().values
        fig.add_trace(
            go.Scatter(
                x=times, y=ma,
                mode="lines",
                name=f"{ma_window}-bar MA",
                line=dict(color=COLORS["amber"], width=1, dash="dot"),
                opacity=0.7,
            ),
            row=1, col=1,
        )

    # ── Bottom panel: drawdown ────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=times, y=-dd,
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["red"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,59,59,0.12)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Drawdown: -$%{customdata[0]:,.0f} ({customdata[1]:.2f}%)"
                "<extra></extra>"
            ),
            customdata=np.stack([dd, dd_pct], axis=-1),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text="Equity Curve & Drawdown", font=dict(color=COLORS["text"], size=14)),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis2=dict(
            rangeslider=dict(visible=True, bgcolor=COLORS["bg_secondary"], thickness=0.05),
        ),
    )
    fig.update_yaxes(tickformat="$,.0f", row=1, col=1, title_text="P&L")
    fig.update_yaxes(tickformat="$,.0f", row=2, col=1, title_text="Drawdown")

    return _apply_dark(fig, height=520)


# ────────────────────────────────────────────────────────────────────────────
# 2. Annual P&L Bar Chart
# ────────────────────────────────────────────────────────────────────────────

def annual_pnl_chart(
    annual_table: pd.DataFrame,
    label: str = "",
) -> go.Figure:
    """Green/red bar chart with dollar + trade-count annotations per year."""
    if annual_table.empty:
        return go.Figure()

    fig = go.Figure()
    colors = [COLORS["green"] if v >= 0 else COLORS["red"]
              for v in annual_table["TOTAL P&L"]]

    fig.add_trace(
        go.Bar(
            x=annual_table["YEAR"].astype(str),
            y=annual_table["TOTAL P&L"],
            marker_color=colors,
            text=[
                f"${v:+,.0f}\nn={n}"
                for v, n in zip(annual_table["TOTAL P&L"], annual_table["TRADES"])
            ],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=10),
            hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Annual P&L {label}",
        yaxis=dict(tickformat="$,.0f"),
        bargap=0.3,
    )
    return _apply_dark(fig, height=380)


# ────────────────────────────────────────────────────────────────────────────
# 3. Monthly P&L Heatmap
# ────────────────────────────────────────────────────────────────────────────

def monthly_heatmap(monthly_table: pd.DataFrame) -> go.Figure:
    """Year × Month heat map of net P&L."""
    if monthly_table.empty:
        return go.Figure()

    z      = monthly_table.values.astype(float)
    years  = monthly_table.index.astype(str).tolist()
    months = list(monthly_table.columns)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=months,
            y=years,
            colorscale=[
                [0.0, COLORS["red_dark"]],
                [0.3, "#662222"],
                [0.45, "#331111"],
                [0.5, COLORS["card_bg"]],
                [0.55, "#003322"],
                [0.7, "#006633"],
                [1.0, COLORS["green"]],
            ],
            zmid=0,
            text=[[f"${v:+,.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="P&L $", tickformat="$,.0f"),
            hovertemplate="<b>%{y} %{x}</b><br>P&L: $%{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly P&L Heatmap")
    return _apply_dark(fig, height=340)


# ────────────────────────────────────────────────────────────────────────────
# 4. Monte Carlo Fan Chart
# ────────────────────────────────────────────────────────────────────────────

def monte_carlo_fan_chart(
    mc_result: MCResult,
    show_paths: bool = False,
    combine_targets: Optional[dict] = None,
) -> go.Figure:
    """
    Fan chart with percentile bands and median path.
    Optionally overlays individual sample paths and combine threshold lines.
    """
    if mc_result is None or len(mc_result.pct_50) == 0:
        return go.Figure()

    n_days = len(mc_result.pct_50)
    days   = np.arange(n_days)
    fig    = go.Figure()

    # ── 5–95 outer band ──────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([mc_result.pct_95, mc_result.pct_5[::-1]]),
            fill="toself",
            fillcolor="rgba(0,200,80,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th–95th pct",
            hoverinfo="skip",
        )
    )

    # ── 25–75 inner band ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([mc_result.pct_75, mc_result.pct_25[::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,136,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th–75th pct",
            hoverinfo="skip",
        )
    )

    # ── Individual paths (optional) ───────────────────────────────────────────
    if show_paths and len(mc_result.paths) > 0:
        cfg  = mc_result.config
        n_disp = min(getattr(cfg, "max_display_paths", 100), len(mc_result.paths))
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(mc_result.paths), n_disp, replace=False)
        for i in idx:
            fig.add_trace(
                go.Scatter(
                    x=days, y=mc_result.paths[i],
                    mode="lines",
                    line=dict(color="rgba(0,255,136,0.06)", width=0.8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # ── Median line ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=days, y=mc_result.pct_50,
            mode="lines",
            name="Median",
            line=dict(color=COLORS["white"], width=2.5),
            hovertemplate="Day %{x}<br>Median P&L: $%{y:,.0f}<extra></extra>",
        )
    )

    # ── Combine threshold lines ───────────────────────────────────────────────
    if combine_targets and mc_result.config and mc_result.config.combine_mode:
        for label, level, color in [
            ("Profit Target", combine_targets.get("profit_target", 12000), COLORS["green"]),
            ("Max DD Limit",  -combine_targets.get("max_drawdown_limit", 6000), COLORS["red"]),
        ]:
            fig.add_hline(
                y=level,
                line=dict(color=color, dash="dash", width=1.5),
                annotation_text=label,
                annotation_font_color=color,
            )

    cfg_obj = mc_result.config
    method  = getattr(cfg_obj, "sampling_method", "") if cfg_obj else ""
    n_paths = len(mc_result.paths) if len(mc_result.paths) > 0 else 0
    horizon = n_days - 1

    fig.update_layout(
        title=f"Monte Carlo — {n_paths:,} paths × {horizon}d  [{method}]",
        xaxis_title="Trading Day",
        yaxis=dict(tickformat="$,.0f"),
        showlegend=True,
    )
    return _apply_dark(fig, height=480)


# ────────────────────────────────────────────────────────────────────────────
# 5. Monte Carlo Final Equity Distribution (histogram)
# ────────────────────────────────────────────────────────────────────────────

def monte_carlo_histogram(mc_result: MCResult) -> go.Figure:
    """Distribution of final equity across all simulation paths."""
    if mc_result is None or len(mc_result.final_equity) == 0:
        return go.Figure()

    final = mc_result.final_equity
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in final]

    fig = go.Figure()
    # Split into profit/loss for color coding
    profit = final[final >= 0]
    loss   = final[final < 0]

    if len(profit):
        fig.add_trace(
            go.Histogram(
                x=profit, name="Profitable",
                marker_color=COLORS["green"], opacity=0.75,
                nbinsx=50,
            )
        )
    if len(loss):
        fig.add_trace(
            go.Histogram(
                x=loss, name="Loss",
                marker_color=COLORS["red"], opacity=0.75,
                nbinsx=30,
            )
        )

    # Percentile lines
    for pct_val, label, color in [
        (mc_result.p5_final,    "5th pct",  COLORS["red"]),
        (np.percentile(final,25), "25th pct", COLORS["amber"]),
        (mc_result.median_final,"Median",   COLORS["white"]),
        (np.percentile(final,75), "75th pct", COLORS["amber"]),
        (mc_result.p95_final,   "95th pct", COLORS["green"]),
    ]:
        fig.add_vline(
            x=pct_val,
            line=dict(color=color, dash="dash", width=1.5),
            annotation_text=label,
            annotation_font_color=color,
        )

    fig.update_layout(
        title="Final Equity Distribution",
        xaxis=dict(tickformat="$,.0f"),
        barmode="overlay",
    )
    return _apply_dark(fig, height=340)


# ────────────────────────────────────────────────────────────────────────────
# 6. Interactive Candlestick Chart with Trade Markers
# ────────────────────────────────────────────────────────────────────────────

def candlestick_chart(
    ohlcv_df:     pd.DataFrame,
    trades_df:    pd.DataFrame,
    show_stops:   bool = True,
    show_targets: bool = True,
    show_shading: bool = True,
    show_volume:  bool = True,
    show_vwap:    bool = False,
    outcome_filter: str = "ALL",
    signal_filter:  list = None,
) -> go.Figure:
    """
    Full TradingView-style candlestick chart with:
      - Green/red candles
      - Long/short entry markers (▲/▼)
      - Stop loss dashed red lines
      - Target dashed green lines
      - Trade shading (green win / red loss)
      - Volume sub-panel
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return go.Figure()

    rows   = 2 if show_volume else 1
    heights = [0.75, 0.25] if show_volume else [1.0]
    fig    = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.03,
    )

    ts  = ohlcv_df.index
    op  = ohlcv_df["open"].values
    hi  = ohlcv_df["high"].values
    lo  = ohlcv_df["low"].values
    cl  = ohlcv_df["close"].values

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=ts, open=op, high=hi, low=lo, close=cl,
            name="Price",
            increasing=dict(
                line=dict(color=COLORS["green"], width=1),
                fillcolor=COLORS["green_mid"],
            ),
            decreasing=dict(
                line=dict(color=COLORS["red"], width=1),
                fillcolor="#CC2222",
            ),
            hoverinfo="x+y",
        ),
        row=1, col=1,
    )

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if show_vwap and "vwap" in ohlcv_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ts, y=ohlcv_df["vwap"],
                mode="lines",
                name="VWAP",
                line=dict(color=COLORS["amber"], width=1, dash="dot"),
                opacity=0.8,
            ),
            row=1, col=1,
        )

    # ── Volume ────────────────────────────────────────────────────────────────
    if show_volume and "volume" in ohlcv_df.columns:
        vol_colors = [
            COLORS["green"] if cl[i] >= op[i] else COLORS["red"]
            for i in range(len(cl))
        ]
        fig.add_trace(
            go.Bar(
                x=ts, y=ohlcv_df["volume"],
                marker_color=vol_colors, opacity=0.5,
                name="Volume",
            ),
            row=2, col=1,
        )

    # ── Trade overlays ────────────────────────────────────────────────────────
    if trades_df is not None and not trades_df.empty:
        _add_trade_overlays(
            fig, trades_df, show_stops, show_targets, show_shading,
            outcome_filter, signal_filter,
        )

    fig.update_layout(
        title="Price Chart with Trade Executions",
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_yaxes(tickformat=",", row=1, col=1)
    if show_volume:
        fig.update_yaxes(tickformat=",", row=2, col=1)

    return _apply_dark(fig, height=600)


def _add_trade_overlays(
    fig, trades_df, show_stops, show_targets, show_shading,
    outcome_filter, signal_filter
):
    """Add entry markers, stop/target lines, and shading rectangles."""
    df = trades_df.copy()

    if outcome_filter != "ALL":
        df = df[df["outcome"] == outcome_filter]
    if signal_filter:
        df = df[df["signal"].isin(signal_filter)]

    if df.empty:
        return

    long_entries  = df[df["direction"] == "LONG"]
    short_entries = df[df["direction"] == "SHORT"]

    # ── Long entries ──────────────────────────────────────────────────────────
    if not long_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=long_entries["entry_time"],
                y=long_entries["entry_price"],
                mode="markers",
                name="Long Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color=COLORS["green"],
                    line=dict(color=COLORS["green"], width=1),
                ),
                hovertemplate=(
                    "<b>LONG: %{customdata[0]}</b><br>"
                    "Entry: $%{y:,.2f}<br>"
                    "Stop: $%{customdata[1]:,.2f}<br>"
                    "Target: $%{customdata[2]:,.2f}<br>"
                    "Outcome: %{customdata[3]}<br>"
                    "P&L: $%{customdata[4]:+,.2f}<extra></extra>"
                ),
                customdata=long_entries[[
                    "signal", "stop_price", "target_price", "outcome", "net_pnl"
                ]].values,
            ),
            row=1, col=1,
        )

    # ── Short entries ─────────────────────────────────────────────────────────
    if not short_entries.empty:
        fig.add_trace(
            go.Scatter(
                x=short_entries["entry_time"],
                y=short_entries["entry_price"],
                mode="markers",
                name="Short Entry",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color=COLORS["red"],
                    line=dict(color=COLORS["red"], width=1),
                ),
                hovertemplate=(
                    "<b>SHORT: %{customdata[0]}</b><br>"
                    "Entry: $%{y:,.2f}<br>"
                    "Stop: $%{customdata[1]:,.2f}<br>"
                    "Target: $%{customdata[2]:,.2f}<br>"
                    "Outcome: %{customdata[3]}<br>"
                    "P&L: $%{customdata[4]:+,.2f}<extra></extra>"
                ),
                customdata=short_entries[[
                    "signal", "stop_price", "target_price", "outcome", "net_pnl"
                ]].values,
            ),
            row=1, col=1,
        )

    # ── Stop and target lines per trade ───────────────────────────────────────
    for _, t in df.iterrows():
        if pd.isna(t.get("entry_time")) or pd.isna(t.get("exit_time")):
            continue
        if show_stops:
            fig.add_shape(
                type="line",
                x0=t["entry_time"], x1=t["exit_time"],
                y0=t["stop_price"], y1=t["stop_price"],
                line=dict(color=COLORS["red"], dash="dash", width=1),
                row=1, col=1,
            )
        if show_targets:
            fig.add_shape(
                type="line",
                x0=t["entry_time"], x1=t["exit_time"],
                y0=t["target_price"], y1=t["target_price"],
                line=dict(color=COLORS["green"], dash="dash", width=1),
                row=1, col=1,
            )
        if show_shading:
            color = "rgba(0,255,136,0.04)" if t.get("net_pnl", 0) >= 0 \
                    else "rgba(255,59,59,0.05)"
            fig.add_vrect(
                x0=t["entry_time"], x1=t["exit_time"],
                fillcolor=color, line_width=0,
                row=1, col=1,
            )


# ────────────────────────────────────────────────────────────────────────────
# 7. Day-of-week bar chart
# ────────────────────────────────────────────────────────────────────────────

def dow_chart(dow_table: pd.DataFrame) -> go.Figure:
    if dow_table.empty:
        return go.Figure()
    colors = [
        COLORS["green"] if float(str(v).replace("$","").replace(",","").replace("+","")) >= 0
        else COLORS["red"]
        for v in dow_table["Avg P&L"]
    ]
    avg_vals = pd.to_numeric(
        dow_table["Avg P&L"].astype(str)
        .str.replace(r"[\$,+]", "", regex=True),
        errors="coerce"
    ).fillna(0)
    fig = go.Figure(
        go.Bar(
            x=dow_table["Day"], y=avg_vals,
            marker_color=colors,
            text=[f"${v:+,.0f}" for v in avg_vals],
            textposition="outside",
        )
    )
    fig.update_layout(title="Average P&L by Day of Week",
                      yaxis=dict(tickformat="$,.0f"))
    return _apply_dark(fig, height=340)


# ────────────────────────────────────────────────────────────────────────────
# 8. Time-of-day heatmap
# ────────────────────────────────────────────────────────────────────────────

def tod_chart(tod_table: pd.DataFrame) -> go.Figure:
    if tod_table.empty:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=tod_table["Hour"],
            y=tod_table["Avg P&L"],
            marker_color=[
                COLORS["green"] if v >= 0 else COLORS["red"]
                for v in tod_table["Avg P&L"]
            ],
            hovertemplate="<b>%{x}</b><br>Avg P&L: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="Average P&L by Hour of Day",
                      yaxis=dict(tickformat="$,.0f"))
    return _apply_dark(fig, height=320)


# ────────────────────────────────────────────────────────────────────────────
# 9. Quarterly bar chart
# ────────────────────────────────────────────────────────────────────────────

def quarterly_chart(trades_df: pd.DataFrame) -> go.Figure:
    if trades_df is None or "entry_time" not in trades_df.columns:
        return go.Figure()
    df = trades_df.copy()
    df["quarter"] = pd.to_datetime(df["entry_time"]).dt.to_period("Q").astype(str)
    grp = df.groupby("quarter")["net_pnl"].sum().reset_index()
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in grp["net_pnl"]]
    fig = go.Figure(
        go.Bar(
            x=grp["quarter"], y=grp["net_pnl"],
            marker_color=colors,
            text=[f"${v:+,.0f}" for v in grp["net_pnl"]],
            textposition="outside",
        )
    )
    fig.update_layout(title="Quarterly P&L", yaxis=dict(tickformat="$,.0f"))
    return _apply_dark(fig, height=340)
