"""
export.py — All download/export functionality.

PDF  : Built with reportlab (dark-themed multi-page report)
CSV  : Trade log, equity curve, Monte Carlo endpoints
JSON : Config snapshot for reproducibility
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── PDF engine (reportlab) ────────────────────────────────────────────────────
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ── Plotly PNG export (kaleido) ───────────────────────────────────────────────
try:
    import plotly.io as pio
    KALEIDO_OK = True
except ImportError:
    KALEIDO_OK = False


# ── Colour palette (mirrors CSS) ─────────────────────────────────────────────
_BLACK      = "#0A0A0A"
_DARK       = "#1A1A1A"
_BORDER     = "#2A2A2A"
_TEXT       = "#E0E0E0"
_GREEN      = "#00FF88"
_RED        = "#FF3B3B"
_AMBER      = "#FFD700"
_BLUE       = "#00BFFF"


# ─────────────────────────────────────────────────────────────────────────────
# CSV exports
# ─────────────────────────────────────────────────────────────────────────────

def export_trade_log_csv(trades_df: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes of the full trade log."""
    if trades_df is None or trades_df.empty:
        return b"No trades to export.\n"
    buf = io.StringIO()
    trades_df.to_csv(buf, index=False, float_format="%.4f")
    return buf.getvalue().encode("utf-8")


def export_equity_curve_csv(equity_df: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes of the equity curve."""
    if equity_df is None or equity_df.empty:
        return b"No equity curve data.\n"
    out = equity_df.reset_index()
    buf = io.StringIO()
    out.to_csv(buf, index=False, float_format="%.4f")
    return buf.getvalue().encode("utf-8")


def export_monte_carlo_csv(mc_result) -> bytes:
    """Return CSV of MC percentile bands + final equity distribution."""
    if mc_result is None or len(mc_result.pct_50) == 0:
        return b"No Monte Carlo results.\n"
    n = len(mc_result.pct_50)
    df = pd.DataFrame({
        "trading_day": range(n),
        "pct_5":       mc_result.pct_5,
        "pct_25":      mc_result.pct_25,
        "pct_50":      mc_result.pct_50,
        "pct_75":      mc_result.pct_75,
        "pct_95":      mc_result.pct_95,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.2f")
    # Append final-equity stats
    buf.write(f"\nFinal Equity Distribution (all {len(mc_result.final_equity)} paths)\n")
    fe_df = pd.DataFrame({"final_equity": mc_result.final_equity})
    fe_df.to_csv(buf, index=False, float_format="%.2f")
    return buf.getvalue().encode("utf-8")


def export_config_json(config_dict: dict, strategy_name: str = "") -> bytes:
    """Serialise backtest configuration to JSON bytes for reproducibility."""
    snap = {
        "export_timestamp":  datetime.utcnow().isoformat() + "Z",
        "schema_version":    "1.0",
        "strategy_name":     strategy_name,
        "config":            config_dict,
    }
    # Make all values JSON-serialisable
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    return json.dumps(snap, indent=2, default=_clean).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# PDF report
# ─────────────────────────────────────────────────────────────────────────────

def export_pdf_report(
    metrics:       dict,
    trades_df:     pd.DataFrame,
    equity_df:     pd.DataFrame,
    mc_result,
    config_dict:   dict,
    strategy_name: str = "Strategy",
    figures:       Optional[dict] = None,   # {name: plotly_fig}
) -> bytes:
    """
    Build a multi-page PDF report and return as bytes.

    Falls back to a plain-text CSV summary if reportlab is not installed.
    """
    if not REPORTLAB_OK:
        return _fallback_text_report(metrics, config_dict, strategy_name)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        rightMargin=15 * mm, leftMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=f"Backtest Report — {strategy_name}",
    )

    styles = getSampleStyleSheet()
    _MONO  = "Courier"
    _SANS  = "Helvetica"

    def S(name, **kw):
        return ParagraphStyle(name, **{"fontName": _SANS, "textColor": rl_colors.white, **kw})

    def mono(text, size=8):
        return Paragraph(f'<font name="Courier" size="{size}">{text}</font>',
                         S("mono", backColor=rl_colors.black))

    story = []

    # ─── Cover ──────────────────────────────────────────────────────────────
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph(
        f'<font name="Courier" size="22" color="#00FF88">'
        f'QUANTITATIVE BACKTEST REPORT</font>',
        S("title", alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f'<font name="Courier" size="13" color="#888888">{strategy_name}</font>',
        S("sub", alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f'<font name="Courier" size="9" color="#555555">Generated: {ts}</font>',
        S("ts", alignment=TA_CENTER),
    ))
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor("#2A2A2A")))
    story.append(Spacer(1, 6 * mm))

    # ─── KPI summary ────────────────────────────────────────────────────────
    story.append(Paragraph(
        '<font name="Courier" size="10" color="#00FF88">◈ PERFORMANCE SUMMARY</font>',
        S("h"),
    ))
    story.append(Spacer(1, 3 * mm))

    kpi_data = [
        ["METRIC", "VALUE"],
        ["Net P&L",            f"${metrics.get('net_pnl', 0):+,.2f}"],
        ["CAGR",               f"{metrics.get('cagr_pct', 0):.2f}%"],
        ["Win Rate",           f"{metrics.get('win_rate_pct', 0):.1f}%"],
        ["Profit Factor",      f"{metrics.get('profit_factor', 0):.2f}"],
        ["Sharpe Ratio",       f"{metrics.get('sharpe', 0):.2f}"],
        ["Sortino Ratio",      f"{metrics.get('sortino', 0):.2f}"],
        ["Max Drawdown",       f"${metrics.get('max_drawdown_dollars', 0):,.2f}  ({metrics.get('max_drawdown_pct', 0):.2f}%)"],
        ["Calmar Ratio",       f"{metrics.get('calmar', 0):.2f}"],
        ["Total Trades",       f"{int(metrics.get('total_trades', 0)):,}"],
        ["Expected Value",     f"${metrics.get('expected_value', 0):+,.2f}"],
        ["Avg Win",            f"${metrics.get('avg_win', 0):+,.2f}"],
        ["Avg Loss",           f"${metrics.get('avg_loss', 0):+,.2f}"],
        ["SQN",                f"{metrics.get('sqn', 0):.2f}"],
        ["Kelly (half)",       f"{metrics.get('kelly_half_pct', 0):.1f}%"],
        ["Total Commission",   f"${metrics.get('total_commission', 0):,.2f}"],
        ["Total Slippage",     f"${metrics.get('total_slippage', 0):,.2f}"],
    ]

    combine = metrics.get("combine_pass_rate", {})
    if isinstance(combine, dict):
        kpi_data.append(["Combine Pass Rate", f"{combine.get('pass_rate', 0):.1f}%"])

    tbl = Table(kpi_data, colWidths=[120 * mm, 80 * mm])
    tbl.setStyle(_dark_table_style())
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))

    # ─── Charts (if kaleido is available) ───────────────────────────────────
    if figures and KALEIDO_OK:
        story.append(PageBreak())
        story.append(Paragraph(
            '<font name="Courier" size="10" color="#00FF88">◈ CHARTS</font>',
            S("h"),
        ))
        story.append(Spacer(1, 3 * mm))
        for chart_name, fig in figures.items():
            if fig is None:
                continue
            try:
                img_bytes = pio.to_image(fig, format="png", width=1100, height=420,
                                          scale=1.5)
                img_buf = io.BytesIO(img_bytes)
                rli = RLImage(img_buf, width=230 * mm, height=85 * mm)
                story.append(Paragraph(
                    f'<font name="Courier" size="9" color="#888">{chart_name}</font>',
                    S("cap"),
                ))
                story.append(rli)
                story.append(Spacer(1, 4 * mm))
            except Exception:
                story.append(Paragraph(
                    f'<font name="Courier" size="8" color="#555">'
                    f'[Chart "{chart_name}" export failed — kaleido required]</font>',
                    S("warn"),
                ))

    # ─── Annual table ────────────────────────────────────────────────────────
    annual = metrics.get("annual_table", pd.DataFrame())
    if not annual.empty:
        story.append(PageBreak())
        story.append(Paragraph(
            '<font name="Courier" size="10" color="#00FF88">◈ ANNUAL BREAKDOWN</font>',
            S("h"),
        ))
        story.append(Spacer(1, 3 * mm))
        header = [str(c) for c in annual.columns]
        rows   = [[str(v) for v in row] for row in annual.values]
        at     = Table([header] + rows)
        at.setStyle(_dark_table_style())
        story.append(at)

    # ─── Config page ─────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph(
        '<font name="Courier" size="10" color="#00FF88">◈ BACKTEST CONFIGURATION</font>',
        S("h"),
    ))
    story.append(Spacer(1, 3 * mm))
    cfg_rows = [["PARAMETER", "VALUE"]]
    for k, v in config_dict.items():
        cfg_rows.append([str(k), str(v)])
    ct = Table(cfg_rows, colWidths=[120 * mm, 80 * mm])
    ct.setStyle(_dark_table_style())
    story.append(ct)

    # ─── Trade log (first 200 rows) ──────────────────────────────────────────
    if trades_df is not None and not trades_df.empty:
        story.append(PageBreak())
        story.append(Paragraph(
            '<font name="Courier" size="10" color="#00FF88">◈ TRADE LOG (first 200 trades)</font>',
            S("h"),
        ))
        story.append(Spacer(1, 3 * mm))
        cols  = ["entry_time", "signal", "direction", "entry_price",
                 "exit_price", "outcome", "net_pnl"]
        cols  = [c for c in cols if c in trades_df.columns]
        sub   = trades_df[cols].head(200)
        hdr   = [c.upper() for c in cols]
        trows = [[str(v) for v in r] for r in sub.values.tolist()]
        lt    = Table([hdr] + trows)
        lt.setStyle(_dark_table_style(font_size=7))
        story.append(lt)

    # ─── Build PDF ────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage=_dark_page_bg,
        onLaterPages=_dark_page_bg,
    )
    buf.seek(0)
    return buf.read()


# ─── ReportLab helpers ────────────────────────────────────────────────────────

def _dark_table_style(font_size: int = 8) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),   rl_colors.HexColor("#111111")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),   rl_colors.HexColor("#00FF88")),
        ("TEXTCOLOR",    (0, 1), (-1, -1),  rl_colors.HexColor("#E0E0E0")),
        ("BACKGROUND",   (0, 1), (-1, -1),  rl_colors.HexColor("#0A0A0A")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.HexColor("#0A0A0A"),
                                              rl_colors.HexColor("#111111")]),
        ("FONTNAME",     (0, 0), (-1, -1),  "Courier"),
        ("FONTSIZE",     (0, 0), (-1, -1),  font_size),
        ("GRID",         (0, 0), (-1, -1),  0.3, rl_colors.HexColor("#2A2A2A")),
        ("TOPPADDING",   (0, 0), (-1, -1),  3),
        ("BOTTOMPADDING",(0, 0), (-1, -1),  3),
        ("LEFTPADDING",  (0, 0), (-1, -1),  5),
        ("RIGHTPADDING", (0, 0), (-1, -1),  5),
        ("ALIGN",        (0, 0), (-1, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
    ])


def _dark_page_bg(canvas, doc):
    """Paint page background dark black."""
    canvas.saveState()
    canvas.setFillColorRGB(0.039, 0.039, 0.039)  # #0A0A0A
    canvas.rect(0, 0, canvas._pagesize[0], canvas._pagesize[1], fill=1, stroke=0)
    canvas.restoreState()


def _fallback_text_report(metrics: dict, config_dict: dict, strategy_name: str) -> bytes:
    """
    Plain-text fallback when reportlab is not available.
    Returns bytes of a human-readable ASCII report.
    """
    lines = [
        "=" * 70,
        f"  BACKTEST REPORT — {strategy_name}",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
        "",
        "  NOTE: Install 'reportlab' for a full PDF report.",
        "  This is a plain-text fallback.",
        "",
        "─" * 50,
        "  PERFORMANCE SUMMARY",
        "─" * 50,
    ]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            lines.append(f"  {k:<35} {v}")
    lines += ["", "─" * 50, "  CONFIGURATION", "─" * 50]
    for k, v in config_dict.items():
        lines.append(f"  {k:<35} {v}")
    return "\n".join(lines).encode("utf-8")
