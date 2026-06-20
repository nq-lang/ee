"""
ui_components.py — Reusable dark-theme UI building blocks.

All HTML/CSS is injected via st.markdown(unsafe_allow_html=True).
Components return None and render directly into the Streamlit stream.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import streamlit as st

from config import COLORS, APP_VERSION


# ────────────────────────────────────────────────────────────────────────────
# Global CSS injection  (call once at app startup)
# ────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
/* ── Root palette ────────────────────────────────────────────────────── */
:root {
  --bg:          #0A0A0A;
  --bg2:         #111111;
  --card:        #1A1A1A;
  --border:      #2A2A2A;
  --text:        #E0E0E0;
  --dim:         #888888;
  --green:       #00FF88;
  --green-dark:  #003322;
  --red:         #FF3B3B;
  --amber:       #FFD700;
  --blue:        #00BFFF;
  --mono:        'JetBrains Mono','Courier New',monospace;
}

/* ── App background ──────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] {
  background-color: var(--bg2) !important;
  border-right: 1px solid var(--border);
}

/* ── Hide default Streamlit branding ─────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDeployButton"] { display:none; }

/* ── Global text ─────────────────────────────────────────────────────── */
*, p, span, label, div { color: var(--text); font-family: sans-serif; }
code, pre, .stCode, [data-testid="stCode"] {
  font-family: var(--mono) !important;
  background-color: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--green) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 0.82rem !important;
  transition: border-color 0.15s, color 0.15s;
}
.stButton > button:hover {
  border-color: var(--green) !important;
  color: var(--green) !important;
}
.btn-run > button {
  background: var(--green-dark) !important;
  border-color: var(--green) !important;
  color: var(--green) !important;
  font-size: 1rem !important;
  font-weight: bold !important;
  padding: 0.6rem 2rem !important;
}
.btn-stop > button {
  border-color: var(--red) !important;
  color: var(--red) !important;
}

/* ── Input fields ────────────────────────────────────────────────────── */
.stNumberInput input, .stTextInput input,
.stSelectbox > div > div, .stDateInput input {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
}

/* ── Dataframes / tables ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
}
.dataframe { font-family: var(--mono) !important; font-size: 0.80rem !important; }

/* ── Expanders ───────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}

/* ── Progress bar ────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div {
  background-color: var(--green) !important;
}

/* ── Scrollbars ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444; }

/* ── Metric tiles ─────────────────────────────────────────────────────── */
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  text-align: center;
}
.kpi-value {
  font-family: var(--mono);
  font-size: 1.9rem;
  font-weight: 800;
  line-height: 1.1;
}
.kpi-label {
  font-size: 0.70rem;
  color: var(--dim);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: 4px;
}

/* ── Config pill badges ───────────────────────────────────────────────── */
.pill {
  display: inline-block;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.72rem;
  font-family: var(--mono);
  color: var(--dim);
  margin: 2px 3px;
}
.pill-green { border-color: var(--green); color: var(--green); }
.pill-red   { border-color: var(--red);   color: var(--red);   }
.pill-amber { border-color: var(--amber); color: var(--amber); }
.pill-blue  { border-color: var(--blue);  color: var(--blue);  }

/* ── Signal table rows ────────────────────────────────────────────────── */
.sig-win  { color: #00FF88 !important; }
.sig-loss { color: #FF3B3B !important; }
.sig-pend { color: #FFD700 !important; }
.sig-time { color: #FFA500 !important; }
.sig-gap  { color: #00BFFF !important; }

/* ── Section header ───────────────────────────────────────────────────── */
.section-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 6px;
  margin-bottom: 16px;
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--green);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Terminal log box ─────────────────────────────────────────────────── */
.terminal-box {
  background: #050505;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px;
  font-family: var(--mono);
  font-size: 0.78rem;
  color: #00FF88;
  white-space: pre-wrap;
  max-height: 340px;
  overflow-y: auto;
}

/* ── Checklist items ──────────────────────────────────────────────────── */
.check-ok   { color: var(--green); }
.check-fail { color: var(--red);   }

/* ── Sidebar nav items ────────────────────────────────────────────────── */
.nav-active {
  color: var(--green) !important;
  font-weight: bold;
}
</style>
"""


def inject_css():
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Section header
# ────────────────────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = ""):
    html = f'<div class="section-header">◈ {title}</div>'
    if subtitle:
        html += f'<div style="color:#888;font-size:0.78rem;margin-bottom:10px;">{subtitle}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# KPI card row — the top "Final Report" bar
# ────────────────────────────────────────────────────────────────────────────

def kpi_card_row(metrics: dict):
    """
    Render 8 KPI cards in a full-width row.
    metrics must contain the keys listed below.
    """
    combine = metrics.get("combine_pass_rate", {})
    if isinstance(combine, dict):
        pass_rate = combine.get("pass_rate", 0.0)
        fail_dd   = combine.get("fail_dd_rate", 0.0)
    else:
        pass_rate = 0.0
        fail_dd   = 0.0

    cards = [
        ("Combine Pass", f"{pass_rate:.1f}%",              "green"),
        ("Fail DD Rate", f"{fail_dd:.1f}%",                "red"),
        ("Avg Days",     f"{metrics.get('avg_hold_all',0)/60/6.5:.1f}d", "amber"),
        ("Exp / Trade",  f"${metrics.get('expected_value',0):+,.0f}",    "amber"),
        ("Win Rate",     f"{metrics.get('win_rate_pct',0):.1f}%",        "green"),
        ("Profit Factor",f"{metrics.get('profit_factor',0):.2f}",        "blue"),
        ("3YR Sharpe",   f"{metrics.get('sharpe_3yr',0):.2f}",           "blue"),
        ("10YR Sharpe",  f"{metrics.get('sharpe_10yr',0):.2f}",          "blue"),
    ]

    color_map = {
        "green": COLORS["green"],
        "red":   COLORS["red"],
        "amber": COLORS["amber"],
        "blue":  COLORS["blue"],
    }

    cols = st.columns(len(cards))
    for col, (label, value, color) in zip(cols, cards):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-value" style="color:{color_map[color]};">{value}</div>
                  <div class="kpi-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
# Strategy subtitle + config pills
# ────────────────────────────────────────────────────────────────────────────

def strategy_subtitle(
    instrument: str,
    strategy_name: str,
    config_label: str,
    date_range: str,
    pills: list[tuple[str, str]] = None,
):
    """
    Render the line:
      NQ Futures · DEMO_MOM_STRATEGY · Fixed 1 contract · 2020-01-01 → 2024-12-31
    followed by config pill badges.
    """
    subtitle = f"{instrument} Futures · {strategy_name} · {config_label} · {date_range}"
    pill_html = ""
    if pills:
        for text, cls in pills:
            pill_html += f'<span class="pill pill-{cls}">{text}</span>'

    st.markdown(
        f"""<div style="margin:8px 0 4px 0;color:#888;font-size:0.82rem;font-family:monospace;">
          {subtitle}
        </div>
        <div style="margin-bottom:12px;">{pill_html}</div>""",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Sidebar status bar
# ────────────────────────────────────────────────────────────────────────────

def sidebar_status_bar(
    instrument:      str  = "—",
    strategy_file:   str  = "none",
    csv_file:        str  = "none",
    last_run:        str  = "never",
    total_trades:    int  = 0,
    engine_status:   str  = "idle",
):
    status_color = {
        "idle":     "#888",
        "running":  COLORS["amber"],
        "complete": COLORS["green"],
        "error":    COLORS["red"],
    }.get(engine_status, "#888")

    st.sidebar.markdown(
        f"""<div style="border-top:1px solid #2A2A2A;margin-top:16px;padding-top:12px;">
        <div style="font-family:monospace;font-size:0.70rem;color:#555;">
          ─── TERMINAL STATUS ─────────────────<br>
          <span style="color:{status_color};">● {engine_status.upper()}</span><br>
          INSTRUMENT : <span style="color:#DDD;">{instrument}</span><br>
          STRATEGY   : <span style="color:#DDD;">{strategy_file[:24]}</span><br>
          DATA       : <span style="color:#DDD;">{csv_file[:24]}</span><br>
          LAST RUN   : <span style="color:#DDD;">{last_run}</span><br>
          TRADES     : <span style="color:{COLORS['green']};">{total_trades:,}</span><br>
          ─────────────────────────────────────<br>
          <span style="color:#333;">v{APP_VERSION} · quant terminal</span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Checklist row (pre-run validation)
# ────────────────────────────────────────────────────────────────────────────

def checklist(items: list[tuple[bool, str, str]]):
    """
    items : list of (is_ok, label, detail)
    Renders ✅/❌ rows in a styled block.
    """
    lines = []
    all_ok = True
    for ok, label, detail in items:
        icon = "✅" if ok else "❌"
        color = COLORS["green"] if ok else COLORS["red"]
        lines.append(
            f'<div style="font-family:monospace;font-size:0.80rem;'
            f'padding:2px 0;color:{color};">'
            f'{icon} <b>{label}</b>: {detail}</div>'
        )
        if not ok:
            all_ok = False
    st.markdown("".join(lines), unsafe_allow_html=True)
    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# Terminal log box
# ────────────────────────────────────────────────────────────────────────────

def terminal_log(text: str, height: int = 300):
    st.markdown(
        f'<div class="terminal-box" style="max-height:{height}px;">{text}</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Signal history table (styled HTML)
# ────────────────────────────────────────────────────────────────────────────

def signal_table(
    trades_df: pd.DataFrame,
    page: int = 0,
    per_page: int = 50,
    search_signal: str = "",
    search_outcome: str = "ALL",
):
    """
    Render the styled signal history table matching the spec's reference screenshots.
    Columns: TIME | DATE | SIGNAL | Q | ENTRY | STOP | TARGET | REGIME | OUTCOME | P&L
    """
    if trades_df is None or trades_df.empty:
        st.info("No trades to display. Run a backtest first.")
        return

    df = trades_df.copy()

    # Filter
    if search_signal:
        df = df[df["signal"].str.contains(search_signal, case=False, na=False)]
    if search_outcome != "ALL":
        df = df[df["outcome"] == search_outcome]

    total    = len(df)
    start_i  = page * per_page
    end_i    = start_i + per_page
    df_page  = df.iloc[start_i:end_i]

    outcome_color = {
        "WIN":      COLORS["green"],
        "LOSS":     COLORS["red"],
        "PENDING":  COLORS["amber"],
        "TIME":     "#FFA500",
        "EOD":      "#FFA500",
        "GAP":      COLORS["blue"],
        "BLOCKED":  "#555",
    }
    signal_color = lambda s: COLORS["green"] if "LONG" in str(s).upper() or "BUY" in str(s).upper() \
                             else COLORS["red"]

    # Build HTML table
    header = """
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:0.76rem;">
    <thead>
    <tr style="background:#1A1A1A;border-bottom:1px solid #333;">
    """
    for col in ["TIME", "DATE", "SIGNAL", "Q", "ENTRY", "STOP", "TARGET",
                "REGIME", "OUTCOME", "P&amp;L"]:
        header += f'<th style="padding:6px 10px;color:#888;text-align:left;">{col}</th>'
    header += "</tr></thead><tbody>"

    rows_html = ""
    for _, row in df_page.iterrows():
        outcome = str(row.get("outcome", "?"))
        pnl_val = row.get("net_pnl", 0)
        pnl_str = f"${pnl_val:+,.2f}" if pd.notna(pnl_val) else "$0.00"
        pnl_color = COLORS["green"] if pnl_val >= 0 else COLORS["red"]
        sig  = str(row.get("signal", "—"))
        reg  = str(row.get("regime", "—"))

        entry_t = row.get("entry_time")
        time_str = entry_t.strftime("%H:%M:%S") if pd.notna(entry_t) else "—"
        date_str = entry_t.strftime("%Y-%m-%d") if pd.notna(entry_t) else "—"

        bg = "background:#0D0D0D;" if int(row.name) % 2 == 0 else ""

        rows_html += f"""<tr style="{bg}border-bottom:1px solid #1E1E1E;">
          <td style="padding:5px 10px;color:#AAA;">{time_str}</td>
          <td style="padding:5px 10px;color:#888;">{date_str}</td>
          <td style="padding:5px 10px;color:{signal_color(sig)};font-weight:600;">{sig}</td>
          <td style="padding:5px 10px;color:#00BFFF;">{row.get('q_score', 0):.1f}</td>
          <td style="padding:5px 10px;color:#E0E0E0;">{row.get('entry_price', 0):,.2f}</td>
          <td style="padding:5px 10px;color:{COLORS['red']};">{row.get('stop_price', 0):,.2f}</td>
          <td style="padding:5px 10px;color:{COLORS['green']};">{row.get('target_price', 0):,.2f}</td>
          <td style="padding:5px 10px;">
            <span style="background:#1A1A1A;border:1px solid #333;border-radius:10px;
              padding:1px 8px;font-size:0.70rem;color:#888;">{reg}</span>
          </td>
          <td style="padding:5px 10px;color:{outcome_color.get(outcome, '#888')};font-weight:600;">
            {outcome}
          </td>
          <td style="padding:5px 10px;color:{pnl_color};font-weight:700;">{pnl_str}</td>
        </tr>"""

    footer = f"""</tbody>
    <tfoot><tr style="background:#1A1A1A;border-top:1px solid #333;">
      <td colspan="10" style="padding:6px 10px;color:#555;font-size:0.72rem;">
        Showing {start_i+1}–{min(end_i, total)} of {total:,} trades
      </td>
    </tr></tfoot>
    </table></div>"""

    st.markdown(header + rows_html + footer, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Metric grid (2-column label → value)
# ────────────────────────────────────────────────────────────────────────────

def metric_grid(rows: list[tuple[str, str, str]], title: str = ""):
    """
    rows: list of (label, value_str, color_key) e.g. ("Win Rate", "66.4%", "green")
    """
    if title:
        section_header(title)

    color_map = {
        "green":  COLORS["green"],
        "red":    COLORS["red"],
        "amber":  COLORS["amber"],
        "blue":   COLORS["blue"],
        "white":  COLORS["text"],
        "dim":    COLORS["text_dim"],
    }

    html = """<table style="width:100%;font-family:monospace;font-size:0.80rem;
              border-collapse:collapse;">"""
    for i, (label, val, clr) in enumerate(rows):
        bg = "background:#111;" if i % 2 == 0 else "background:#0D0D0D;"
        c  = color_map.get(clr, COLORS["text"])
        html += (
            f'<tr style="{bg}border-bottom:1px solid #1E1E1E;">'
            f'<td style="padding:6px 14px;color:#888;">{label}</td>'
            f'<td style="padding:6px 14px;color:{c};font-weight:600;text-align:right;">{val}</td>'
            f'</tr>'
        )
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Instrument spec display
# ────────────────────────────────────────────────────────────────────────────

def instrument_spec_card(instrument: str, spec: dict):
    st.markdown(
        f"""<div style="background:{COLORS['card_bg']};border:1px solid {COLORS['card_border']};
          border-radius:8px;padding:12px 18px;font-family:monospace;font-size:0.78rem;">
          <div style="color:{COLORS['green']};font-size:0.90rem;font-weight:700;
            margin-bottom:8px;">⬡ {instrument} — {spec['name']}</div>
          <div style="color:#888;">Point Value : <span style="color:#DDD;">${spec['point_value']:.2f}/pt</span></div>
          <div style="color:#888;">Tick Size   : <span style="color:#DDD;">{spec['tick_size']} pts = ${spec['tick_value']:.2f}</span></div>
          <div style="color:#888;">Daily Proxy : <span style="color:{COLORS['amber']};">{spec['yf_daily']}</span>
            &nbsp; Intraday Proxy : <span style="color:{COLORS['amber']};">{spec['yf_proxy']}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Banner helpers
# ────────────────────────────────────────────────────────────────────────────

def banner_success(msg: str):
    st.markdown(
        f'<div style="background:#003322;border:1px solid {COLORS["green"]};border-radius:6px;'
        f'padding:10px 16px;color:{COLORS["green"]};font-family:monospace;'
        f'font-size:0.82rem;">✅ {msg}</div>',
        unsafe_allow_html=True,
    )


def banner_warning(msg: str):
    st.markdown(
        f'<div style="background:#2A2000;border:1px solid {COLORS["amber"]};border-radius:6px;'
        f'padding:10px 16px;color:{COLORS["amber"]};font-family:monospace;'
        f'font-size:0.82rem;">⚠️ {msg}</div>',
        unsafe_allow_html=True,
    )


def banner_error(msg: str):
    st.markdown(
        f'<div style="background:#1A0000;border:1px solid {COLORS["red"]};border-radius:6px;'
        f'padding:10px 16px;color:{COLORS["red"]};font-family:monospace;'
        f'font-size:0.82rem;">❌ {msg}</div>',
        unsafe_allow_html=True,
    )


def banner_info(msg: str):
    st.markdown(
        f'<div style="background:#001A2A;border:1px solid {COLORS["blue"]};border-radius:6px;'
        f'padding:10px 16px;color:{COLORS["blue"]};font-family:monospace;'
        f'font-size:0.82rem;">ℹ️ {msg}</div>',
        unsafe_allow_html=True,
    )
