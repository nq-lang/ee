"""
config.py — Global constants and instrument specifications.
All P&L, tick, and margin parameters for NQ/ES/MNQ/MES futures.
"""

APP_VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = "1.0"

# ─── Instrument Specifications ──────────────────────────────────────────────
INSTRUMENTS = {
    "NQ": {
        "name":        "E-mini NASDAQ-100",
        "point_value": 20.0,      # $ per full point
        "tick_size":   0.25,      # minimum price increment
        "tick_value":  5.0,       # $ per tick  (0.25 × $20)
        "yf_daily":    "NQ=F",    # yfinance continuous contract
        "yf_proxy":    "QQQ",     # intraday proxy
        "price_low":   8_000,     # auto-detect range
        "price_high":  25_000,
        "micro":       "MNQ",
    },
    "ES": {
        "name":        "E-mini S&P 500",
        "point_value": 50.0,
        "tick_size":   0.25,
        "tick_value":  12.50,
        "yf_daily":    "ES=F",
        "yf_proxy":    "SPY",
        "price_low":   1_800,
        "price_high":  7_000,
        "micro":       "MES",
    },
    "MNQ": {
        "name":        "Micro E-mini NASDAQ-100",
        "point_value": 2.0,
        "tick_size":   0.25,
        "tick_value":  0.50,
        "yf_daily":    "NQ=F",
        "yf_proxy":    "QQQ",
        "price_low":   8_000,
        "price_high":  25_000,
        "micro":       None,
    },
    "MES": {
        "name":        "Micro E-mini S&P 500",
        "point_value": 5.0,
        "tick_size":   0.25,
        "tick_value":  0.25,
        "yf_daily":    "ES=F",
        "yf_proxy":    "SPY",
        "price_low":   1_800,
        "price_high":  7_000,
        "micro":       None,
    },
}

# ─── Options Underlyings ─────────────────────────────────────────────────────
OPTIONS_UNDERLYINGS = ["SPY", "QQQ"]

# ─── Default Backtest Configuration ─────────────────────────────────────────
DEFAULT_CONFIG = {
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
    "max_bars_in_trade":    0,       # 0 = disabled
    "eod_exit":             True,
    "eod_exit_time":        "15:45",
    "session_filter":       "Full Session",
    "day_of_week_filter":   [0, 1, 2, 3, 4],  # Mon-Fri
    "date_range_mode":      "Full CSV History",
}

# ─── Combine / Prop Firm Defaults ────────────────────────────────────────────
DEFAULT_COMBINE = {
    "daily_loss_limit":     3_000.0,
    "max_drawdown_limit":   6_000.0,
    "profit_target":        12_000.0,
    "min_trading_days":     5,
    "max_trading_days":     30,
}

# ─── Monte Carlo Defaults ────────────────────────────────────────────────────
DEFAULT_MONTE_CARLO = {
    "num_paths":        1_000,
    "horizon_days":     252,
    "sampling_method":  "Trade-by-Trade Bootstrap",
}

# ─── Dark Theme Color Palette ─────────────────────────────────────────────────
COLORS = {
    "bg":           "#0A0A0A",
    "bg_secondary": "#111111",
    "card_bg":      "#1A1A1A",
    "card_border":  "#2A2A2A",
    "text":         "#E0E0E0",
    "text_dim":     "#888888",
    "green":        "#00FF88",
    "green_dark":   "#003322",
    "green_mid":    "#00AA44",
    "red":          "#FF3B3B",
    "red_dark":     "#330011",
    "amber":        "#FFD700",
    "blue":         "#00BFFF",
    "white":        "#FFFFFF",
    "grid":         "#1E1E1E",
    "axis":         "#333333",
}

# ─── Plotly Dark Template ─────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor="#0F0F0F",
    font=dict(color=COLORS["text"], family="'JetBrains Mono','Courier New',monospace"),
    xaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], zerolinecolor=COLORS["axis"]),
    yaxis=dict(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], zerolinecolor=COLORS["axis"]),
    legend=dict(bgcolor=COLORS["card_bg"], bordercolor=COLORS["card_border"]),
    hoverlabel=dict(bgcolor=COLORS["card_bg"], bordercolor=COLORS["card_border"],
                    font=dict(color=COLORS["text"])),
)

# ─── Bar-interval detection ──────────────────────────────────────────────────
INTERVAL_MINUTES = {
    "1m":  1, "5m":  5, "15m": 15, "30m": 30,
    "1h":  60, "4h": 240, "1d": 1440,
}

# ─── Supported C++ strategy signal names (parser seed list) ──────────────────
KNOWN_SIGNAL_NAMES = [
    "MOM_LONG", "MOM_SHORT", "ABSORPTION_SHORT", "ABSORPTION_LONG",
    "REVERSAL_LONG", "REVERSAL_SHORT", "BREAKOUT_LONG", "BREAKOUT_SHORT",
    "MEAN_REVERT_LONG", "MEAN_REVERT_SHORT", "TREND_LONG", "TREND_SHORT",
]

KNOWN_REGIME_NAMES = [
    "ROTATIONAL", "TRENDING", "MEAN_REVERT", "VOLATILE",
    "BULL", "BEAR", "NEUTRAL", "CHOPPY",
]
