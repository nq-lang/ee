"""
volatility_regime.py
═════════════════════
Volatility regime classification and profitability analytics.

Regimes
───────
TRENDING     — directional price movement, ATR expanding, slope positive/negative
ROTATIONAL   — price oscillating around VWAP, moderate ATR
MEAN_REVERT  — low volatility, price snapping back to mean
VOLATILE     — spike / event-driven, ATR >> historical average
CHOPPY       — tight range, low ATR, no clear direction

Analytics surfaces
──────────────────
- Per-regime win rate, avg P&L, total P&L, trade count
- Most / least profitable regime
- Regime duration analysis
- Regime-conditional edge (EV × regime frequency)
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


REGIMES = ["TRENDING", "ROTATIONAL", "MEAN_REVERT", "VOLATILE", "CHOPPY"]

REGIME_COLORS = {
    "TRENDING":    "#00FF88",
    "ROTATIONAL":  "#00BFFF",
    "MEAN_REVERT": "#FFD700",
    "VOLATILE":    "#FF3B3B",
    "CHOPPY":      "#888888",
}


# ══════════════════════════════════════════════════════════════════════════════
# Bar-level regime classifier
# ══════════════════════════════════════════════════════════════════════════════

def classify_regimes(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Classify each bar into a volatility regime.

    Parameters
    ----------
    df     : OHLCV DataFrame with DatetimeIndex
    window : lookback window for rolling stats (default 20 bars)

    Returns
    -------
    pd.Series of regime strings, same index as df
    """
    if df.empty or "close" not in df.columns:
        return pd.Series([], dtype=str)

    c  = df["close"].values.astype(float)
    h  = df["high"].values.astype(float)  if "high"   in df.columns else c
    l  = df["low"].values.astype(float)   if "low"    in df.columns else c
    v  = df["volume"].values.astype(float)if "volume" in df.columns else np.ones(len(c))
    n  = len(c)

    # ATR (Wilder)
    tr  = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c,1)), np.abs(l - np.roll(c,1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=window, adjust=False).mean().values

    # Rolling ATR percentile rank
    atr_ser   = pd.Series(atr)
    atr_rank  = atr_ser.rolling(window * 5, min_periods=window).rank(pct=True).fillna(0.5).values

    # VWAP
    typ      = (h + l + c) / 3
    cum_pv   = np.cumsum(typ * np.where(v > 0, v, 1))
    cum_v    = np.cumsum(np.where(v > 0, v, 1))
    vwap     = cum_pv / cum_v

    # Price relative to VWAP
    vwap_dev = np.abs(c - vwap) / np.where(vwap > 0, vwap, 1)

    # Rolling slope of close (linear trend strength)
    slope = pd.Series(c).rolling(window).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x[-1] if len(x) == window else 0,
        raw=True,
    ).fillna(0).values

    regimes = []
    for i in range(n):
        ar = float(atr_rank[i])
        vd = float(vwap_dev[i])
        sl = float(abs(slope[i]))

        if ar > 0.85:
            # Top 15% ATR — spike / volatile event
            regime = "VOLATILE"
        elif ar > 0.65 and sl > 0.0003:
            # Above-average ATR with directional slope
            regime = "TRENDING"
        elif ar > 0.40 and vd > 0.002:
            # Moderate ATR, price away from VWAP oscillating
            regime = "ROTATIONAL"
        elif ar < 0.25:
            # Low ATR, tight range
            regime = "CHOPPY"
        else:
            # Price mean-reverting around VWAP
            regime = "MEAN_REVERT"

        regimes.append(regime)

    return pd.Series(regimes, index=df.index, name="regime")


# ══════════════════════════════════════════════════════════════════════════════
# Trade-level regime tagging
# ══════════════════════════════════════════════════════════════════════════════

def tag_trades_with_regime(
    trades:  pd.DataFrame,
    ohlcv:   pd.DataFrame,
    window:  int = 20,
) -> pd.DataFrame:
    """
    For each trade, look up the volatility regime at entry time and attach it.

    Parameters
    ----------
    trades : trade log DataFrame (must have 'entry_time' column)
    ohlcv  : OHLCV DataFrame (same instrument, compatible timestamps)
    window : regime classification window

    Returns
    -------
    trades DataFrame with 'regime' column added / updated
    """
    if trades.empty or ohlcv.empty:
        return trades

    regime_series = classify_regimes(ohlcv, window)

    def _lookup(entry_time):
        try:
            ts = pd.Timestamp(entry_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            # Find nearest bar at or before entry
            idx = regime_series.index.searchsorted(ts, side="right")
            idx = max(0, idx - 1)
            return regime_series.iloc[idx]
        except Exception:
            return "UNKNOWN"

    out = trades.copy()
    if "entry_time" in out.columns:
        out["regime"] = out["entry_time"].apply(_lookup)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Regime analytics
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_analytics(trades: pd.DataFrame) -> dict:
    """
    Compute per-regime profitability stats and rank regimes.

    Returns
    -------
    dict with keys:
      breakdown    : DataFrame (regime × stats)
      best_regime  : str
      worst_regime : str
      regime_ev    : dict {regime: expected_value}
      session_pnl  : DataFrame (date × pnl)
    """
    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        return {
            "breakdown":    pd.DataFrame(),
            "best_regime":  "N/A",
            "worst_regime": "N/A",
            "regime_ev":    {},
            "session_pnl":  pd.DataFrame(),
        }

    df = trades.copy()
    reg_col = df.get("regime", pd.Series(["UNKNOWN"] * len(df)))
    if "regime" not in df.columns:
        df["regime"] = "UNKNOWN"

    rows = []
    for regime in df["regime"].unique():
        sub   = df[df["regime"] == regime]
        n     = len(sub)
        wins  = (sub["net_pnl"] > 0).sum()
        total = sub["net_pnl"].sum()
        avg   = sub["net_pnl"].mean()
        wr    = wins / n if n > 0 else 0.0
        pf_g  = sub.loc[sub["net_pnl"] > 0, "net_pnl"].sum()
        pf_l  = abs(sub.loc[sub["net_pnl"] < 0, "net_pnl"].sum())
        pf    = pf_g / pf_l if pf_l > 0 else float("inf")

        rows.append({
            "REGIME":        regime,
            "TRADES":        n,
            "WIN RATE":      f"{wr*100:.1f}%",
            "AVG P&L":       round(avg, 2),
            "TOTAL P&L":     round(total, 2),
            "PROFIT FACTOR": round(pf, 2),
            "SHARE %":       round(n / len(df) * 100, 1),
        })

    breakdown = pd.DataFrame(rows).sort_values("TOTAL P&L", ascending=False)

    regime_ev = {
        r["REGIME"]: r["AVG P&L"]
        for r in rows
    }

    best  = breakdown.iloc[0]["REGIME"]  if not breakdown.empty else "N/A"
    worst = breakdown.iloc[-1]["REGIME"] if not breakdown.empty else "N/A"

    # Session P&L (daily)
    session_pnl = pd.DataFrame()
    if "entry_time" in df.columns:
        df["date"] = pd.to_datetime(df["entry_time"]).dt.date
        sp = df.groupby("date").agg(
            session_pnl=("net_pnl", "sum"),
            trades=("net_pnl", "count"),
            wins=("net_pnl", lambda x: (x > 0).sum()),
        ).reset_index()
        sp["win_rate"]    = sp["wins"] / sp["trades"]
        sp["cumulative"]  = sp["session_pnl"].cumsum()
        session_pnl       = sp

    return {
        "breakdown":    breakdown,
        "best_regime":  best,
        "worst_regime": worst,
        "regime_ev":    regime_ev,
        "session_pnl":  session_pnl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo regime-conditional simulation
# ══════════════════════════════════════════════════════════════════════════════

def regime_conditional_paths(
    trades:     pd.DataFrame,
    regime:     str,
    n_paths:    int = 1_000,
    horizon:    int = 252,
    seed:       int = 42,
) -> np.ndarray:
    """
    Run bootstrap simulation using only trades from a specific regime.
    Returns array of shape (n_paths, horizon+1).
    """
    if "regime" not in trades.columns or trades.empty:
        return np.zeros((n_paths, horizon + 1))

    sub = trades[trades["regime"] == regime]["net_pnl"].values
    if len(sub) < 5:
        return np.zeros((n_paths, horizon + 1))

    rng   = np.random.default_rng(seed)
    tpd   = max(1.0, len(sub) / max(horizon // 4, 1))
    paths = np.zeros((n_paths, horizon + 1))

    for i in range(n_paths):
        n_trades = int(round(horizon * tpd))
        sample   = rng.choice(sub, size=n_trades, replace=True)
        daily    = np.array_split(sample, horizon)
        cumsum   = 0.0
        paths[i, 0] = 0.0
        for d, block in enumerate(daily):
            cumsum      += block.sum()
            paths[i, d+1] = cumsum

    return paths
