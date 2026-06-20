"""
monte_carlo.py — Monte Carlo forward-projection simulation engine.

Sampling methods:
  1. Trade-by-Trade Bootstrap   — resample individual trades with replacement
  2. Daily P&L Bootstrap        — resample daily P&L values with replacement
  3. Parametric Normal          — fit N(μ,σ) to trade P&L
  4. Parametric T-Distribution  — fit heavy-tailed t-dist to trade P&L
  5. Block Bootstrap            — resample contiguous blocks to preserve autocorrelation

All simulations support:
  - Walk-forward variance stressors (noise, win-rate degradation, etc.)
  - Combine / prop-firm challenge mode
  - Percentile-band and full-path output for fan charts

Performance target:
  5,000 paths × 252 days < 10 seconds on a standard laptop (numpy vectorised).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    num_paths:        int   = 1_000
    horizon_days:     int   = 252
    sampling_method:  str   = "Trade-by-Trade Bootstrap"  # see METHODS below
    seed:             int   = 42

    # Stressors
    noise_pct:        float = 0.0   # ±% random P&L perturbation per trade
    win_rate_haircut: float = 0.0   # reduce win-rate by this % (0-100)
    stop_size_pct:    float = 0.0   # increase stop losses by this %
    size_reduction:   float = 0.0   # reduce position size by this %
    trade_removal:    float = 0.0   # randomly remove this % of trades

    # Combine mode
    combine_mode:     bool  = False
    daily_loss_limit: float = 3_000.0
    max_drawdown_limit: float = 6_000.0
    profit_target:    float = 12_000.0
    min_trading_days: int   = 5
    max_trading_days: int   = 30

    # Block bootstrap block size
    block_size:       int   = 10

    # Display
    show_individual_paths: bool  = False
    max_display_paths:     int   = 100


METHODS = [
    "Trade-by-Trade Bootstrap",
    "Daily P&L Bootstrap",
    "Parametric (Normal)",
    "Parametric (T-Distribution)",
    "Block Bootstrap",
]


# ────────────────────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class MCResult:
    paths:         np.ndarray = field(default_factory=lambda: np.array([]))
    # shape: (num_paths, horizon_days+1)  — cumulative P&L per day per path

    pct_5:         np.ndarray = field(default_factory=lambda: np.array([]))
    pct_25:        np.ndarray = field(default_factory=lambda: np.array([]))
    pct_50:        np.ndarray = field(default_factory=lambda: np.array([]))
    pct_75:        np.ndarray = field(default_factory=lambda: np.array([]))
    pct_95:        np.ndarray = field(default_factory=lambda: np.array([]))

    final_equity:  np.ndarray = field(default_factory=lambda: np.array([]))

    # Statistics
    median_final:     float = 0.0
    mean_final:       float = 0.0
    p5_final:         float = 0.0
    p95_final:        float = 0.0
    prob_profit:      float = 0.0   # % of paths ending positive
    prob_50k:         float = 0.0   # % of paths ending > $50k
    max_dd_median:    float = 0.0
    max_dd_p95:       float = 0.0
    prob_ruin:        float = 0.0   # % of paths hitting $0 or below
    combine_pass_rate: float = 0.0
    combine_fail_rate: float = 0.0

    config: Optional[MonteCarloConfig] = None


# ────────────────────────────────────────────────────────────────────────────
# Main simulation
# ────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    trades_df: pd.DataFrame,
    config:    MonteCarloConfig,
) -> MCResult:
    """
    Run Monte Carlo simulation on trade P&L history.

    Parameters
    ----------
    trades_df : must contain 'net_pnl' column; optionally 'entry_time'
    config    : simulation parameters

    Returns
    -------
    MCResult with all paths and pre-computed statistics
    """
    if trades_df is None or len(trades_df) == 0:
        return MCResult(config=config)

    rng     = np.random.default_rng(config.seed)
    pnl_raw = trades_df["net_pnl"].values.astype(float)

    # Apply stressors
    pnl = _apply_stressors(pnl_raw, config, rng)

    # Sample builder depending on method
    sampler = _get_sampler(pnl, config, trades_df)

    n_paths  = config.num_paths
    horizon  = config.horizon_days
    paths    = np.zeros((n_paths, horizon + 1))  # day 0 = 0 P&L

    # Determine trades-per-day estimate
    if "entry_time" in trades_df.columns:
        active_days = max(1, trades_df["entry_time"].dt.date.nunique())
        tpd = max(1.0, len(pnl) / active_days)
    else:
        tpd = max(1.0, len(pnl) / max(1, horizon))

    combine_pass = 0
    combine_fail = 0

    for p in range(n_paths):
        daily_pnl = sampler(rng, horizon, tpd)
        paths[p, 1:] = np.cumsum(daily_pnl)

        # Combine mode tracking
        if config.combine_mode:
            result = _eval_combine(
                daily_pnl,
                config.daily_loss_limit,
                config.max_drawdown_limit,
                config.profit_target,
                config.min_trading_days,
                config.max_trading_days,
            )
            if result == "pass":
                combine_pass += 1
            elif result == "fail":
                combine_fail += 1

    # Percentile bands across paths at each time step
    pct_5  = np.percentile(paths, 5,  axis=0)
    pct_25 = np.percentile(paths, 25, axis=0)
    pct_50 = np.percentile(paths, 50, axis=0)
    pct_75 = np.percentile(paths, 75, axis=0)
    pct_95 = np.percentile(paths, 95, axis=0)

    final  = paths[:, -1]

    # Max drawdown per path
    dd_per_path = np.array([_path_max_dd(paths[i]) for i in range(n_paths)])

    return MCResult(
        paths         = paths,
        pct_5         = pct_5,
        pct_25        = pct_25,
        pct_50        = pct_50,
        pct_75        = pct_75,
        pct_95        = pct_95,
        final_equity  = final,
        median_final  = float(np.median(final)),
        mean_final    = float(final.mean()),
        p5_final      = float(np.percentile(final, 5)),
        p95_final     = float(np.percentile(final, 95)),
        prob_profit   = float((final > 0).mean() * 100),
        prob_50k      = float((final > 50_000).mean() * 100),
        max_dd_median = float(np.median(dd_per_path)),
        max_dd_p95    = float(np.percentile(dd_per_path, 95)),
        prob_ruin     = float((final <= 0).mean() * 100),
        combine_pass_rate = combine_pass / n_paths * 100 if config.combine_mode else 0.0,
        combine_fail_rate = combine_fail / n_paths * 100 if config.combine_mode else 0.0,
        config        = config,
    )


# ────────────────────────────────────────────────────────────────────────────
# Samplers
# ────────────────────────────────────────────────────────────────────────────

def _get_sampler(pnl: np.ndarray, config: MonteCarloConfig, trades_df: pd.DataFrame):
    """Return a function(rng, horizon_days, tpd) → daily_pnl array."""
    method = config.sampling_method

    if method == "Trade-by-Trade Bootstrap":
        def sampler(rng, h, tpd):
            n_trades = int(round(h * tpd))
            sample   = rng.choice(pnl, size=n_trades, replace=True)
            # Aggregate into daily buckets
            splits = np.array_split(sample, h)
            return np.array([s.sum() for s in splits])

    elif method == "Daily P&L Bootstrap":
        daily = _to_daily_pnl(pnl, trades_df)
        def sampler(rng, h, tpd):
            return rng.choice(daily, size=h, replace=True)

    elif method == "Parametric (Normal)":
        mu, sigma = pnl.mean(), pnl.std()
        tpd_ref   = config.horizon_days  # scale to trades per day later
        def sampler(rng, h, tpd):
            n_trades  = int(round(h * tpd))
            trade_pnl = rng.normal(mu, sigma, n_trades)
            splits    = np.array_split(trade_pnl, h)
            return np.array([s.sum() for s in splits])

    elif method == "Parametric (T-Distribution)":
        df_t, loc_t, scale_t = stats.t.fit(pnl)
        def sampler(rng, h, tpd):
            n_trades  = int(round(h * tpd))
            trade_pnl = stats.t.rvs(df_t, loc_t, scale_t, size=n_trades,
                                     random_state=int(rng.integers(0, 2**31)))
            splits    = np.array_split(trade_pnl, h)
            return np.array([s.sum() for s in splits])

    elif method == "Block Bootstrap":
        bs = config.block_size
        def sampler(rng, h, tpd):
            n_trades = int(round(h * tpd))
            n_blocks = max(1, n_trades // bs)
            max_start = max(1, len(pnl) - bs)
            starts   = rng.integers(0, max_start, size=n_blocks)
            blocks   = [pnl[s:s+bs] for s in starts]
            sample   = np.concatenate(blocks)[:n_trades]
            splits   = np.array_split(sample, h)
            return np.array([s.sum() for s in splits])

    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return sampler


def _to_daily_pnl(pnl: np.ndarray, trades_df: pd.DataFrame) -> np.ndarray:
    """Aggregate trade P&L into daily buckets."""
    if "entry_time" not in trades_df.columns:
        # Rough aggregation: average 5 trades/day
        splits = np.array_split(pnl, max(1, len(pnl) // 5))
        return np.array([s.sum() for s in splits])
    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["entry_time"]).dt.date
    return df.groupby("date")["net_pnl"].sum().values


# ────────────────────────────────────────────────────────────────────────────
# Stressors
# ────────────────────────────────────────────────────────────────────────────

def _apply_stressors(pnl: np.ndarray, config: MonteCarloConfig,
                     rng: np.random.Generator) -> np.ndarray:
    out = pnl.copy()

    if config.noise_pct > 0:
        noise = rng.uniform(-config.noise_pct / 100, config.noise_pct / 100, len(out))
        out   = out * (1 + noise)

    if config.win_rate_haircut > 0:
        # Flip a fraction of wins to losses
        win_idx = np.where(out > 0)[0]
        n_flip  = int(len(win_idx) * config.win_rate_haircut / 100)
        if n_flip > 0:
            flip_idx = rng.choice(win_idx, size=n_flip, replace=False)
            out[flip_idx] = -out[flip_idx]

    if config.stop_size_pct > 0:
        # Increase losses by the stop_size_pct
        loss_idx = np.where(out < 0)[0]
        out[loss_idx] = out[loss_idx] * (1 + config.stop_size_pct / 100)

    if config.size_reduction > 0:
        out = out * (1 - config.size_reduction / 100)

    if config.trade_removal > 0:
        n_remove = int(len(out) * config.trade_removal / 100)
        if n_remove > 0:
            remove_idx = rng.choice(len(out), size=n_remove, replace=False)
            out = np.delete(out, remove_idx)

    return out


# ────────────────────────────────────────────────────────────────────────────
# Combine evaluation
# ────────────────────────────────────────────────────────────────────────────

def _eval_combine(
    daily_pnl: np.ndarray,
    daily_loss_limit: float,
    max_drawdown_limit: float,
    profit_target: float,
    min_days: int,
    max_days: int,
) -> Literal["pass", "fail", "timeout"]:
    cumulative = 0.0
    peak       = 0.0
    for day_idx, dpnl in enumerate(daily_pnl):
        cumulative += dpnl
        peak        = max(peak, cumulative)

        if dpnl < -daily_loss_limit:
            return "fail"
        if (peak - cumulative) >= max_drawdown_limit:
            return "fail"
        if cumulative >= profit_target and (day_idx + 1) >= min_days:
            return "pass"
        if (day_idx + 1) >= max_days:
            # Ran out of days
            return "fail" if cumulative < profit_target else "pass"

    return "timeout"


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def _path_max_dd(path: np.ndarray) -> float:
    """Maximum drawdown of a single cumulative P&L path."""
    running_max = np.maximum.accumulate(path)
    dd = running_max - path
    return float(dd.max())
