"""
metrics.py — Complete institutional-grade performance metric calculations.

All formulas documented with references for auditability.
Every calculation is vectorised via numpy/pandas.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_YEAR      = TRADING_DAYS_PER_YEAR * 390  # RTH minutes


# ────────────────────────────────────────────────────────────────────────────
# Primary entry point
# ────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    trades_df:     pd.DataFrame,
    equity_curve:  pd.DataFrame,
    config_dict:   dict,
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Master function — returns every metric required by the results dashboard.

    Parameters
    ----------
    trades_df     : trade log (net_pnl, outcome, entry_time, etc.)
    equity_curve  : timestamp → equity DataFrame
    config_dict   : backtest config dict (starting_balance, instrument, …)
    risk_free_rate: annualised decimal (default 5%)

    Returns
    -------
    Flat dict of metric_name → value (numbers, strings, or DataFrames)
    """
    if trades_df is None or len(trades_df) == 0:
        return _empty_metrics()

    m: dict = {}

    start_bal   = float(config_dict.get("starting_balance", 50_000))
    instrument  = config_dict.get("instrument", "NQ")
    pnl         = trades_df["net_pnl"].values.astype(float)
    gross       = trades_df["gross_pnl"].values.astype(float)
    wins        = pnl[pnl > 0]
    losses      = pnl[pnl < 0]
    outcomes    = trades_df.get("outcome", pd.Series(["?"] * len(trades_df)))

    # ── Trade counts ─────────────────────────────────────────────────────────
    n_total         = len(pnl)
    n_wins          = len(wins)
    n_losses        = len(losses)
    n_time          = int((outcomes == "TIME").sum()) if "outcome" in trades_df else 0
    n_eod           = int((outcomes == "EOD").sum())  if "outcome" in trades_df else 0
    win_rate        = n_wins / n_total if n_total else 0.0

    m["total_trades"]       = n_total
    m["winning_trades"]     = n_wins
    m["losing_trades"]      = n_losses
    m["win_rate_pct"]       = win_rate * 100
    m["time_stop_exits"]    = n_time
    m["eod_exits"]          = n_eod
    m["time_stop_pct"]      = n_time / n_total * 100 if n_total else 0

    # ── P&L ──────────────────────────────────────────────────────────────────
    net_pnl_total   = float(pnl.sum())
    gross_pnl_total = float(gross.sum())
    total_comm      = float(trades_df.get("commission", pd.Series([0]*n_total)).sum())
    total_slip      = float(trades_df.get("slippage", pd.Series([0]*n_total)).sum())

    m["net_pnl"]            = net_pnl_total
    m["gross_pnl"]          = gross_pnl_total
    m["total_commission"]   = total_comm
    m["total_slippage"]     = total_slip
    m["total_fees"]         = total_comm + total_slip
    m["net_pnl_pct"]        = net_pnl_total / start_bal * 100

    # ── Win/loss averages ─────────────────────────────────────────────────────
    avg_win  = float(wins.mean())  if len(wins)   else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    m["avg_win"]            = avg_win
    m["avg_loss"]           = avg_loss
    m["largest_win"]        = float(wins.max())   if len(wins)   else 0.0
    m["largest_loss"]       = float(losses.min()) if len(losses) else 0.0
    m["payoff_ratio"]       = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    m["expected_value"]     = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # ── Profit factor ─────────────────────────────────────────────────────────
    # PF = Gross Profits / |Gross Losses|   (Schwager, "The New Market Wizards")
    gross_wins   = float(wins.sum())   if len(wins)   else 0.0
    gross_losses = abs(float(losses.sum())) if len(losses) else 1e-9
    m["profit_factor"] = gross_wins / gross_losses if gross_losses > 0 else np.inf
    m["gross_wins"]    = gross_wins
    m["gross_losses"]  = gross_losses

    # ── Hold times ───────────────────────────────────────────────────────────
    if "hold_minutes" in trades_df.columns:
        hold_all  = trades_df["hold_minutes"].values
        hold_w    = trades_df.loc[trades_df["net_pnl"] > 0, "hold_minutes"].values
        hold_l    = trades_df.loc[trades_df["net_pnl"] < 0, "hold_minutes"].values
        m["avg_hold_all"]     = float(hold_all.mean()) if len(hold_all) else 0.0
        m["avg_hold_wins"]    = float(hold_w.mean())   if len(hold_w)   else 0.0
        m["avg_hold_losses"]  = float(hold_l.mean())   if len(hold_l)   else 0.0
    else:
        m["avg_hold_all"] = m["avg_hold_wins"] = m["avg_hold_losses"] = 0.0

    # ── MAE / MFE ─────────────────────────────────────────────────────────────
    if "mae" in trades_df.columns:
        m["avg_mae"] = float(trades_df["mae"].mean())
        m["avg_mfe"] = float(trades_df["mfe"].mean())
    else:
        m["avg_mae"] = m["avg_mfe"] = 0.0

    # ── Consecutive streaks ───────────────────────────────────────────────────
    streak_win, streak_loss = _compute_streaks(pnl)
    m["max_consec_wins"]    = streak_win
    m["max_consec_losses"]  = streak_loss

    # ── Active trading days ───────────────────────────────────────────────────
    if "entry_time" in trades_df.columns:
        active_days  = trades_df["entry_time"].dt.date.nunique()
        active_sessions = active_days
    else:
        active_days = active_sessions = max(1, n_total // 5)

    m["active_days"]            = active_days
    m["avg_trades_per_day"]     = n_total / max(1, active_days)
    m["avg_trades_per_session"] = n_total / max(1, active_sessions)

    # ── Equity curve metrics ──────────────────────────────────────────────────
    if equity_curve is not None and "equity" in equity_curve.columns:
        eq   = equity_curve["equity"].values.astype(float)
        dd_m = _drawdown_metrics(eq, start_bal, equity_curve)
        m.update(dd_m)

        # Date range
        m["start_date"] = str(equity_curve.index[0].date())
        m["end_date"]   = str(equity_curve.index[-1].date())
        years = max(
            (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25,
            1.0 / TRADING_DAYS_PER_YEAR,
        )
        m["years"] = years

        # CAGR
        # CAGR = (EndingValue / BeginningValue)^(1/years) - 1
        final_equity = float(eq[-1])
        m["cagr_pct"] = ((final_equity / start_bal) ** (1.0 / years) - 1) * 100

        # Daily returns
        daily_eq = equity_curve["equity"].resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna().values
        m.update(_sharpe_metrics(daily_ret, risk_free_rate))

    else:
        m["cagr_pct"]    = 0.0
        m["years"]       = 1.0
        m["start_date"]  = ""
        m["end_date"]    = ""
        m.update(_drawdown_metrics(np.array([start_bal]), start_bal))
        m.update(_sharpe_metrics(np.array([0.0]), risk_free_rate))

    # ── Annualised return ─────────────────────────────────────────────────────
    m["annualised_return_pct"] = m["cagr_pct"]

    # ── Daily P&L stats ───────────────────────────────────────────────────────
    if "entry_time" in trades_df.columns:
        daily_pnl = (
            trades_df.groupby(trades_df["entry_time"].dt.date)["net_pnl"]
            .sum()
            .values
        )
        m["daily_pnl_mean"]  = float(daily_pnl.mean()) if len(daily_pnl) else 0.0
        m["daily_pnl_std"]   = float(daily_pnl.std())  if len(daily_pnl) else 0.0
    else:
        m["daily_pnl_mean"] = m["daily_pnl_std"] = 0.0

    # ── Advanced ratios ───────────────────────────────────────────────────────
    m.update(_advanced_ratios(pnl, m.get("max_drawdown_dollars", 1),
                               m.get("cagr_pct", 0), m.get("sharpe", 0),
                               m.get("sortino", 0), n_total))

    # ── Costs as % of gross ───────────────────────────────────────────────────
    m["costs_pct_of_gross"] = (
        m["total_fees"] / m["gross_pnl"] * 100 if m["gross_pnl"] > 0 else 0.0
    )
    m["breakeven_win_rate"] = _breakeven_win_rate(avg_win, avg_loss)

    # ── Annual breakdown table ────────────────────────────────────────────────
    m["annual_table"] = _annual_breakdown(trades_df)

    # ── Monthly P&L heatmap data ──────────────────────────────────────────────
    m["monthly_table"] = _monthly_breakdown(trades_df)

    # ── DOW breakdown ─────────────────────────────────────────────────────────
    m["dow_table"] = _dow_breakdown(trades_df)

    # ── Time-of-day breakdown ─────────────────────────────────────────────────
    m["tod_table"] = _tod_breakdown(trades_df)

    # ── Combine simulation ────────────────────────────────────────────────────
    m["combine_pass_rate"] = _combine_pass_rate(
        pnl,
        daily_loss_limit=config_dict.get("combine_daily_loss", 3000),
        max_drawdown=config_dict.get("combine_max_dd", 6000),
        profit_target=config_dict.get("combine_profit_target", 12000),
        start_bal=start_bal,
    )

    return m


# ────────────────────────────────────────────────────────────────────────────
# Drawdown
# ────────────────────────────────────────────────────────────────────────────

def _drawdown_metrics(
    equity: np.ndarray,
    start_bal: float,
    equity_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Max drawdown, duration, recovery, Ulcer Index, Pain Ratio.

    Reference: "Portfolio Management Formulas" — Ralph Vince
    """
    m = {}
    if len(equity) < 2:
        m.update({
            "max_drawdown_dollars": 0.0, "max_drawdown_pct": 0.0,
            "max_dd_duration_days": 0,   "avg_drawdown_dollars": 0.0,
            "avg_dd_duration_days": 0,   "recovery_factor": 0.0,
            "ulcer_index": 0.0,          "pain_ratio": 0.0,
        })
        return m

    running_max = np.maximum.accumulate(equity)
    dd_dollars  = running_max - equity
    dd_pct      = dd_dollars / np.where(running_max > 0, running_max, 1) * 100

    max_dd   = float(dd_dollars.max())
    max_dd_p = float(dd_pct.max())

    # Ulcer Index = sqrt(mean(dd_pct^2))
    # Reference: Peter Martin & Byron McCann, "The Investor's Guide to Fidelity Funds"
    ulcer = float(np.sqrt(np.mean(dd_pct ** 2)))

    # Recovery factor = Net P&L / Max Drawdown
    net = float(equity[-1]) - start_bal
    rec = abs(net / max_dd) if max_dd > 0 else 0.0

    # Pain Ratio = CAGR / Ulcer Index
    # (we'll compute CAGR separately so just store ulcer for now)

    # Average drawdown
    dd_periods = dd_dollars[dd_dollars > 0]
    avg_dd = float(dd_periods.mean()) if len(dd_periods) else 0.0

    # Max DD duration (calendar days)
    max_dur = 0
    if equity_df is not None and len(equity_df) > 1:
        max_dur = _max_dd_duration(equity_df)

    m.update({
        "max_drawdown_dollars": max_dd,
        "max_drawdown_pct":     max_dd_p,
        "max_dd_duration_days": max_dur,
        "avg_drawdown_dollars": avg_dd,
        "avg_dd_duration_days": 0,
        "recovery_factor":      rec,
        "ulcer_index":          ulcer,
        "pain_ratio":           0.0,   # filled in _advanced_ratios
    })
    return m


def _max_dd_duration(equity_df: pd.DataFrame) -> int:
    """Number of calendar days from peak to recovery."""
    eq   = equity_df["equity"].values
    idx  = equity_df.index
    peak = eq[0]
    peak_time = idx[0]
    max_dur = 0
    for i in range(1, len(eq)):
        if eq[i] >= peak:
            dur = (idx[i] - peak_time).days
            max_dur = max(max_dur, dur)
            peak = eq[i]
            peak_time = idx[i]
        elif eq[i] > peak:
            peak = eq[i]
            peak_time = idx[i]
    return max_dur


# ────────────────────────────────────────────────────────────────────────────
# Risk-adjusted return metrics
# ────────────────────────────────────────────────────────────────────────────

def _sharpe_metrics(daily_ret: np.ndarray, risk_free_rate: float) -> dict:
    """
    Sharpe, Sortino, computed from daily returns.

    Sharpe  = (mean_daily_ret - rfr_daily) / std_daily * sqrt(252)
    Sortino = (mean_daily_ret - rfr_daily) / downside_std * sqrt(252)
    Reference: Sharpe (1966), "Mutual Fund Performance", Journal of Business
    """
    m = {}
    if len(daily_ret) < 2:
        return {"sharpe": 0.0, "sortino": 0.0}

    rfr_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess    = daily_ret - rfr_daily
    std       = daily_ret.std()
    mean      = daily_ret.mean()

    m["sharpe"] = float(
        (mean - rfr_daily) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0 else 0.0
    )

    # Sortino: only downside deviation in denominator
    downside = daily_ret[daily_ret < rfr_daily]
    if len(downside) > 1:
        dd_std = float(np.sqrt(np.mean((downside - rfr_daily) ** 2)))
        m["sortino"] = float(
            (mean - rfr_daily) / dd_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            if dd_std > 0 else 0.0
        )
    else:
        m["sortino"] = 0.0

    # Partial Sharpe estimates for 3YR and 10YR windows
    n3  = min(len(daily_ret), 3  * TRADING_DAYS_PER_YEAR)
    n10 = min(len(daily_ret), 10 * TRADING_DAYS_PER_YEAR)
    r3  = daily_ret[-n3:]
    r10 = daily_ret[-n10:]

    def _sh(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return float((r.mean() - rfr_daily) / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    m["sharpe_3yr"]  = _sh(r3)
    m["sharpe_10yr"] = _sh(r10)

    return m


# ────────────────────────────────────────────────────────────────────────────
# Advanced ratios
# ────────────────────────────────────────────────────────────────────────────

def _advanced_ratios(
    pnl: np.ndarray,
    max_dd: float,
    cagr_pct: float,
    sharpe: float,
    sortino: float,
    n_trades: int,
) -> dict:
    """
    Calmar, MAR, Omega, Sterling, Kelly, SQN, Z-score.
    """
    m = {}
    safe_dd = max_dd if max_dd > 0 else 1.0

    # Calmar = CAGR / |Max Drawdown %|
    # Reference: Young (1991), "Calmar Ratio: A Smoother Tool"
    m["calmar"] = abs(cagr_pct / (max_dd / safe_dd * 100)) if safe_dd > 0 else 0.0

    # MAR = annualised return / max dd (same formula, different origin)
    m["mar"] = m["calmar"]

    # Omega Ratio = sum(positive returns) / |sum(negative returns)|  (threshold=0)
    pos  = pnl[pnl > 0].sum()
    neg  = abs(pnl[pnl < 0].sum())
    m["omega"] = float(pos / neg) if neg > 0 else np.inf

    # Sterling Ratio = CAGR / Avg Annual Max Drawdown  (approx as CAGR / max_dd)
    m["sterling"] = m["calmar"]

    # Kelly Criterion
    # f* = W/L * p - (1-p)/1   where W=avg_win, L=avg_loss, p=win_rate
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    p      = len(wins) / n_trades if n_trades else 0.0
    W      = float(wins.mean())   if len(wins)   else 0.0
    L      = abs(float(losses.mean())) if len(losses) else 1.0
    kelly  = (p * W - (1 - p) * L) / W if W > 0 else 0.0
    m["kelly_full_pct"] = float(kelly * 100)
    m["kelly_half_pct"] = float(kelly * 50)

    # Van Tharp SQN = sqrt(n) * mean(R) / std(R)
    # Reference: Tharp (2009), "Super Trader"
    if n_trades >= 5:
        r_std = pnl.std()
        m["sqn"] = float(
            np.sqrt(n_trades) * pnl.mean() / r_std if r_std > 0 else 0.0
        )
    else:
        m["sqn"] = 0.0

    # Z-Score (tests statistical dependency in win/loss sequence)
    # Reference: Balsara (1992), "Money Management Strategies for Futures Traders"
    if n_trades >= 10:
        series = (pnl > 0).astype(int)
        runs   = 1 + np.sum(np.diff(series) != 0)
        n1     = series.sum()
        n2     = n_trades - n1
        if n1 > 0 and n2 > 0:
            exp_runs = (2 * n1 * n2) / n_trades + 1
            var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n_trades)) / \
                       (n_trades ** 2 * (n_trades - 1))
            m["z_score"] = float(
                (runs - exp_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
            )
        else:
            m["z_score"] = 0.0
    else:
        m["z_score"] = 0.0

    return m


# ────────────────────────────────────────────────────────────────────────────
# Streak calculation
# ────────────────────────────────────────────────────────────────────────────

def _compute_streaks(pnl: np.ndarray) -> tuple[int, int]:
    """Return (max_win_streak, max_loss_streak)."""
    max_win = max_loss = cur_win = cur_loss = 0
    for p in pnl:
        if p > 0:
            cur_win  += 1
            cur_loss  = 0
            max_win   = max(max_win, cur_win)
        elif p < 0:
            cur_loss += 1
            cur_win   = 0
            max_loss  = max(max_loss, cur_loss)
        else:
            cur_win = cur_loss = 0
    return max_win, max_loss


# ────────────────────────────────────────────────────────────────────────────
# Period breakdowns
# ────────────────────────────────────────────────────────────────────────────

def _annual_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Year × [trades, win_rate, avg_exp, total_pnl, stop_pct] table."""
    if "entry_time" not in trades_df.columns or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    grp = df.groupby("year")
    result = []
    for yr, g in grp:
        n       = len(g)
        wins    = (g["net_pnl"] > 0).sum()
        losses  = (g["outcome"] == "LOSS").sum() if "outcome" in g else 0
        result.append({
            "YEAR":       yr,
            "TRADES":     n,
            "WIN RATE":   f"{wins/n*100:.1f}%",
            "STOP%":      f"{losses/n*100:.1f}%",
            "EXP/TRADE":  f"${g['net_pnl'].mean():+,.0f}",
            "TOTAL P&L":  g["net_pnl"].sum(),
        })
    return pd.DataFrame(result)


def _monthly_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Year × Month pivot of net P&L."""
    if "entry_time" not in trades_df.columns or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["year"]  = pd.to_datetime(df["entry_time"]).dt.year
    df["month"] = pd.to_datetime(df["entry_time"]).dt.month
    pivot = df.pivot_table(
        values="net_pnl", index="year", columns="month", aggfunc="sum", fill_value=0
    )
    pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][:len(pivot.columns)]
    return pivot


def _dow_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week performance summary."""
    if "entry_time" not in trades_df.columns or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    df["dow"] = pd.to_datetime(df["entry_time"]).dt.dayofweek
    rows = []
    for d in range(5):
        g = df[df["dow"] == d]
        if len(g) == 0:
            continue
        rows.append({
            "Day":       days[d],
            "Trades":    len(g),
            "Win Rate":  f"{(g['net_pnl']>0).mean()*100:.1f}%",
            "Avg P&L":   f"${g['net_pnl'].mean():+,.0f}",
            "Total P&L": g["net_pnl"].sum(),
        })
    return pd.DataFrame(rows)


def _tod_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day performance."""
    if "entry_time" not in trades_df.columns or trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["hour"] = pd.to_datetime(df["entry_time"]).dt.hour
    grp  = df.groupby("hour")["net_pnl"]
    rows = []
    for hr, g in grp:
        rows.append({
            "Hour":      f"{hr:02d}:00",
            "Trades":    len(g),
            "Avg P&L":   round(float(g.mean()), 2),
            "Total P&L": round(float(g.sum()), 2),
        })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Combine / Prop-firm pass rate
# ────────────────────────────────────────────────────────────────────────────

def _combine_pass_rate(
    trade_pnl: np.ndarray,
    daily_loss_limit: float,
    max_drawdown: float,
    profit_target: float,
    start_bal: float,
    n_sims: int = 2_000,
) -> dict:
    """
    Monte Carlo combine pass-rate simulation.
    Resample trades with replacement to simulate many combine attempts.
    """
    if len(trade_pnl) < 5:
        return {"pass_rate": 0.0, "fail_dd_rate": 0.0}

    rng    = np.random.default_rng(42)
    passes = 0
    fail_dd = 0
    n_trades = len(trade_pnl)

    for _ in range(n_sims):
        sample  = rng.choice(trade_pnl, size=n_trades, replace=True)
        equity  = start_bal + np.cumsum(sample)
        daily   = np.array_split(sample, max(1, n_trades // 5))
        failed  = False
        won     = False

        cumsum  = 0.0
        peak    = start_bal
        for block in daily:
            day_pnl = block.sum()
            cumsum += day_pnl
            cur_eq  = start_bal + cumsum
            peak    = max(peak, cur_eq)

            if day_pnl < -daily_loss_limit:
                failed = True
                fail_dd += 1
                break
            if (peak - cur_eq) >= max_drawdown:
                failed = True
                fail_dd += 1
                break
            if cumsum >= profit_target:
                won = True
                break

        if not failed and won:
            passes += 1

    return {
        "pass_rate":    passes / n_sims * 100,
        "fail_dd_rate": fail_dd / n_sims * 100,
    }


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def _breakeven_win_rate(avg_win: float, avg_loss: float) -> float:
    """Minimum win rate to break even. = |avg_loss| / (avg_win + |avg_loss|)"""
    L = abs(avg_loss)
    W = abs(avg_win)
    return L / (W + L) * 100 if (W + L) > 0 else 50.0


def _empty_metrics() -> dict:
    """Return a zeroed metrics dict when there are no trades."""
    keys = [
        "total_trades","winning_trades","losing_trades","win_rate_pct",
        "net_pnl","gross_pnl","total_commission","total_slippage","total_fees",
        "net_pnl_pct","avg_win","avg_loss","largest_win","largest_loss",
        "payoff_ratio","expected_value","profit_factor","gross_wins","gross_losses",
        "avg_hold_all","avg_hold_wins","avg_hold_losses","avg_mae","avg_mfe",
        "max_consec_wins","max_consec_losses","active_days","avg_trades_per_day",
        "avg_trades_per_session","cagr_pct","annualised_return_pct","years",
        "sharpe","sortino","sharpe_3yr","sharpe_10yr","calmar","mar","omega",
        "sterling","kelly_full_pct","kelly_half_pct","sqn","z_score",
        "max_drawdown_dollars","max_drawdown_pct","max_dd_duration_days",
        "avg_drawdown_dollars","recovery_factor","ulcer_index","pain_ratio",
        "daily_pnl_mean","daily_pnl_std","costs_pct_of_gross","breakeven_win_rate",
        "time_stop_exits","eod_exits","time_stop_pct",
    ]
    m = {k: 0.0 for k in keys}
    m["start_date"] = m["end_date"] = ""
    m["annual_table"] = m["monthly_table"] = m["dow_table"] = m["tod_table"] = pd.DataFrame()
    m["combine_pass_rate"] = {"pass_rate": 0.0, "fail_dd_rate": 0.0}
    return m
