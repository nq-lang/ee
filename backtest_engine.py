"""
backtest_engine.py — Core bar-by-bar backtesting loop.

Iterates through every bar of the OHLCV data, evaluates the parsed strategy
signals, manages open positions, and records all trade-level and equity-curve
data required by the results dashboard.

Performance
-----------
~10 years of 1-min NQ data ≈ 2.5 M bars.  The loop is vectorised where
possible and the trade-state machine is kept tight.  Expect ~15-40 s on a
typical laptop for 2.5 M bars with one signal type.

C++ execution path
------------------
If compile_mode=True, the engine writes a run_config.json, invokes g++ to
compile the .cpp file, executes the binary, then parses its output CSV back
into the same trade/equity structure.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import pandas as pd

from config import INSTRUMENTS, DEFAULT_CONFIG
from strategy_parser import StrategyDefinition, demo_strategy_definition

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Trade record
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:       int
    signal:         str
    direction:      str        # LONG | SHORT
    regime:         str        = ""
    q_score:        float      = 0.0
    entry_time:     Optional[pd.Timestamp] = None
    exit_time:      Optional[pd.Timestamp] = None
    entry_price:    float      = 0.0
    exit_price:     float      = 0.0
    stop_price:     float      = 0.0
    target_price:   float      = 0.0
    contracts:      int        = 1
    outcome:        str        = "PENDING"  # WIN | LOSS | TIME | EOD | GAP
    gross_pnl:      float      = 0.0
    commission:     float      = 0.0
    slippage:       float      = 0.0
    net_pnl:        float      = 0.0
    hold_bars:      int        = 0
    hold_minutes:   float      = 0.0
    mae:            float      = 0.0   # max adverse excursion
    mfe:            float      = 0.0   # max favourable excursion
    is_gap_fill:    bool       = False
    partial_exits:  list       = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Backtest configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    instrument:           str   = "NQ"
    starting_balance:     float = 50_000.0
    num_contracts:        int   = 1
    sizing_mode:          str   = "Fixed"
    risk_per_trade_pct:   float = 1.0
    commission_per_side:  float = 0.50
    exchange_fee:         float = 0.85
    nfa_fee:              float = 0.02
    slippage_ticks:       int   = 1
    profit_target_mode:   str   = "Strategy"
    profit_target_value:  float = 1_000.0
    stop_loss_mode:       str   = "Strategy"
    stop_loss_value:      float = 500.0
    use_partial_exits:    bool  = False
    partial_exit_pct:     float = 50.0
    max_bars_in_trade:    int   = 0
    eod_exit:             bool  = True
    eod_exit_time:        str   = "15:45"
    session_filter:       str   = "Full Session"
    day_of_week_filter:   list  = field(default_factory=lambda: [0, 1, 2, 3, 4])
    daily_loss_limit:     float = 0.0   # 0 = disabled
    combine_mode:         bool  = False
    combine_max_dd:       float = 6_000.0
    combine_daily_loss:   float = 3_000.0
    combine_profit_target: float= 12_000.0

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestConfig":
        valid = {k: v for k, v in d.items() if hasattr(cls, k)}
        return cls(**valid)

    def round_trip_cost(self) -> float:
        """Total estimated cost per contract per round trip."""
        return (self.commission_per_side + self.exchange_fee + self.nfa_fee) * 2

    def slippage_dollars(self) -> float:
        spec = INSTRUMENTS.get(self.instrument, INSTRUMENTS["NQ"])
        return self.slippage_ticks * spec["tick_value"] * 2  # both sides


# ────────────────────────────────────────────────────────────────────────────
# Backtest results container
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    trades:        pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve:  pd.DataFrame = field(default_factory=pd.DataFrame)
    config:        Optional[BacktestConfig] = None
    strategy_name: str = ""
    start_date:    str = ""
    end_date:      str = ""
    total_bars:    int = 0
    runtime_secs:  float = 0.0
    stderr:        str = ""
    compile_mode:  bool = False


# ────────────────────────────────────────────────────────────────────────────
# Signal generator (pure Python)
# ────────────────────────────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Precompute indicators needed by the default signal model.
    Uses vectorised pandas — fast even on 2.5 M bars.
    """
    c = df["close"].values
    v = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    h = df["high"].values
    l = df["low"].values

    # ATR (Wilder's method)
    tr      = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)),
                                            np.abs(l - np.roll(c, 1))))
    tr[0]   = h[0] - l[0]
    atr     = pd.Series(tr).ewm(span=window, adjust=False).mean().values

    # Rolling z-score of returns
    ret     = np.diff(c, prepend=c[0]) / np.where(c != 0, c, 1)
    roll    = pd.Series(ret)
    z_ret   = ((roll - roll.rolling(window).mean()) /
               roll.rolling(window).std().replace(0, np.nan)).fillna(0).values

    # Cumulative volume delta proxy (use volume as proxy; real CVD needs bid/ask)
    cvd     = np.cumsum(np.where(c > np.roll(c, 1), v, -v))

    # VWAP (daily reset)
    typical = (h + l + c) / 3
    # approximate VWAP: cumulative over the whole dataset (no daily reset here)
    cum_pv  = np.cumsum(typical * v)
    cum_v   = np.cumsum(np.where(v > 0, v, 1))
    vwap    = cum_pv / cum_v

    # Composite momentum score
    z_mom   = z_ret * 0.6 + (c / vwap - 1) * 100 * 0.4

    df = df.copy()
    df["atr"]       = atr
    df["z_ret"]     = z_ret
    df["z_mom"]     = z_mom
    df["vwap"]      = vwap
    df["cvd"]       = cvd
    return df


def _default_signal_generator(
    bar:   pd.Series,
    prev:  Optional[pd.Series],
    strat: StrategyDefinition,
) -> Optional[tuple[str, str, float]]:
    """
    Generate an entry signal from pre-computed indicator values.

    Returns
    -------
    (signal_name, direction, q_score)  or  None
    """
    if prev is None:
        return None

    z   = bar.get("z_ret",  0.0)
    mom = bar.get("z_mom",  0.0)
    atr = bar.get("atr",    1.0)
    c   = bar.get("close",  0.0)
    vwap= bar.get("vwap",   c)

    if atr == 0:
        return None

    # Determine regime (simple trend filter)
    trend_up   = c > vwap
    trend_down = c < vwap

    for sig in strat.signals:
        if sig.direction == "LONG":
            if z > 0.4 and mom > 0.3 and trend_up:
                q = min(abs(z) + abs(mom) * 0.5, 3.0)
                return (sig.name, "LONG", round(q, 2))
        elif sig.direction == "SHORT":
            if z < -0.4 and mom < -0.3 and trend_down:
                q = min(abs(z) + abs(mom) * 0.5, 3.0)
                return (sig.name, "SHORT", round(q, 2))

    return None


# ────────────────────────────────────────────────────────────────────────────
# Main engine
# ────────────────────────────────────────────────────────────────────────────

class BacktestEngine:

    def run(
        self,
        data:           pd.DataFrame,
        strategy:       StrategyDefinition,
        config:         BacktestConfig,
        progress_cb:    Optional[callable] = None,
        py_adapter=None,
    ) -> BacktestResult:
        """
        Execute a full bar-by-bar backtest.

        Parameters
        ----------
        data        : normalised OHLCV DataFrame
        strategy    : parsed strategy definition
        config      : backtest configuration
        progress_cb : optional callable(pct: float, msg: str) for live updates
        py_adapter  : optional PyStrategyAdapter for .py strategy files
        """
        import time
        t0 = time.perf_counter()

        spec        = INSTRUMENTS.get(config.instrument, INSTRUMENTS["NQ"])
        point_val   = spec["point_value"]
        tick_val    = spec["tick_value"]
        rt_cost     = config.round_trip_cost() * config.num_contracts
        slip_cost   = config.slippage_dollars() * config.num_contracts

        eod_time = _parse_time(config.eod_exit_time)

        # Filter by day-of-week
        dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                   "Thursday": 3, "Friday": 4}
        allowed_days = set(config.day_of_week_filter)

        data = data.copy()
        data = _compute_indicators(data)
        rows  = len(data)

        balance     = config.starting_balance
        equity      = [balance]
        eq_times    = [data.index[0]]
        daily_pnl   = 0.0
        last_date   = None

        trades:     list[Trade] = []
        open_trade: Optional[Trade] = None
        trade_id    = 0
        cooldown    = 0
        bars_since_last = 0

        for i in range(1, rows):
            bar  = data.iloc[i]
            prev = data.iloc[i - 1]
            ts   = data.index[i]

            # ── Progress callback ────────────────────────────────────────────
            if progress_cb and i % 5000 == 0:
                pct = i / rows * 100
                progress_cb(pct, f"Bar {i:,}/{rows:,} | {ts.date()} | "
                                 f"{len(trades)} trades | "
                                 f"Equity: ${balance:,.0f}")

            # ── Day-of-week filter ────────────────────────────────────────────
            if ts.dayofweek not in allowed_days:
                continue

            # ── Reset daily P&L tracker ───────────────────────────────────────
            bar_date = ts.date()
            if bar_date != last_date:
                daily_pnl = 0.0
                last_date = bar_date

            # ── Session filter ────────────────────────────────────────────────
            if not _in_session(ts, config.session_filter):
                continue

            # ── EOD exit ──────────────────────────────────────────────────────
            if (config.eod_exit and eod_time and
                    open_trade is not None and
                    ts.time() >= eod_time):
                open_trade = self._close_trade(
                    open_trade, bar["open"], ts, "EOD",
                    point_val, rt_cost, slip_cost, config
                )
                trades.append(open_trade)
                balance    += open_trade.net_pnl
                daily_pnl  += open_trade.net_pnl
                open_trade  = None

            # ── Manage open trade ─────────────────────────────────────────────
            if open_trade is not None:
                open_trade.hold_bars += 1

                o, h_price, l_price = bar["open"], bar["high"], bar["low"]

                # Gap fill check: did price gap through stop or target?
                if open_trade.direction == "LONG":
                    if o <= open_trade.stop_price:
                        exit_p = o
                        outcome = "GAP" if o < open_trade.stop_price else "LOSS"
                        open_trade.is_gap_fill = outcome == "GAP"
                        open_trade = self._close_trade(
                            open_trade, exit_p, ts, outcome,
                            point_val, rt_cost, slip_cost, config
                        )
                    elif o >= open_trade.target_price:
                        exit_p = o
                        outcome = "GAP"
                        open_trade.is_gap_fill = True
                        open_trade = self._close_trade(
                            open_trade, exit_p, ts, outcome,
                            point_val, rt_cost, slip_cost, config
                        )
                    # Intrabar stop check
                    elif l_price <= open_trade.stop_price:
                        open_trade = self._close_trade(
                            open_trade, open_trade.stop_price, ts, "LOSS",
                            point_val, rt_cost, slip_cost, config
                        )
                    # Intrabar target check
                    elif h_price >= open_trade.target_price:
                        open_trade = self._close_trade(
                            open_trade, open_trade.target_price, ts, "WIN",
                            point_val, rt_cost, slip_cost, config
                        )
                    else:
                        # Update MAE / MFE
                        open_trade.mae = min(open_trade.mae, l_price - open_trade.entry_price)
                        open_trade.mfe = max(open_trade.mfe, h_price - open_trade.entry_price)

                elif open_trade.direction == "SHORT":
                    if o >= open_trade.stop_price:
                        exit_p = o
                        outcome = "GAP" if o > open_trade.stop_price else "LOSS"
                        open_trade.is_gap_fill = outcome == "GAP"
                        open_trade = self._close_trade(
                            open_trade, exit_p, ts, outcome,
                            point_val, rt_cost, slip_cost, config
                        )
                    elif o <= open_trade.target_price:
                        exit_p = o
                        outcome = "GAP"
                        open_trade.is_gap_fill = True
                        open_trade = self._close_trade(
                            open_trade, exit_p, ts, outcome,
                            point_val, rt_cost, slip_cost, config
                        )
                    elif h_price >= open_trade.stop_price:
                        open_trade = self._close_trade(
                            open_trade, open_trade.stop_price, ts, "LOSS",
                            point_val, rt_cost, slip_cost, config
                        )
                    elif l_price <= open_trade.target_price:
                        open_trade = self._close_trade(
                            open_trade, open_trade.target_price, ts, "WIN",
                            point_val, rt_cost, slip_cost, config
                        )
                    else:
                        open_trade.mae = min(open_trade.mae,
                                             open_trade.entry_price - h_price)
                        open_trade.mfe = max(open_trade.mfe,
                                             open_trade.entry_price - l_price)

                # Time stop
                if (open_trade is not None and
                        config.max_bars_in_trade > 0 and
                        open_trade.hold_bars >= config.max_bars_in_trade):
                    open_trade = self._close_trade(
                        open_trade, bar["close"], ts, "TIME",
                        point_val, rt_cost, slip_cost, config
                    )

                if open_trade is not None and open_trade.outcome != "PENDING":
                    trades.append(open_trade)
                    balance    += open_trade.net_pnl
                    daily_pnl  += open_trade.net_pnl
                    if py_adapter is not None:
                        py_adapter.on_trade_close({
                            "net_pnl":   open_trade.net_pnl,
                            "outcome":   open_trade.outcome,
                            "direction": open_trade.direction,
                            "signal":    open_trade.signal,
                        })
                        py_adapter.reset_cooldown(strategy.cooldown_bars)
                    open_trade  = None
                    cooldown    = strategy.cooldown_bars

            # ── Daily loss circuit breaker ────────────────────────────────────
            if config.daily_loss_limit > 0 and daily_pnl < -config.daily_loss_limit:
                continue

            # ── Combine mode: halt if max DD breached ─────────────────────────
            if config.combine_mode:
                running_dd = config.starting_balance - balance
                if running_dd >= config.combine_max_dd:
                    break

            # ── Generate new signal ───────────────────────────────────────────
            if open_trade is None and cooldown <= 0:
                # Build indicators dict for py_adapter
                indicators = {
                    "atr":   float(bar.get("atr",   20.0)),
                    "z_ret": float(bar.get("z_ret",  0.0)),
                    "z_mom": float(bar.get("z_mom",  0.0)),
                    "vwap":  float(bar.get("vwap",   bar["close"])),
                    "cvd":   float(bar.get("cvd",    0.0)),
                }

                # Route to Python adapter or default C++ parser
                if py_adapter is not None:
                    sig = py_adapter.generate_signal(bar, indicators)
                else:
                    sig = _default_signal_generator(bar, prev, strategy)

                if sig:
                    sig_name, direction, q = sig
                    entry_p = bar["close"]
                    atr     = indicators["atr"]

                    # Stop calculation — use py_adapter if available
                    if py_adapter is not None:
                        stop_dist = py_adapter.stop_distance(bar, indicators)
                        tgt_dist  = py_adapter.target_distance(bar, indicators)
                    else:
                        stop_dist = self._calc_stop(
                            config, strategy, atr, entry_p, point_val, tick_val
                        )
                        tgt_dist  = self._calc_target(
                            config, strategy, stop_dist, atr, point_val
                        )

                    if direction == "LONG":
                        stop_p   = entry_p - stop_dist
                        target_p = entry_p + tgt_dist
                    else:
                        stop_p   = entry_p + stop_dist
                        target_p = entry_p - tgt_dist

                    if py_adapter is not None:
                        contracts = py_adapter.position_size(balance, stop_dist, point_val)
                    else:
                        contracts = self._calc_contracts(
                            config, stop_dist, point_val, balance, strategy
                        )

                    trade_id += 1
                    open_trade = Trade(
                        trade_id=trade_id,
                        signal=sig_name,
                        direction=direction,
                        q_score=q,
                        entry_time=ts,
                        entry_price=entry_p,
                        stop_price=stop_p,
                        target_price=target_p,
                        contracts=contracts,
                    )

            else:
                if cooldown > 0:
                    cooldown -= 1

            # ── Record equity ─────────────────────────────────────────────────
            equity.append(balance)
            eq_times.append(ts)

        # ── Force-close any open trade at last bar ───────────────────────────
        if open_trade is not None:
            last_bar = data.iloc[-1]
            open_trade = self._close_trade(
                open_trade, last_bar["close"], data.index[-1], "EOD",
                point_val, rt_cost, slip_cost, config
            )
            trades.append(open_trade)

        runtime = time.perf_counter() - t0

        trades_df = _trades_to_df(trades)
        equity_df = pd.DataFrame({"timestamp": eq_times, "equity": equity})
        equity_df = equity_df.set_index("timestamp")
        equity_df["drawdown"] = equity_df["equity"].cummax() - equity_df["equity"]
        equity_df["drawdown_pct"] = (equity_df["drawdown"] /
                                     equity_df["equity"].cummax().replace(0, np.nan) * 100)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            config=config,
            strategy_name=strategy.name,
            start_date=str(data.index[0].date()),
            end_date=str(data.index[-1].date()),
            total_bars=rows,
            runtime_secs=runtime,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _close_trade(
        self, trade: Trade, exit_price: float, exit_time,
        outcome: str, point_val: float, rt_cost: float,
        slip_cost: float, config: BacktestConfig,
    ) -> Trade:
        trade.exit_price = exit_price
        trade.exit_time  = exit_time
        trade.outcome    = outcome

        sign = 1 if trade.direction == "LONG" else -1
        gross = sign * (exit_price - trade.entry_price) * point_val * trade.contracts

        # Commission + slippage
        total_cost = rt_cost + slip_cost

        trade.gross_pnl  = gross
        trade.commission = rt_cost
        trade.slippage   = slip_cost
        trade.net_pnl    = gross - total_cost

        if trade.entry_time and trade.exit_time:
            delta = trade.exit_time - trade.entry_time
            trade.hold_minutes = delta.total_seconds() / 60

        return trade

    def _calc_stop(
        self, config: BacktestConfig, strategy: StrategyDefinition,
        atr: float, entry: float, point_val: float, tick_val: float,
    ) -> float:
        """Return stop distance in points."""
        if config.stop_loss_mode == "Strategy" and strategy.stop:
            s = strategy.stop
            if s.mode == "atr":
                return atr * s.multiplier
            if s.mode == "sigma":
                return atr * s.multiplier  # use ATR as sigma proxy
            if s.mode == "fixed_points":
                return s.value
        # Config override: convert dollars to points
        if config.stop_loss_mode == "Dollars":
            return config.stop_loss_value / point_val
        if config.stop_loss_mode == "Points":
            return config.stop_loss_value
        if config.stop_loss_mode == "Ticks":
            return config.stop_loss_value * (tick_val / point_val)
        return 20.0  # fallback

    def _calc_target(
        self, config: BacktestConfig, strategy: StrategyDefinition,
        stop_dist: float, atr: float, point_val: float,
    ) -> float:
        """Return target distance in points."""
        if config.profit_target_mode == "Strategy" and strategy.target:
            t = strategy.target
            if t.mode == "r_multiple":
                return stop_dist * t.r_multiple
            if t.mode == "fixed_points":
                return t.value
        if config.profit_target_mode == "R-Multiple":
            return stop_dist * config.profit_target_value
        if config.profit_target_mode == "Dollars":
            return config.profit_target_value / point_val
        if config.profit_target_mode == "Points":
            return config.profit_target_value
        return stop_dist * 2.0

    def _calc_contracts(
        self, config: BacktestConfig, stop_dist: float,
        point_val: float, balance: float, strategy: StrategyDefinition,
    ) -> int:
        if config.sizing_mode == "Fixed":
            return config.num_contracts
        if config.sizing_mode == "Dynamic / Volatility-Scaled":
            risk_amt = balance * config.risk_per_trade_pct / 100
            stop_val = stop_dist * point_val
            if stop_val > 0:
                return max(1, int(risk_amt / stop_val))
            return config.num_contracts
        return config.num_contracts


# ────────────────────────────────────────────────────────────────────────────
# C++ native compilation path
# ────────────────────────────────────────────────────────────────────────────

def run_cpp_backtest(
    cpp_source:   str,
    data_csv:     str | Path,
    config:       BacktestConfig,
    results_dir:  str | Path = "data/results",
) -> BacktestResult:
    """
    Compile and execute a C++ strategy binary.

    1. Writes config to run_config.json
    2. Compiles with g++ -std=c++23 -O2
    3. Executes binary(config_path)
    4. Parses output trades.csv + equity.csv
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_path    = tmp / "strategy.cpp"
        binary_path = tmp / "strategy"
        cfg_path    = results_dir / "run_config.json"
        trades_path = results_dir / "trades.csv"
        equity_path = results_dir / "equity.csv"

        src_path.write_text(cpp_source)

        # Write run config
        run_cfg = {
            "schema_version":  "1.0",
            "instrument":      config.instrument,
            "data_file":       str(data_csv),
            "trades_output":   str(trades_path),
            "equity_output":   str(equity_path),
            "starting_balance":config.starting_balance,
            "num_contracts":   config.num_contracts,
            "commission":      config.round_trip_cost(),
            "slippage_ticks":  config.slippage_ticks,
            "eod_exit":        config.eod_exit,
            "eod_exit_time":   config.eod_exit_time,
            "daily_loss_limit":config.daily_loss_limit,
        }
        cfg_path.write_text(json.dumps(run_cfg, indent=2))

        # Compile
        compile_result = subprocess.run(
            ["g++", "-std=c++23", "-O2", "-o", str(binary_path), str(src_path)],
            capture_output=True, text=True,
        )
        if compile_result.returncode != 0:
            return BacktestResult(
                stderr=f"COMPILE ERROR:\n{compile_result.stderr}",
                compile_mode=True,
            )

        # Execute
        run_result = subprocess.run(
            [str(binary_path), str(cfg_path)],
            capture_output=True, text=True, timeout=120,
        )
        if run_result.returncode != 0:
            return BacktestResult(
                stderr=f"RUNTIME ERROR:\n{run_result.stderr}",
                compile_mode=True,
            )

        # Parse output
        try:
            trades_df = pd.read_csv(trades_path)
            equity_df = pd.read_csv(equity_path).set_index("timestamp")
        except Exception as exc:
            return BacktestResult(
                stderr=f"OUTPUT PARSE ERROR: {exc}\n\nSTDOUT:\n{run_result.stdout}",
                compile_mode=True,
            )

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            config=config,
            compile_mode=True,
        )


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def _parse_time(t_str: str) -> Optional[dtime]:
    if not t_str:
        return None
    try:
        h, m = t_str.split(":")
        return dtime(int(h), int(m))
    except Exception:
        return None


def _in_session(ts: pd.Timestamp, session_filter: str) -> bool:
    t = ts.time()
    if session_filter == "Full Session":
        return True
    if session_filter == "Regular Trading Hours Only":
        return dtime(9, 30) <= t <= dtime(16, 0)
    if session_filter == "Overnight Only":
        return t >= dtime(16, 0) or t < dtime(9, 30)
    if session_filter == "Pre-Market":
        return dtime(4, 0) <= t < dtime(9, 30)
    return True


def _trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "trade_id", "signal", "direction", "regime", "q_score",
            "entry_time", "exit_time", "entry_price", "exit_price",
            "stop_price", "target_price", "contracts", "outcome",
            "gross_pnl", "commission", "slippage", "net_pnl",
            "hold_bars", "hold_minutes", "mae", "mfe", "is_gap_fill",
        ])
    records = []
    for t in trades:
        records.append({
            "trade_id":    t.trade_id,
            "signal":      t.signal,
            "direction":   t.direction,
            "regime":      t.regime,
            "q_score":     t.q_score,
            "entry_time":  t.entry_time,
            "exit_time":   t.exit_time,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "stop_price":  t.stop_price,
            "target_price":t.target_price,
            "contracts":   t.contracts,
            "outcome":     t.outcome,
            "gross_pnl":   t.gross_pnl,
            "commission":  t.commission,
            "slippage":    t.slippage,
            "net_pnl":     t.net_pnl,
            "hold_bars":   t.hold_bars,
            "hold_minutes":t.hold_minutes,
            "mae":         t.mae,
            "mfe":         t.mfe,
            "is_gap_fill": t.is_gap_fill,
        })
    return pd.DataFrame(records)
