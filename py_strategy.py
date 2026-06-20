"""
py_strategy.py — Python Strategy Interface
==========================================
Allows traders to upload .py strategy files that implement the same
StrategyDefinition interface used by the C++ parser — enabling the
terminal to backtest BOTH .cpp AND .py systematic models through the
same engine, charts, metrics, and Monte Carlo simulator.

PYTHON STRATEGY CONTRACT
────────────────────────
A valid .py strategy file must define a class named Strategy that
inherits from PyStrategyBase and implements:

    on_bar(self, bar: dict, indicators: dict) -> Optional[str]
        Called on every bar. Return a signal name string or None.
        bar keys       : timestamp, open, high, low, close, volume
        indicator keys : atr, z_ret, z_mom, vwap, cvd

    stop_distance(self, bar: dict, indicators: dict) -> float
        Return stop distance in POINTS from entry price.

    target_distance(self, bar: dict, indicators: dict) -> float
        Return target distance in POINTS from entry price.

    (optional) position_size(self, equity: float, stop_dist: float,
                              point_value: float) -> int
        Return number of contracts. Defaults to fixed 1 if not defined.

EXAMPLE .py STRATEGY FILE
──────────────────────────
from py_strategy import PyStrategyBase

class Strategy(PyStrategyBase):
    Z_THRESHOLD = 0.45
    ATR_MULT    = 1.5
    TP_R        = 2.0

    def on_bar(self, bar, indicators):
        z   = indicators.get("z_ret", 0)
        mom = indicators.get("z_mom", 0)
        c   = bar["close"]
        vwap= indicators.get("vwap", c)
        if z > self.Z_THRESHOLD and mom > 0.3 and c > vwap:
            return "MOM_LONG"
        if z < -self.Z_THRESHOLD and mom < -0.3 and c < vwap:
            return "MOM_SHORT"
        return None

    def stop_distance(self, bar, indicators):
        return indicators.get("atr", 20.0) * self.ATR_MULT

    def target_distance(self, bar, indicators):
        return self.stop_distance(bar, indicators) * self.TP_R
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import INSTRUMENTS
from strategy_parser import (
    StrategyDefinition, SignalDef, StopDef, TargetDef,
    SizingDef, RiskDef, ParseReport, ParseWarning,
)


# ────────────────────────────────────────────────────────────────────────────
# Base class (traders inherit from this)
# ────────────────────────────────────────────────────────────────────────────

class PyStrategyBase:
    """
    Base class for Python strategy files.
    Subclass this in your .py strategy file as `class Strategy(PyStrategyBase)`.
    """

    # ── Override these in your subclass ──────────────────────────────────────
    NAME: str = "PyStrategy"
    VERSION: str = "1.0"
    INSTRUMENT: str = "NQ"
    COOLDOWN_BARS: int = 5

    def on_bar(self, bar: dict, indicators: dict) -> Optional[str]:
        """Return a signal name string (e.g. 'MOM_LONG') or None."""
        raise NotImplementedError("Implement on_bar() in your Strategy subclass.")

    def stop_distance(self, bar: dict, indicators: dict) -> float:
        """Return stop distance in POINTS (positive number)."""
        return indicators.get("atr", 20.0) * 1.5

    def target_distance(self, bar: dict, indicators: dict) -> float:
        """Return target distance in POINTS (positive number)."""
        return self.stop_distance(bar, indicators) * 2.0

    def position_size(self, equity: float, stop_dist: float,
                      point_value: float) -> int:
        """Return number of contracts. Default: 1 fixed."""
        return 1

    def on_trade_close(self, trade: dict) -> None:
        """Called when a trade closes. Override for state-machine sizing."""
        pass

    def on_session_start(self, date: str) -> None:
        """Called at the start of each new trading day."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# Python strategy loader & validator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class PyParseReport:
    strategy_class:  Optional[type]    = None
    strategy_name:   str               = ""
    source_file:     str               = ""
    is_valid:        bool              = False
    parse_score:     float             = 0.0
    line_count:      int               = 0
    summary:         list[str]         = field(default_factory=list)
    warnings:        list[str]         = field(default_factory=list)
    errors:          list[str]         = field(default_factory=list)
    strategy_def:    Optional[StrategyDefinition] = None


def load_py_strategy(source_code: str, filename: str = "strategy.py") -> PyParseReport:
    """
    Dynamically load and validate a Python strategy file.

    1. Write source to a temp file
    2. Import as a module
    3. Find the Strategy class (must inherit PyStrategyBase)
    4. Validate required methods
    5. Convert to StrategyDefinition for the backtest engine
    """
    report = PyParseReport(source_file=filename, line_count=len(source_code.splitlines()))

    # ── Write to temp file & import ──────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                     delete=False, encoding="utf-8") as f:
        f.write(source_code)
        tmp_path = f.name

    try:
        spec   = importlib.util.spec_from_file_location("user_strategy", tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        report.errors.append(f"❌ SyntaxError: {exc}")
        report.summary = report.errors
        return report
    except Exception as exc:
        report.errors.append(f"❌ Import error: {exc}")
        report.errors.append(traceback.format_exc()[-500:])
        report.summary = report.errors
        return report
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ── Find Strategy class ───────────────────────────────────────────────────
    strategy_class = None
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name == "Strategy" or (
            inspect.isclass(obj) and
            issubclass(obj, PyStrategyBase) and
            obj is not PyStrategyBase
        ):
            strategy_class = obj
            break

    if strategy_class is None:
        report.errors.append(
            "❌ No Strategy class found. "
            "Your .py file must define `class Strategy(PyStrategyBase):`"
        )
        report.summary = report.errors
        return report

    # ── Instantiate & validate ────────────────────────────────────────────────
    try:
        instance = strategy_class()
    except Exception as exc:
        report.errors.append(f"❌ Strategy.__init__() raised: {exc}")
        report.summary = report.errors
        return report

    report.strategy_class = strategy_class
    report.strategy_name  = getattr(instance, "NAME", strategy_class.__name__)
    report.is_valid       = True

    # ── Check method implementations ──────────────────────────────────────────
    methods_ok = {
        "on_bar":          hasattr(instance, "on_bar") and callable(instance.on_bar),
        "stop_distance":   hasattr(instance, "stop_distance"),
        "target_distance": hasattr(instance, "target_distance"),
        "position_size":   hasattr(instance, "position_size"),
    }

    score = 0.0
    for method, ok in methods_ok.items():
        if ok:
            score += 25.0
            report.summary.append(f"✅ {method}() — implemented")
        else:
            report.warnings.append(f"⚠️ {method}() — not found, using base default")

    report.parse_score = score

    # ── Inspect source for known signal names ─────────────────────────────────
    import re
    sig_names = re.findall(r"""return\s+["']([A-Z][A-Z0-9_]{2,})["']""", source_code)
    sig_names = list(dict.fromkeys(sig_names))  # unique, order-preserved

    signals = []
    for sn in sig_names:
        direction = "LONG" if any(x in sn for x in ("LONG", "BUY")) else "SHORT"
        signals.append(SignalDef(name=sn, direction=direction,
                                 conditions=["(Python logic)"], raw_code=["user.py"]))

    if signals:
        report.summary.append(
            f"✅ Signals detected: {', '.join(s.name for s in signals)}"
        )
    else:
        report.warnings.append(
            "⚠️ No signal name strings detected. "
            "Ensure on_bar() returns string literals like 'MOM_LONG'."
        )

    # ── Extract constants ──────────────────────────────────────────────────────
    const_pattern = re.compile(
        r"""^\s+([A-Z][A-Z0-9_]+)\s*=\s*([\d.+-]+)""", re.MULTILINE
    )
    raw_constants = {m.group(1): float(m.group(2))
                     for m in const_pattern.finditer(source_code)}

    if raw_constants:
        report.summary.append(
            f"✅ Constants: {', '.join(f'{k}={v}' for k, v in list(raw_constants.items())[:8])}"
        )

    # ── Convert to StrategyDefinition ─────────────────────────────────────────
    cooldown = getattr(instance, "COOLDOWN_BARS", 5)

    # Probe stop / target using a dummy bar
    dummy_bar  = {"close": 20000.0, "open": 20000.0, "high": 20010.0,
                  "low": 19990.0, "volume": 1000, "timestamp": ""}
    dummy_ind  = {"atr": 20.0, "z_ret": 0.0, "z_mom": 0.0, "vwap": 20000.0}
    try:
        stop_pts = instance.stop_distance(dummy_bar, dummy_ind)
        tgt_pts  = instance.target_distance(dummy_bar, dummy_ind)
        r_mult   = round(tgt_pts / stop_pts, 2) if stop_pts > 0 else 2.0
    except Exception:
        stop_pts, tgt_pts, r_mult = 20.0, 40.0, 2.0

    strat_def = StrategyDefinition(
        name          = report.strategy_name,
        source_file   = filename,
        signals       = signals if signals else [
            SignalDef("SIGNAL", "LONG", ["(Python on_bar logic)"], [])
        ],
        stop          = StopDef(mode="fixed_points", value=stop_pts,
                                description=f"Python stop_distance() → {stop_pts:.1f} pts"),
        target        = TargetDef(mode="r_multiple", r_multiple=r_mult,
                                  description=f"Python target_distance() → {tgt_pts:.1f} pts ({r_mult}R)"),
        sizing        = SizingDef(mode="fixed", base_size=1,
                                  description="Python position_size()"),
        cooldown_bars = cooldown,
        indicators    = _detect_indicators(source_code),
        regime_names  = _detect_regimes(source_code),
        raw_constants = raw_constants,
        fully_parsed  = True,
    )

    report.strategy_def  = strat_def
    report.summary += report.warnings
    return report


# ────────────────────────────────────────────────────────────────────────────
# Python strategy backtest adapter
# ────────────────────────────────────────────────────────────────────────────

class PyStrategyAdapter:
    """
    Wraps a loaded Python Strategy class so the BacktestEngine can call it
    using the same interface it uses for parsed C++ strategies.
    """

    def __init__(self, strategy_class: type, config_dict: dict):
        self._instance   = strategy_class()
        self._config     = config_dict
        self._cooldown   = 0
        self._last_date  = None

    def generate_signal(
        self,
        bar:        pd.Series,
        indicators: dict,
    ) -> Optional[tuple[str, str, float]]:
        """
        Call the Python strategy's on_bar() method.

        Returns
        -------
        (signal_name, direction, q_score)  or  None
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # Session-start callback
        ts = bar.name if hasattr(bar, "name") else None
        if ts is not None:
            bar_date = str(ts)[:10]
            if bar_date != self._last_date:
                self._instance.on_session_start(bar_date)
                self._last_date = bar_date

        bar_dict = {
            "timestamp": str(ts) if ts else "",
            "open":      float(bar.get("open",  0)),
            "high":      float(bar.get("high",  0)),
            "low":       float(bar.get("low",   0)),
            "close":     float(bar.get("close", 0)),
            "volume":    float(bar.get("volume",0)),
        }

        try:
            sig_name = self._instance.on_bar(bar_dict, indicators)
        except Exception as exc:
            return None  # never crash the engine from user code

        if sig_name is None:
            return None

        direction = (
            "LONG" if any(x in str(sig_name).upper()
                          for x in ("LONG", "BUY", "UP"))
            else "SHORT"
        )

        # Q-score: use composite momentum if available, else 1.0
        q = abs(float(indicators.get("z_mom", 1.0))) + abs(float(indicators.get("z_ret", 0.0)))
        q = round(min(q, 5.0), 2)

        return (str(sig_name), direction, q)

    def stop_distance(self, bar: pd.Series, indicators: dict) -> float:
        bar_dict = {"close": float(bar.get("close", 0)),
                    "open": float(bar.get("open", 0)),
                    "high": float(bar.get("high", 0)),
                    "low":  float(bar.get("low", 0))}
        try:
            return float(self._instance.stop_distance(bar_dict, indicators))
        except Exception:
            return float(indicators.get("atr", 20.0)) * 1.5

    def target_distance(self, bar: pd.Series, indicators: dict) -> float:
        bar_dict = {"close": float(bar.get("close", 0)),
                    "open": float(bar.get("open", 0)),
                    "high": float(bar.get("high", 0)),
                    "low":  float(bar.get("low", 0))}
        try:
            return float(self._instance.target_distance(bar_dict, indicators))
        except Exception:
            return self.stop_distance(bar, indicators) * 2.0

    def position_size(self, equity: float, stop_dist: float,
                      point_value: float) -> int:
        try:
            return int(self._instance.position_size(equity, stop_dist, point_value))
        except Exception:
            return 1

    def on_trade_close(self, trade_dict: dict) -> None:
        try:
            self._instance.on_trade_close(trade_dict)
        except Exception:
            pass

    def reset_cooldown(self, bars: int) -> None:
        self._cooldown = bars


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _detect_indicators(source: str) -> list[str]:
    import re
    pattern = re.compile(
        r"""\b(vwap|delta|cvd|composite|momentum|atr|rsi|ema|sma|
               z_ret|z_mom|zscore|slope|sigma|std|iv|gamma|theta|vega)\b""",
        re.VERBOSE | re.IGNORECASE,
    )
    return sorted(set(m.lower() for m in pattern.findall(source)))


def _detect_regimes(source: str) -> list[str]:
    import re
    from config import KNOWN_REGIME_NAMES
    found = []
    for r in KNOWN_REGIME_NAMES:
        if r in source:
            found.append(r)
    return found


# ────────────────────────────────────────────────────────────────────────────
# Example strategy for in-app "Load Example .py Strategy" button
# ────────────────────────────────────────────────────────────────────────────

EXAMPLE_PY_STRATEGY = '''"""
Example Python strategy for the Quant Terminal.
Upload this file in Section 2 — Strategy Loader.
"""
from py_strategy import PyStrategyBase


class Strategy(PyStrategyBase):
    """
    Momentum + VWAP Python Strategy
    ─────────────────────────────────
    Enters long when z-return is strong positive and price is above VWAP.
    Enters short when z-return is strong negative and price is below VWAP.
    Uses ATR-based dynamic stops and 2R fixed targets.
    """
    NAME           = "MOM_VWAP_PY"
    VERSION        = "1.0"
    INSTRUMENT     = "NQ"
    COOLDOWN_BARS  = 5

    Z_THRESHOLD    = 0.45
    MOM_THRESHOLD  = 0.30
    ATR_STOP_MULT  = 1.50
    TP_R_MULTIPLE  = 2.00
    DAILY_LOSS_LIM = 3000.0

    def __init__(self):
        super().__init__()
        self._daily_pnl  = 0.0
        self._last_date  = None
        self._win_streak = 0
        self._los_streak = 0

    def on_session_start(self, date: str) -> None:
        """Reset daily P&L on new trading day."""
        self._daily_pnl = 0.0
        self._last_date = date

    def on_bar(self, bar: dict, indicators: dict) -> str | None:
        """Main signal logic evaluated on every bar."""
        # Daily loss circuit breaker
        if self._daily_pnl < -self.DAILY_LOSS_LIM:
            return None

        z    = indicators.get("z_ret",  0.0)
        mom  = indicators.get("z_mom",  0.0)
        c    = bar["close"]
        vwap = indicators.get("vwap",   c)

        trend_up   = c > vwap
        trend_down = c < vwap

        # ── Long entry ──────────────────────────────────────────────────────
        if (z > self.Z_THRESHOLD and
                mom > self.MOM_THRESHOLD and
                trend_up):
            return "MOM_LONG"

        # ── Short entry ─────────────────────────────────────────────────────
        if (z < -self.Z_THRESHOLD and
                mom < -self.MOM_THRESHOLD and
                trend_down):
            return "MOM_SHORT"

        return None

    def stop_distance(self, bar: dict, indicators: dict) -> float:
        """ATR × 1.5 dynamic stop."""
        atr = indicators.get("atr", 20.0)
        return atr * self.ATR_STOP_MULT

    def target_distance(self, bar: dict, indicators: dict) -> float:
        """2.0R fixed target."""
        return self.stop_distance(bar, indicators) * self.TP_R_MULTIPLE

    def position_size(self, equity: float, stop_dist: float,
                      point_value: float) -> int:
        """Risk 1% of equity per trade."""
        risk_dollars = equity * 0.01
        stop_dollars = stop_dist * point_value
        if stop_dollars <= 0:
            return 1
        return max(1, int(risk_dollars / stop_dollars))

    def on_trade_close(self, trade: dict) -> None:
        """Update daily P&L and streak counters."""
        self._daily_pnl += trade.get("net_pnl", 0.0)
        if trade.get("net_pnl", 0) > 0:
            self._win_streak += 1
            self._los_streak  = 0
        else:
            self._los_streak += 1
            self._win_streak  = 0
'''
