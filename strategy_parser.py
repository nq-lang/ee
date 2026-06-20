"""
strategy_parser.py — C++ trading strategy parser.

Extracts and faithfully replicates all trading logic from .cpp source files:
  - Entry signals (long/short, conditions, thresholds)
  - Stop loss & take profit placement logic
  - Position sizing (fixed, state-machine, volatility-scaled)
  - Regime detection and market-state labels
  - Cooldown / blocking logic
  - Risk management circuit breakers

Returns a StrategyDefinition dataclass and a ParseReport for display.
The Python backtest engine consumes StrategyDefinition to drive the simulation.
"""

from __future__ import annotations

import re
import textwrap
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from config import KNOWN_SIGNAL_NAMES, KNOWN_REGIME_NAMES

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalDef:
    name:       str
    direction:  str          # "LONG" | "SHORT"
    conditions: list[str]    # human-readable condition strings
    raw_code:   list[str]    # raw C++ condition snippets
    q_var:      Optional[str] = None
    regime:     Optional[str] = None


@dataclass
class StopDef:
    mode:        str          # "fixed_points" | "atr" | "sigma" | "ticks" | "dollars"
    value:       float = 0.0
    multiplier:  float = 1.0
    description: str   = ""


@dataclass
class TargetDef:
    mode:        str          # "fixed_points" | "r_multiple" | "dollars"
    value:       float = 0.0
    r_multiple:  float = 0.0
    description: str   = ""
    partial_exit: bool = False
    partial_pct:  float = 0.0


@dataclass
class SizingDef:
    mode:        str          # "fixed" | "state_machine" | "volatility_scaled"
    base_size:   int   = 1
    description: str   = ""
    raw_code:    str   = ""


@dataclass
class RiskDef:
    daily_loss_limit:       Optional[float] = None
    max_consecutive_losses: Optional[int]   = None
    equity_filter:          Optional[str]   = None
    description:            str             = ""


@dataclass
class ParseWarning:
    element:  str
    message:  str
    severity: str  # "error" | "warning" | "info"


@dataclass
class StrategyDefinition:
    name:            str = "Unnamed Strategy"
    source_file:     str = ""
    signals:         list[SignalDef]   = field(default_factory=list)
    stop:            Optional[StopDef]   = None
    target:          Optional[TargetDef] = None
    sizing:          Optional[SizingDef] = None
    risk:            Optional[RiskDef]   = None
    regime_names:    list[str]          = field(default_factory=list)
    indicators:      list[str]          = field(default_factory=list)
    cooldown_bars:   int                = 0
    eod_exit:        bool               = True
    pyramid_allowed: bool               = False
    raw_constants:   dict[str, Any]     = field(default_factory=dict)
    warnings:        list[ParseWarning] = field(default_factory=list)
    fully_parsed:    bool               = False


@dataclass
class ParseReport:
    strategy:    StrategyDefinition
    raw_source:  str
    line_count:  int
    parse_score: float   # 0-100% confidence
    summary:     list[str] = field(default_factory=list)
    warnings:    list[ParseWarning] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────────────────────────────────

class CppStrategyParser:
    """
    Regex-based C++ strategy parser.

    Covers common systematic strategy patterns:
      - if/else signal blocks with condition chains
      - Named constants (#define / constexpr / const double)
      - ATR/sigma/fixed stop/target patterns
      - State-machine sizing blocks
      - Regime enum/switch patterns
    """

    # ── Regex patterns ───────────────────────────────────────────────────────

    # Named numeric constants: #define FOO 1.5  |  const double FOO = 1.5;
    RE_CONST = re.compile(
        r"(?:\#define\s+(\w+)\s+([\d.+\-]+))"
        r"|(?:(?:const(?:expr)?)\s+(?:double|float|int|long)\s+(\w+)\s*=\s*([\d.+\-]+))"
    )

    # Entry signal blocks — looks for IF blocks that set a signal variable
    RE_SIGNAL_BLOCK = re.compile(
        r"""if\s*\(([^)]{5,})\)\s*\{[^}]*?(?:signal|sig|entry|trade|order)\s*=\s*
            ["']?([A-Z_]{3,})["']?""",
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )

    # Inline signal assignment: signalName = MOM_LONG; or signal = "MOM_SHORT";
    RE_SIGNAL_ASSIGN = re.compile(
        r"""(?:signal|sig|entry|trade_dir|direction)\s*=\s*["']?([A-Z][A-Z0-9_]{2,})["']?\s*;""",
        re.IGNORECASE,
    )

    # Stop loss patterns
    RE_STOP = re.compile(
        r"""(?:stop(?:_?loss|_?price|Level)?|sl)\s*=\s*([^\n;]{3,60})""",
        re.IGNORECASE,
    )

    # Take profit patterns
    RE_TARGET = re.compile(
        r"""(?:take_?profit|target(?:_?price|Level)?|tp)\s*=\s*([^\n;]{3,60})""",
        re.IGNORECASE,
    )

    # ATR-based stop: stopDist = k * atr  or  stop = entry - 1.5 * atr
    RE_ATR_STOP = re.compile(r"""(\d+\.?\d*)\s*\*\s*atr""", re.IGNORECASE)
    RE_SIGMA_STOP = re.compile(r"""(\d+\.?\d*)\s*\*\s*(?:sigma|std|stddev|vol)""", re.IGNORECASE)

    # R-multiple target: target = entry + R * 1.5  or  tpDist = 2.0 * risk
    RE_R_TARGET = re.compile(r"""(\d+\.?\d*)\s*\*\s*(?:risk|stop|r\b|rratio)""", re.IGNORECASE)

    # Regime detection
    RE_REGIME = re.compile(
        r"""regime\s*=\s*["']?([A-Z][A-Z0-9_]{2,})["']?""",
        re.IGNORECASE,
    )

    # Common indicator references
    RE_INDICATOR = re.compile(
        r"""\b(vwap|delta|cvd|composite|momentum|atr|rsi|ema|sma|z(?:ret|vol|score)|
            slope|sigma|std|iv|gamma|theta|vega)\b""",
        re.VERBOSE | re.IGNORECASE,
    )

    # Cooldown / min-bars-between-trades
    RE_COOLDOWN = re.compile(
        r"""(?:cooldown|minBars(?:Between)?|bar(?:s)?Since)\s*[>=<]+\s*(\d+)""",
        re.IGNORECASE,
    )

    # State machine sizing
    RE_STATE_SIZE = re.compile(
        r"""(?:size|qty|contracts?|lots?)\s*[+\-]?=\s*\d+""",
        re.IGNORECASE,
    )

    # Daily loss circuit breaker
    RE_DAILY_LOSS = re.compile(
        r"""dailyLoss\s*[>=<]+\s*([\d.]+)""",
        re.IGNORECASE,
    )

    # Pyramid / add-to-position
    RE_PYRAMID = re.compile(
        r"""pyramid|addToPosition|scale_?in|stackPosition""",
        re.IGNORECASE,
    )

    # Function/strategy name
    RE_STRATEGY_NAME = re.compile(
        r"""(?:class|struct|namespace|strategy_name|STRATEGY_NAME)\s+(\w+)""",
        re.IGNORECASE,
    )

    # ────────────────────────────────────────────────────────────────────────
    def parse(self, source_code: str, filename: str = "strategy.cpp") -> ParseReport:
        strat  = StrategyDefinition(source_file=filename)
        source = source_code
        lines  = source.splitlines()

        # ── Strategy name ────────────────────────────────────────────────────
        m = self.RE_STRATEGY_NAME.search(source)
        strat.name = m.group(1) if m else Path_stem(filename)

        # ── Named constants ──────────────────────────────────────────────────
        strat.raw_constants = self._extract_constants(source)

        # ── Signals ──────────────────────────────────────────────────────────
        strat.signals = self._extract_signals(source)
        if not strat.signals:
            strat.warnings.append(ParseWarning(
                "signals", "No entry signal blocks detected. "
                "The terminal will use default long-only logic for demonstration.",
                "warning",
            ))

        # ── Stop loss ────────────────────────────────────────────────────────
        strat.stop = self._extract_stop(source, strat.raw_constants)

        # ── Take profit ──────────────────────────────────────────────────────
        strat.target = self._extract_target(source, strat.raw_constants)

        # ── Sizing ───────────────────────────────────────────────────────────
        strat.sizing = self._extract_sizing(source)

        # ── Regimes ──────────────────────────────────────────────────────────
        regimes = set(re.findall(r"""["']?([A-Z][A-Z0-9_]{2,})["']?""", source))
        strat.regime_names = [r for r in regimes if r in KNOWN_REGIME_NAMES]

        # ── Indicators ───────────────────────────────────────────────────────
        inds = set(m.lower() for m in self.RE_INDICATOR.findall(source))
        strat.indicators = sorted(inds)

        # ── Cooldown ─────────────────────────────────────────────────────────
        m = self.RE_COOLDOWN.search(source)
        if m:
            strat.cooldown_bars = int(m.group(1))

        # ── Risk / circuit breaker ───────────────────────────────────────────
        strat.risk = self._extract_risk(source, strat.raw_constants)

        # ── Pyramid ──────────────────────────────────────────────────────────
        strat.pyramid_allowed = bool(self.RE_PYRAMID.search(source))

        # ── Completeness score ───────────────────────────────────────────────
        score = self._score_parse(strat)
        strat.fully_parsed = score >= 60.0

        # ── Build readable summary ───────────────────────────────────────────
        summary = self._build_summary(strat)

        return ParseReport(
            strategy=strat,
            raw_source=source,
            line_count=len(lines),
            parse_score=score,
            summary=summary,
            warnings=strat.warnings,
        )

    # ── Extraction helpers ───────────────────────────────────────────────────

    def _extract_constants(self, src: str) -> dict[str, float]:
        constants: dict[str, float] = {}
        for m in self.RE_CONST.finditer(src):
            name  = m.group(1) or m.group(3)
            value = m.group(2) or m.group(4)
            if name and value:
                try:
                    constants[name] = float(value)
                except ValueError:
                    pass
        return constants

    def _extract_signals(self, src: str) -> list[SignalDef]:
        signals: list[SignalDef] = []
        seen: set[str] = set()

        # Method 1: block-level IF → signal assignment
        for m in self.RE_SIGNAL_BLOCK.finditer(src):
            cond_block = m.group(1).strip()
            sig_name   = m.group(2).upper()
            if sig_name in seen:
                continue
            seen.add(sig_name)
            direction = "LONG" if any(x in sig_name for x in ("LONG", "BUY", "UP")) else "SHORT"
            conditions = self._parse_conditions(cond_block)
            signals.append(SignalDef(
                name=sig_name, direction=direction,
                conditions=conditions, raw_code=[cond_block],
            ))

        # Method 2: scan for known signal names used as string literals
        for sig_name in KNOWN_SIGNAL_NAMES:
            if sig_name in src and sig_name not in seen:
                seen.add(sig_name)
                direction = "LONG" if "LONG" in sig_name or "BUY" in sig_name else "SHORT"
                # Search for nearby conditions
                idx = src.find(sig_name)
                context = src[max(0, idx - 300): idx + 50]
                conditions = self._parse_conditions(context)
                signals.append(SignalDef(
                    name=sig_name, direction=direction,
                    conditions=conditions, raw_code=[context[:120]],
                ))

        return signals

    def _parse_conditions(self, block: str) -> list[str]:
        """Convert C++ condition string into human-readable bullet points."""
        # Split on && and || operators
        parts = re.split(r"&&|\|\|", block)
        readable = []
        for p in parts:
            p = p.strip().strip("()").strip()
            if len(p) > 3:
                p = self._humanise_condition(p)
                readable.append(p)
        return readable[:8]  # cap at 8 conditions

    def _humanise_condition(self, cond: str) -> str:
        """Apply simple substitutions to make C++ conditions more readable."""
        replacements = [
            (r"zRet\s*>\s*", "z-Return > "),
            (r"zVol\s*>\s*", "z-Volume > "),
            (r"delta\s*>\s*", "Delta > "),
            (r"cvd\s*>\s*", "CVD > "),
            (r"atr\s*>\s*", "ATR > "),
            (r"vwap\s*>\s*", "Price > VWAP"),
            (r"close\s*>\s*vwap", "Close > VWAP"),
            (r"slope\s*>\s*", "Slope > "),
            (r"composite\s*>\s*", "Composite Score > "),
            (r"momentum\s*>\s*", "Momentum > "),
        ]
        for pattern, replacement in replacements:
            cond = re.sub(pattern, replacement, cond, flags=re.IGNORECASE)
        return cond.strip()

    def _extract_stop(self, src: str, constants: dict) -> StopDef:
        # ATR-based stop
        m = self.RE_ATR_STOP.search(src)
        if m:
            mult = float(m.group(1))
            return StopDef(
                mode="atr", multiplier=mult,
                description=f"ATR × {mult} (dynamic stop)",
            )

        # Sigma-based stop
        m = self.RE_SIGMA_STOP.search(src)
        if m:
            mult = float(m.group(1))
            return StopDef(
                mode="sigma", multiplier=mult,
                description=f"{mult}σ volatility stop",
            )

        # Look for named constant used as stop
        stop_consts = {k: v for k, v in constants.items()
                       if any(x in k.lower() for x in ("stop", "sl", "loss", "risk"))}
        if stop_consts:
            name, val = next(iter(stop_consts.items()))
            return StopDef(mode="fixed_points", value=val,
                           description=f"Fixed stop: {val} points ({name})")

        # Generic stop expression
        m = self.RE_STOP.search(src)
        if m:
            expr = m.group(1).strip()
            return StopDef(mode="fixed_points", description=f"Stop expression: {expr}")

        return StopDef(mode="fixed_points", value=20.0,
                       description="⚠️ Stop not detected — defaulting to 20 points. "
                                   "Override in Configuration → Stop Loss Settings.")

    def _extract_target(self, src: str, constants: dict) -> TargetDef:
        # R-multiple target
        m = self.RE_R_TARGET.search(src)
        if m:
            r = float(m.group(1))
            return TargetDef(mode="r_multiple", r_multiple=r,
                             description=f"{r}R fixed target")

        # Named constant target
        tp_consts = {k: v for k, v in constants.items()
                     if any(x in k.lower() for x in ("target", "tp", "profit", "take"))}
        if tp_consts:
            name, val = next(iter(tp_consts.items()))
            return TargetDef(mode="fixed_points", value=val,
                             description=f"Fixed target: {val} points ({name})")

        m = self.RE_TARGET.search(src)
        if m:
            expr = m.group(1).strip()
            return TargetDef(mode="fixed_points", description=f"Target expression: {expr}")

        return TargetDef(mode="fixed_points", value=40.0,
                         description="⚠️ Target not detected — defaulting to 40 points.")

    def _extract_sizing(self, src: str) -> SizingDef:
        # State machine sizing: changes to qty variable based on wins/losses
        state_lines = self.RE_STATE_SIZE.findall(src)
        if len(state_lines) >= 2:
            return SizingDef(
                mode="state_machine",
                raw_code="\n".join(state_lines[:5]),
                description="State-machine sizing detected (win/loss streak adjustment)",
            )

        # Volatility-scaled: ATR or sigma in sizing logic
        if re.search(r"(?:atr|sigma|vol)\s*\*.*?(?:size|qty|contracts?)", src, re.IGNORECASE):
            return SizingDef(
                mode="volatility_scaled",
                description="Volatility-scaled sizing (ATR/sigma-based)",
            )

        return SizingDef(mode="fixed", base_size=1, description="Fixed contract sizing")

    def _extract_risk(self, src: str, constants: dict) -> RiskDef:
        daily_loss = None
        m = self.RE_DAILY_LOSS.search(src)
        if m:
            daily_loss = float(m.group(1))

        daily_consts = {k: v for k, v in constants.items()
                        if any(x in k.lower() for x in ("daily", "dayloss", "daylimit"))}
        if daily_consts and daily_loss is None:
            daily_loss = next(iter(daily_consts.values()))

        desc_parts = []
        if daily_loss:
            desc_parts.append(f"Daily loss limit: ${daily_loss:,.0f}")

        return RiskDef(
            daily_loss_limit=daily_loss,
            description=", ".join(desc_parts) if desc_parts else "No risk limits detected",
        )

    def _score_parse(self, strat: StrategyDefinition) -> float:
        score = 0.0
        if strat.signals:                    score += 30.0
        if strat.stop and strat.stop.mode != "fixed_points":  score += 20.0
        elif strat.stop:                     score += 10.0
        if strat.target and strat.target.r_multiple > 0:      score += 20.0
        elif strat.target:                   score += 10.0
        if strat.sizing and strat.sizing.mode != "fixed":      score += 15.0
        elif strat.sizing:                   score += 5.0
        if strat.indicators:                 score += 10.0
        if strat.regime_names:               score += 5.0
        return min(score, 100.0)

    def _build_summary(self, strat: StrategyDefinition) -> list[str]:
        lines = []
        sigs  = strat.signals

        if sigs:
            sig_names = ", ".join(s.name for s in sigs)
            lines.append(f"✅ Entry Signals Detected: {sig_names}")
            for sig in sigs:
                lines.append(f"   ↳ {sig.name} ({sig.direction}): "
                              + "; ".join(sig.conditions[:3] or ["conditions not parsed"]))
        else:
            lines.append("❌ Entry Signals: Not detected")

        if strat.stop:
            lines.append(f"✅ Stop Loss: {strat.stop.description}")
        else:
            lines.append("❌ Stop Loss: Not detected")

        if strat.target:
            lines.append(f"✅ Take Profit: {strat.target.description}")
        else:
            lines.append("❌ Take Profit: Not detected")

        if strat.sizing:
            lines.append(f"✅ Position Sizing: {strat.sizing.description}")

        if strat.regime_names:
            lines.append(f"✅ Regimes: {', '.join(strat.regime_names)}")
        else:
            lines.append("ℹ️  Regime Detection: None found")

        if strat.indicators:
            lines.append(f"✅ Indicators Referenced: {', '.join(strat.indicators)}")

        if strat.cooldown_bars:
            lines.append(f"✅ Signal Cooldown: {strat.cooldown_bars} bars")

        if strat.risk and strat.risk.daily_loss_limit:
            lines.append(f"✅ Risk Management: {strat.risk.description}")

        if strat.pyramid_allowed:
            lines.append("✅ Pyramid Entries: Detected")

        for w in strat.warnings:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(w.severity, "•")
            lines.append(f"{icon} {w.element}: {w.message}")

        return lines


# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────

def Path_stem(filename: str) -> str:
    """Return the filename stem without extension."""
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def parse_cpp_strategy(source_code: str, filename: str = "strategy.cpp") -> ParseReport:
    """Convenience wrapper — instantiate parser and return ParseReport."""
    return CppStrategyParser().parse(source_code, filename)


def demo_strategy_definition() -> StrategyDefinition:
    """
    Return a fully-defined demo strategy for use when no .cpp file is uploaded.
    Mimics a momentum + regime-filtered NQ strategy.
    """
    return StrategyDefinition(
        name="DEMO_MOM_STRATEGY",
        source_file="<built-in demo>",
        signals=[
            SignalDef(
                name="MOM_LONG",
                direction="LONG",
                conditions=["z-Return > 0.45", "z-Volume > 1.52", "Close > VWAP",
                            "Regime = TRENDING"],
                raw_code=["demo"],
            ),
            SignalDef(
                name="MOM_SHORT",
                direction="SHORT",
                conditions=["z-Return < -0.45", "z-Volume > 1.52", "Close < VWAP",
                            "Regime = TRENDING"],
                raw_code=["demo"],
            ),
        ],
        stop=StopDef(mode="atr", multiplier=1.5,
                     description="ATR × 1.5 dynamic stop"),
        target=TargetDef(mode="r_multiple", r_multiple=2.0,
                         description="2.0R take profit with 50% partial at 1.0R",
                         partial_exit=True, partial_pct=50.0),
        sizing=SizingDef(mode="fixed", base_size=1,
                         description="1 contract fixed"),
        risk=RiskDef(daily_loss_limit=3000.0,
                     description="Daily loss limit $3,000"),
        regime_names=["TRENDING", "ROTATIONAL"],
        indicators=["vwap", "atr", "zret", "zvol", "cvd"],
        cooldown_bars=5,
        eod_exit=True,
        fully_parsed=True,
    )
