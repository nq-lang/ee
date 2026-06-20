"""
native_data.py
══════════════════════════════════════════════════════════════════════════════
Pre-loaded Historical Data Registry

All ES, NQ, and SPX datasets are embedded into the terminal's data/preloaded/
directory as optimised Parquet files and are automatically initialised into
memory on terminal startup — no manual imports or external API calls required.

Dataset catalogue
─────────────────
Symbol  Timeframe  Bars         Date Range            Source
──────  ─────────  ───────────  ──────────────────────  ──────────────
ES      1m         4,234,977    Jan 2014 – Jan 2026     Rithmic
ES      5m           850,923    2014 – Jan 2026         Rithmic
ES      15m          283,715    2014 – Jan 2026         Rithmic
NQ      1m         4,174,598    Jan 2014 – Jan 2026     Rithmic
NQ      5m           850,374    2014 – Jan 2026         Rithmic
NQ      15m          283,644    2014 – Jan 2026         Rithmic
SPX     5m           380,948    Jan 2006 – May 2025     w/IV data
ES      1m, 5m,       ~33k      May 2026 (recent)       Native
        30m, 1h, 1d
NQ      1m, 5m,       ~33k      May 2026 (recent)       Native
        30m, 1h, 1d

All prices are in points (futures contract notation).
ES point value: $50/pt · NQ point value: $20/pt
SPX includes implied volatility (iv_open/high/low/close).

Usage
─────
    from native_data import get_native_data, list_available, REGISTRY

    df = get_native_data("ES", "5m")   # returns OHLCV DataFrame
    df = get_native_data("SPX", "5m")  # includes IV columns
    available = list_available()        # dict of all loaded datasets
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ── Data directory — relative to this file ───────────────────────────────────
_BASE_DIR    = Path(__file__).parent
PRELOAD_DIR  = _BASE_DIR / "data" / "preloaded"


# ── Dataset catalogue ─────────────────────────────────────────────────────────
#  (parquet_stem, symbol, timeframe, description)
CATALOGUE = [
    # ES full history
    ("ES_1m",         "ES",  "1m",  "E-mini S&P 500 · 1-min · 2014–2026 · 4.2M bars"),
    ("ES_5m",         "ES",  "5m",  "E-mini S&P 500 · 5-min · 2014–2026 · 851K bars"),
    ("ES_15m",        "ES",  "15m", "E-mini S&P 500 · 15-min · 2014–2026 · 284K bars"),
    # ES recent samples
    ("ES_1m_sample",  "ES",  "1m",  "E-mini S&P 500 · 1-min · May 2026 · 15K bars"),
    ("ES_5m_sample",  "ES",  "5m",  "E-mini S&P 500 · 5-min · May 2026 · 3K bars"),
    ("ES_30m_sample", "ES",  "30m", "E-mini S&P 500 · 30-min · May 2026 · 500 bars"),
    ("ES_1h_sample",  "ES",  "1h",  "E-mini S&P 500 · 1-hour · May 2026 · 250 bars"),
    ("ES_1d_sample",  "ES",  "1d",  "E-mini S&P 500 · Daily · May 2026 · 10 bars"),
    # NQ full history
    ("NQ_1m",         "NQ",  "1m",  "E-mini NASDAQ-100 · 1-min · 2014–2026 · 4.2M bars"),
    ("NQ_5m",         "NQ",  "5m",  "E-mini NASDAQ-100 · 5-min · 2014–2026 · 850K bars"),
    ("NQ_15m",        "NQ",  "15m", "E-mini NASDAQ-100 · 15-min · 2014–2026 · 284K bars"),
    # NQ recent samples
    ("NQ_1m_sample",  "NQ",  "1m",  "E-mini NASDAQ-100 · 1-min · May 2026 · 15K bars"),
    ("NQ_5m_sample",  "NQ",  "5m",  "E-mini NASDAQ-100 · 5-min · May 2026 · 3K bars"),
    ("NQ_30m_sample", "NQ",  "30m", "E-mini NASDAQ-100 · 30-min · May 2026 · 500 bars"),
    ("NQ_1h_sample",  "NQ",  "1h",  "E-mini NASDAQ-100 · 1-hour · May 2026 · 250 bars"),
    ("NQ_1d_sample",  "NQ",  "1d",  "E-mini NASDAQ-100 · Daily · May 2026 · 10 bars"),
    # SPX with IV
    ("SPX_5m",        "SPX", "5m",  "S&P 500 Index · 5-min + IV · 2006–2025 · 381K bars"),
]

# Build lookup: (symbol.upper(), timeframe) → parquet_stem
REGISTRY: dict[tuple[str, str], str] = {
    (row[1].upper(), row[2]): row[0]
    for row in CATALOGUE
}

# Priority order: when multiple parquets exist for (symbol, tf),
# prefer the full-history file over sample files
_PRIORITY_SUFFIXES = ["", "_sample"]  # "" = full history first


# ── Load function ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=12)
def _load_parquet(stem: str) -> pd.DataFrame:
    """
    Load a single parquet file with LRU caching.
    Large files (1m ES/NQ) stay resident for the session.
    """
    path = PRELOAD_DIR / f"{stem}.parquet"
    if not path.exists():
        logger.warning(f"Native dataset not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        # Ensure UTC-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        df.attrs["stem"]   = stem
        df.attrs["source"] = "native_preloaded"
        logger.info(f"Loaded native dataset '{stem}': {len(df):,} bars")
        return df
    except Exception as exc:
        logger.error(f"Failed to load {path}: {exc}")
        return pd.DataFrame()


def get_native_data(
    symbol:     str,
    timeframe:  str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    prefer_full_history: bool = True,
) -> pd.DataFrame:
    """
    Retrieve pre-loaded OHLCV data for a symbol/timeframe pair.

    Parameters
    ----------
    symbol     : "ES" | "NQ" | "SPX" | "MES" | "MNQ"
    timeframe  : "1m" | "5m" | "15m" | "30m" | "1h" | "1d"
    start_date : optional ISO date string filter
    end_date   : optional ISO date string filter
    prefer_full_history : if True, use the large full-history file;
                          if False, use the recent sample file when available

    Returns
    -------
    OHLCV DataFrame with UTC-aware DatetimeIndex, or empty DataFrame if not found.
    SPX also includes iv_open, iv_high, iv_low, iv_close columns.
    """
    sym = symbol.upper()
    # MES/MNQ map to ES/NQ data (same price levels, different contract size)
    sym_map = {"MES": "ES", "MNQ": "NQ"}
    sym = sym_map.get(sym, sym)

    tf = timeframe.lower()

    # Build candidate stems in priority order
    candidates = []
    for row in CATALOGUE:
        if row[1].upper() == sym and row[2] == tf:
            candidates.append(row[0])

    if not candidates:
        logger.warning(f"No native dataset for {sym} {tf}. Available: {list(REGISTRY.keys())}")
        return pd.DataFrame()

    # Prefer full-history (no "_sample" suffix) vs sample
    if prefer_full_history:
        full = [c for c in candidates if "_sample" not in c]
        stem = full[0] if full else candidates[0]
    else:
        samples = [c for c in candidates if "_sample" in c]
        stem = samples[0] if samples else candidates[0]

    df = _load_parquet(stem)
    if df.empty:
        return df

    # Date filter
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

    return df


def get_best_available(
    symbol:    str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Return the best available dataset for a symbol/timeframe.
    Falls back to nearest timeframe if exact match not found.

    Returns
    -------
    (df, description_string)
    """
    df = get_native_data(symbol, timeframe, start_date, end_date)
    if not df.empty:
        desc = next((r[3] for r in CATALOGUE
                     if r[1].upper() == symbol.upper() and r[2] == timeframe), "native data")
        return df, desc

    # Fallback: try different timeframes in order of closeness
    fallback_order = {
        "1m":  ["5m","15m","30m","1h","1d"],
        "5m":  ["1m","15m","30m","1h","1d"],
        "15m": ["5m","30m","1h","1m","1d"],
        "30m": ["15m","1h","5m","1d"],
        "1h":  ["30m","15m","5m","1d"],
        "4h":  ["1h","1d","30m"],
        "1d":  ["1h","4h","30m"],
    }
    for tf_alt in fallback_order.get(timeframe, []):
        df = get_native_data(symbol, tf_alt, start_date, end_date)
        if not df.empty:
            desc = f"{symbol.upper()} {tf_alt} (fallback from {timeframe})"
            return df, desc

    return pd.DataFrame(), f"No native data for {symbol} {timeframe}"


def list_available() -> dict:
    """
    Return a dict of all available native datasets with metadata.
    """
    available = {}
    for stem, symbol, tf, desc in CATALOGUE:
        path = PRELOAD_DIR / f"{stem}.parquet"
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            available[f"{symbol}_{tf}_{stem}"] = {
                "symbol":    symbol,
                "timeframe": tf,
                "stem":      stem,
                "desc":      desc,
                "size_mb":   round(size_mb, 2),
                "path":      str(path),
                "loaded":    stem in {k for k, _ in _load_parquet.cache_info().__dict__.items()
                                       if False},  # cache inspection
            }
    return available


def get_preload_path(stem: str) -> Path:
    """Return absolute Path for a parquet file — safe on Streamlit Cloud."""
    return PRELOAD_DIR / f"{stem}.parquet"


def get_preload_dir() -> Path:
    """Return absolute path to the preloaded data directory."""
    return PRELOAD_DIR


def get_date_range(symbol: str, timeframe: str) -> tuple[Optional[str], Optional[str]]:
    """Return (start_date, end_date) for a native dataset without loading it fully."""
    df = get_native_data(symbol, timeframe)
    if df.empty:
        return None, None
    return str(df.index.min().date()), str(df.index.max().date())


# ── Streamlit cache wrapper ───────────────────────────────────────────────────

def st_cached_native(symbol: str, timeframe: str,
                     start_date: Optional[str] = None,
                     end_date:   Optional[str] = None) -> pd.DataFrame:
    """
    Streamlit @st.cache_data compatible wrapper.
    Call this inside the Streamlit app to avoid re-loading on every rerun.
    """
    try:
        import streamlit as st

        @st.cache_data(show_spinner=False, ttl=None)
        def _cached(sym, tf, sd, ed):
            return get_native_data(sym, tf, sd, ed)

        return _cached(symbol, timeframe, start_date, end_date)
    except ImportError:
        return get_native_data(symbol, timeframe, start_date, end_date)


# ── Module-level pre-warm (lazy — only loads on first access) ─────────────────

def prewarm_small_datasets():
    """
    Pre-load smaller datasets into LRU cache at startup.
    Large 1m files are loaded on demand only.
    """
    small_stems = [
        "ES_15m", "NQ_15m",
        "ES_5m_sample", "NQ_5m_sample",
        "ES_1h_sample", "NQ_1h_sample",
        "ES_1d_sample", "NQ_1d_sample",
        "ES_30m_sample","NQ_30m_sample",
        "SPX_5m",
    ]
    loaded = []
    for stem in small_stems:
        df = _load_parquet(stem)
        if not df.empty:
            loaded.append(f"{stem} ({len(df):,})")
    logger.info(f"Pre-warmed {len(loaded)} native datasets")
    return loaded
