"""
data_loader.py — Free historical market data pipeline.

Futures  : ES=F / NQ=F via yfinance (daily); SPY/QQQ proxy for intraday.
Options  : yfinance .option_chain() for current chains; local .parquet cache
           for historical EOD snapshots.  All data normalised to canonical
           schemas before being handed to the C++ binary or backtest engine.

Canonical schemas
─────────────────
Futures  : timestamp | open | high | low | close | volume
Options  : timestamp | underlying_price | expiry | strike | option_type |
           bid | ask | iv | delta | gamma | theta | vega | volume | open_interest
"""

from __future__ import annotations

import os
import io
import hashlib
import logging
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from config import INSTRUMENTS, OPTIONS_UNDERLYINGS

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ─── Cache directory layout ──────────────────────────────────────────────────
DATA_DIR    = Path("data")
RAW_DIR     = DATA_DIR / "raw"
PROC_DIR    = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

for _d in (RAW_DIR, PROC_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _cache_path(prefix: str, key: str, ext: str = ".parquet") -> Path:
    """Deterministic local path for a cached dataset."""
    safe = hashlib.md5(key.encode()).hexdigest()[:12]
    return PROC_DIR / f"{prefix}_{safe}{ext}"


def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, path, compression="snappy")


def _parquet_to_df(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _is_stale(path: Path, ttl_hours: int = 12) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=ttl_hours)


# ────────────────────────────────────────────────────────────────────────────
# Futures data
# ────────────────────────────────────────────────────────────────────────────

def get_futures_data(
    ticker: str,
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    use_cache:  bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a futures instrument.

    Parameters
    ----------
    ticker     : "NQ", "ES", "MNQ", or "MES"
    timeframe  : yfinance interval string — "1d","1h","30m","15m","5m","1m"
    start_date : ISO date string or None (defaults to 2 years back)
    end_date   : ISO date string or None (defaults to today)
    use_cache  : if True, return cached parquet if fresh

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume]
    indexed by a UTC-aware DatetimeIndex named 'timestamp'.

    Notes
    -----
    yfinance limits intraday history:
      - ≤ 60 days for intervals < 1d
      - ≤  7 days for 1m bars
    When intraday is requested the proxy ETF (SPY/QQQ) is used and prices
    are scaled to approximate futures-point magnitude.
    """
    instrument = ticker.upper()
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{ticker}'. Choose from {list(INSTRUMENTS)}")

    spec = INSTRUMENTS[instrument]
    end   = end_date   or datetime.today().strftime("%Y-%m-%d")
    start = start_date or (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    cache_key = f"{instrument}_{timeframe}_{start}_{end}"
    cache_file = _cache_path("futures", cache_key)

    if use_cache and not _is_stale(cache_file, ttl_hours=4):
        logger.info(f"Loading futures cache: {cache_file}")
        return _parquet_to_df(cache_file)

    # ── Choose ticker & build the appropriate request ────────────────────────
    intraday = timeframe not in ("1d", "1wk", "1mo")

    if intraday:
        # yfinance intraday for continuous futures is unreliable;
        # use the proxy ETF and scale to approximate futures prices.
        yf_ticker = spec["yf_proxy"]
        logger.info(
            f"Intraday requested for {instrument}; fetching {yf_ticker} as proxy. "
            "Prices will be scaled to approximate futures levels."
        )
        df = _fetch_yf(yf_ticker, timeframe, start, end)
        df = _scale_proxy_to_futures(df, instrument)
    else:
        yf_ticker = spec["yf_daily"]
        df = _fetch_yf(yf_ticker, timeframe, start, end)

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {yf_ticker} "
            f"(interval={timeframe}, {start}→{end}). "
            "Check your date range or network connection."
        )

    df = _normalise_ohlcv(df)
    df.attrs["instrument"]    = instrument
    df.attrs["yf_ticker"]     = yf_ticker
    df.attrs["is_proxy"]      = intraday
    df.attrs["timeframe"]     = timeframe
    df.attrs["point_value"]   = spec["point_value"]
    df.attrs["tick_value"]    = spec["tick_value"]

    _df_to_parquet(df, cache_file)
    return df


def _fetch_yf(ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
    """Raw yfinance download, returns DataFrame with OHLCV columns."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, interval=interval, auto_adjust=True)
    except Exception as exc:
        raise RuntimeError(f"yfinance error for {ticker}: {exc}") from exc
    return df


def _scale_proxy_to_futures(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Linear-scale ETF prices to approximate futures-point ranges.

    SPY ≈ ES / 10    →  ES ≈ SPY × 10
    QQQ ≈ NQ / 40    →  NQ ≈ QQQ × 40

    These ratios are approximate and change over time; for real backtesting
    use a proper continuous contract from a paid vendor (Databento, Rithmic…).
    """
    scale = {"NQ": 40.0, "MNQ": 40.0, "ES": 10.0, "MES": 10.0}
    factor = scale.get(instrument, 1.0)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] * factor
    return df


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, ensure UTC index, remove NA/zeros."""
    rename = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "Datetime": "timestamp", "Date": "timestamp",
    }
    df = df.rename(columns=rename)

    # Ensure index is the timestamp
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index.name = "timestamp"

    # Convert to UTC-aware
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ────────────────────────────────────────────────────────────────────────────
# Options data
# ────────────────────────────────────────────────────────────────────────────

def get_options_chain(
    underlying:  str,
    trade_date:  Optional[str] = None,
    dte_filter:  Optional[list[int]] = None,
    use_cache:   bool = True,
) -> pd.DataFrame:
    """
    Retrieve an options chain for SPY or QQQ.

    Strategy
    --------
    1. Check for a locally cached .parquet file that was pre-downloaded from
       Kaggle/GitHub historical EOD datasets (manual one-time setup; the function
       will guide users if the file is absent).
    2. Fall back to yfinance `.option_chain()` to fetch the current/near-term
       live chain snapshot.  This gives today's chain only — useful for
       testing the data shape and pipeline plumbing.

    Parameters
    ----------
    underlying : "SPY" or "QQQ"
    trade_date : "YYYY-MM-DD" — if None, use today's live chain
    dte_filter : list of DTE integers to keep, e.g. [0, 1, 2] for 0-2 DTE.
                 If None, all expirations are returned.

    Returns
    -------
    DataFrame with canonical options schema.
    """
    underlying = underlying.upper()
    if underlying not in OPTIONS_UNDERLYINGS:
        raise ValueError(f"Options data only for {OPTIONS_UNDERLYINGS}")

    target_date = trade_date or datetime.today().strftime("%Y-%m-%d")

    # ── 1. Try local parquet historical archive ──────────────────────────────
    historical_archive = PROC_DIR / f"options_eod_{underlying.lower()}.parquet"
    if historical_archive.exists():
        try:
            df = _load_historical_options(historical_archive, target_date, dte_filter)
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning(f"Historical parquet read failed: {exc}; falling back to API")

    # ── 2. Live yfinance fallback ────────────────────────────────────────────
    cache_key  = f"{underlying}_{target_date}"
    cache_file = _cache_path("options", cache_key)

    if use_cache and not _is_stale(cache_file, ttl_hours=1):
        return _parquet_to_df(cache_file)

    df = _fetch_yf_options(underlying, dte_filter)
    if df.empty:
        raise RuntimeError(
            f"No options data available for {underlying} on {target_date}. "
            "For historical backtesting, download the free EOD dataset from "
            "https://github.com/optionstrat and place it at "
            f"{historical_archive}. "
            "The yfinance fallback only provides today's live chain."
        )

    _df_to_parquet(df, cache_file)
    return df


def _load_historical_options(
    archive: Path,
    target_date: str,
    dte_filter: Optional[list[int]],
) -> pd.DataFrame:
    """Read a slice of the historical EOD options parquet archive."""
    t_date = pd.Timestamp(target_date)

    # Read only the relevant date partition to save memory
    # (assumes the parquet file has a 'quote_date' column)
    df = pq.read_table(
        archive,
        filters=[("quote_date", "=", t_date.date())],
    ).to_pandas()

    if df.empty:
        return df

    df = _normalise_options(df)

    if dte_filter is not None:
        ref = pd.Timestamp(target_date)
        df["dte"] = (pd.to_datetime(df["expiry"]) - ref).dt.days
        df = df[df["dte"].isin(dte_filter)]

    return df


def _fetch_yf_options(
    underlying: str,
    dte_filter: Optional[list[int]],
) -> pd.DataFrame:
    """Fetch live options chain via yfinance."""
    try:
        t = yf.Ticker(underlying)
        underlying_price = t.fast_info.get("lastPrice", np.nan)
        expiries = t.options  # tuple of expiry date strings

        today = datetime.today().date()
        frames = []

        for exp_str in expiries:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte_filter and dte not in dte_filter:
                continue
            if dte < 0:
                continue

            try:
                chain = t.option_chain(exp_str)
            except Exception:
                continue

            for side, df_side in (("call", chain.calls), ("put", chain.puts)):
                if df_side is None or df_side.empty:
                    continue
                df_side = df_side.copy()
                df_side["option_type"]      = side
                df_side["expiry"]           = exp_str
                df_side["underlying_price"] = underlying_price
                df_side["timestamp"]        = pd.Timestamp.utcnow().normalize()
                frames.append(df_side)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return _normalise_options(df)

    except Exception as exc:
        logger.error(f"yfinance options error for {underlying}: {exc}")
        return pd.DataFrame()


def _normalise_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map varied source column names to the canonical options schema:
    timestamp | underlying_price | expiry | strike | option_type |
    bid | ask | iv | delta | gamma | theta | vega | volume | open_interest
    """
    col_map = {
        # yfinance names
        "contractSymbol":   "contract_symbol",
        "strike":           "strike",
        "lastPrice":        "last",
        "bid":              "bid",
        "ask":              "ask",
        "impliedVolatility":"iv",
        "inTheMoney":       "itm",
        "volume":           "volume",
        "openInterest":     "open_interest",
        # Historical dataset names
        "quote_date":       "timestamp",
        "expiration":       "expiry",
        "option_type":      "option_type",
        "underlying_mid":   "underlying_price",
        "delta":            "delta",
        "gamma":            "gamma",
        "theta":            "theta",
        "vega":             "vega",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure canonical columns exist (fill with NaN if absent)
    for col in ("timestamp", "underlying_price", "expiry", "strike", "option_type",
                "bid", "ask", "iv", "delta", "gamma", "theta", "vega",
                "volume", "open_interest"):
        if col not in df.columns:
            df[col] = np.nan

    numeric = ["strike", "bid", "ask", "iv", "delta", "gamma",
               "theta", "vega", "volume", "open_interest", "underlying_price"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["option_type"] = df["option_type"].str.lower().str.strip()
    df["expiry"]      = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")

    return df[[
        "timestamp", "underlying_price", "expiry", "strike", "option_type",
        "bid", "ask", "iv", "delta", "gamma", "theta", "vega",
        "volume", "open_interest",
    ]].reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────────
# CSV upload processing (used by the Data Loader section in the app)
# ────────────────────────────────────────────────────────────────────────────

def load_csv_upload(
    file_obj,
    column_map: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Parse an uploaded CSV file into a normalised OHLCV DataFrame.

    Returns
    -------
    (df, report) where report contains data-quality metadata.
    """
    try:
        raw = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if raw.empty:
        raise ValueError("The uploaded CSV file is empty.")

    report: dict = {
        "raw_rows":       len(raw),
        "raw_columns":    list(raw.columns),
        "duplicate_rows": 0,
        "anomalies":      [],
        "missing_bars":   [],
        "gaps_detected":  0,
    }

    # ── Column mapping ───────────────────────────────────────────────────────
    if column_map:
        raw = raw.rename(columns=column_map)
    else:
        raw = _auto_map_columns(raw)

    # ── Build a proper DatetimeIndex ─────────────────────────────────────────
    raw = _parse_datetime_index(raw)

    # ── Duplicate removal ────────────────────────────────────────────────────
    before = len(raw)
    raw = raw[~raw.index.duplicated(keep="last")]
    report["duplicate_rows"] = before - len(raw)

    # ── Sort chronologically ─────────────────────────────────────────────────
    raw = raw.sort_index()

    # ── Ensure numeric OHLCV ─────────────────────────────────────────────────
    for col in ("open", "high", "low", "close"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["close"])

    # ── Price anomaly detection ───────────────────────────────────────────────
    anomalies = raw[(raw["high"] < raw["low"]) |
                    (raw["close"] > raw["high"]) |
                    (raw["close"] < raw["low"])]
    if len(anomalies) > 0:
        report["anomalies"] = anomalies.index.strftime("%Y-%m-%d %H:%M").tolist()

    # ── Gap detection ────────────────────────────────────────────────────────
    if len(raw) > 1:
        gaps = _detect_gaps(raw)
        report["gaps_detected"] = len(gaps)
        report["missing_bars"]  = gaps[:20]  # cap display at 20

    report["final_rows"]   = len(raw)
    report["date_start"]   = str(raw.index.min())
    report["date_end"]     = str(raw.index.max())
    report["interval"]     = _detect_interval(raw)
    report["has_weekends"] = bool(raw.index.dayofweek.isin([5, 6]).any())

    return raw, report


def _auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Heuristically map column names to open/high/low/close/volume."""
    lower = {c: c.lower().strip() for c in df.columns}
    rename = {}
    target_map = {
        "open":   ("open", "o", "open price"),
        "high":   ("high", "h", "high price", "max"),
        "low":    ("low", "l", "low price", "min"),
        "close":  ("close", "c", "last", "settle", "close price"),
        "volume": ("volume", "vol", "v", "qty"),
    }
    for canonical, variants in target_map.items():
        for original, mapped in lower.items():
            if mapped in variants and canonical not in rename.values():
                rename[original] = canonical
                break
    return df.rename(columns=rename)


def _parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Find date/time columns and build a DatetimeIndex."""
    dt_candidates = [c for c in df.columns if c.lower() in
                     ("date", "time", "datetime", "timestamp", "date time",
                      "bar time", "bar_time")]

    if len(dt_candidates) >= 2:
        # Separate date and time columns — combine them
        date_col = next((c for c in dt_candidates if "date" in c.lower()), dt_candidates[0])
        time_col = next((c for c in dt_candidates if "time" in c.lower() and c != date_col),
                        None)
        if time_col:
            combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
            df.index = pd.to_datetime(combined, infer_datetime_format=True, errors="coerce")
            df = df.drop(columns=[date_col, time_col])
        else:
            df.index = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
            df = df.drop(columns=[date_col])
    elif len(dt_candidates) == 1:
        df.index = pd.to_datetime(df[dt_candidates[0]], infer_datetime_format=True, errors="coerce")
        df = df.drop(columns=[dt_candidates[0]])
    else:
        # Try the first column
        try:
            df.index = pd.to_datetime(df.iloc[:, 0], infer_datetime_format=True, errors="coerce")
            df = df.iloc[:, 1:]
        except Exception:
            df.index = pd.RangeIndex(len(df))

    df.index.name = "timestamp"
    df = df.dropna(how="all")
    return df


def _detect_interval(df: pd.DataFrame) -> str:
    """Infer bar interval from median time delta."""
    if len(df) < 2:
        return "unknown"
    deltas = df.index.to_series().diff().dropna()
    med = deltas.median().total_seconds() / 60  # minutes
    if med < 2:     return "1m"
    if med < 7:     return "5m"
    if med < 20:    return "15m"
    if med < 45:    return "30m"
    if med < 120:   return "1h"
    if med < 600:   return "4h"
    return "1d"


def _detect_gaps(df: pd.DataFrame) -> list[str]:
    """Return list of gap timestamps (> 2× median bar interval)."""
    if len(df) < 2:
        return []
    deltas = df.index.to_series().diff().dropna()
    med    = deltas.median()
    threshold = med * 2.5
    gaps = deltas[deltas > threshold]
    return [str(ts) for ts in gaps.index[:50]]


# ────────────────────────────────────────────────────────────────────────────
# Instrument auto-detection
# ────────────────────────────────────────────────────────────────────────────

def auto_detect_instrument(df: pd.DataFrame, filename: str = "") -> dict:
    """
    Determine whether the data is NQ, ES, MNQ, or MES futures.

    Returns
    -------
    dict with keys:
      detected   : str | None  — best guess or None
      confidence : "high" | "medium" | "low"
      method     : str          — which heuristic matched
      message    : str          — human-readable explanation
    """
    median_close = float(df["close"].median()) if "close" in df.columns else np.nan

    # ── Filename keyword scan ────────────────────────────────────────────────
    fn_lower = filename.lower()
    fn_hit = None
    for sym in ("mnq", "mes", "nq", "es"):
        if sym in fn_lower:
            fn_hit = sym.upper()
            break

    # ── Price range scan ─────────────────────────────────────────────────────
    price_hit = None
    if not np.isnan(median_close):
        for sym, spec in INSTRUMENTS.items():
            if spec["price_low"] <= median_close <= spec["price_high"]:
                price_hit = sym
                break

    # ── Column header scan ───────────────────────────────────────────────────
    col_str = " ".join(df.columns).lower()
    col_hit = None
    for sym in ("mnq", "mes", "nq", "es"):
        if sym in col_str:
            col_hit = sym.upper()
            break

    # ── Reconcile ────────────────────────────────────────────────────────────
    hits = [h for h in (fn_hit, price_hit, col_hit) if h is not None]
    unique = list(dict.fromkeys(hits))  # preserve order, deduplicate

    if len(unique) == 1:
        return dict(
            detected=unique[0],
            confidence="high",
            method=f"filename={fn_hit}, price={price_hit}, columns={col_hit}",
            message=f"✅ CSV identified as **{unique[0]}** futures data "
                    f"(median close: {median_close:,.1f}).",
        )
    elif len(unique) > 1:
        return dict(
            detected=unique[0],
            confidence="medium",
            method="conflicting signals",
            message=f"⚠️ Ambiguous signals ({unique}). Best guess: **{unique[0]}**. "
                    "Please confirm the instrument selection.",
        )
    else:
        return dict(
            detected=None,
            confidence="low",
            method="none",
            message="⚠️ Could not auto-detect instrument. "
                    "Please select manually in the Configuration panel.",
        )


# ────────────────────────────────────────────────────────────────────────────
# Write normalised data for C++ consumption
# ────────────────────────────────────────────────────────────────────────────

def write_normalised_for_cpp(
    df: pd.DataFrame,
    output_path: str | Path,
    schema: Literal["futures", "options"] = "futures",
) -> Path:
    """
    Write a normalised CSV at the path expected by the C++ binary.

    The C++ parser expects a clean, header-labelled CSV with no NaN values
    and an ISO-8601 timestamp column.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    out.columns = out.columns.str.lower()
    out = out.fillna(0)
    out.to_csv(path, index=False, float_format="%.6f")
    return path
