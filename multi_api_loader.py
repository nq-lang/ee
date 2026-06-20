"""
multi_api_loader.py
════════════════════
Unified historical OHLCV pipeline with an intelligent priority fallback chain.

Priority order for intraday data
──────────────────────────────────
1. Tastytrade  — best for futures (/ES /NQ), requires funded account
2. Polygon.io  — excellent intraday for SPY/QQQ, good free tier
3. Alpha Vantage — solid stocks + ETFs, rate-limited (5 calls/min free)
4. Finnhub     — good stocks + crypto, generous free tier
5. yfinance    — always available, limited intraday for futures

Each source returns a normalised OHLCV DataFrame compatible with the
backtest engine.  The router tries each source in order and returns the
first successful non-empty response.

Credentials are loaded from st.secrets (Streamlit Cloud) or env vars.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Credential loader ─────────────────────────────────────────────────────────

def _secret(section: str, key: str, env_fallback: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(section, {}).get(key, os.getenv(env_fallback, ""))
    except Exception:
        return os.getenv(env_fallback, "")


# ── Symbol normalisation maps per source ──────────────────────────────────────

_POLYGON_SYMS = {
    "NQ":  "NQ",        # Polygon futures: NQ (some plans) or I:NQ*1 continuous
    "ES":  "ES",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "MNQ": "MNQ",
    "MES": "MES",
}

_AV_SYMS = {
    "NQ":  "NQ",        # Alpha Vantage: limited futures support
    "ES":  "ES",
    "SPY": "SPY",
    "QQQ": "QQQ",
}

_FINNHUB_SYMS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "NQ":  "CME:NQ1!",  # Finnhub continuous futures symbol
    "ES":  "CME:ES1!",
}

_YFINANCE_SYMS = {
    "NQ":  "NQ=F",
    "ES":  "ES=F",
    "MNQ": "NQ=F",
    "MES": "ES=F",
    "SPY": "SPY",
    "QQQ": "QQQ",
}

# ── Polygon interval map ──────────────────────────────────────────────────────
_POLY_TF = {
    "1m":  ("1",  "minute"),
    "5m":  ("5",  "minute"),
    "15m": ("15", "minute"),
    "30m": ("30", "minute"),
    "1h":  ("1",  "hour"),
    "4h":  ("4",  "hour"),
    "1d":  ("1",  "day"),
}

# ── Alpha Vantage interval map ────────────────────────────────────────────────
_AV_TF_INTRA = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"60min"}


# ══════════════════════════════════════════════════════════════════════════════
# Source 2: Polygon.io
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_polygon(
    symbol:    str,
    timeframe: str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    key = _secret("polygon", "api_key", "POLYGON_API_KEY")
    if not key:
        raise ValueError("Polygon API key not configured")

    mult, span = _POLY_TF.get(timeframe, ("1", "day"))
    sym = _POLYGON_SYMS.get(symbol.upper(), symbol.upper())

    url = f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/{mult}/{span}/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   key,
    }

    all_results = []
    while url:
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"Polygon error: {e}") from e

        results = data.get("results", [])
        all_results.extend(results)

        # Pagination
        next_url = data.get("next_url")
        if next_url:
            url    = next_url
            params = {"apiKey": key}   # next_url already has other params
        else:
            break

        time.sleep(0.12)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()
    df.attrs["source"] = "polygon"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Source 3: Alpha Vantage
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_alpha_vantage(
    symbol:    str,
    timeframe: str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    key = _secret("alpha_vantage", "api_key", "ALPHAVANTAGE_API_KEY")
    if not key:
        raise ValueError("Alpha Vantage API key not configured")

    sym = _AV_SYMS.get(symbol.upper(), symbol.upper())

    if timeframe == "1d":
        # Daily
        url = "https://www.alphavantage.co/query"
        params = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":sym,
                  "outputsize":"full","apikey":key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        ts   = data.get("Time Series (Daily)", {})
        if not ts:
            raise RuntimeError(f"Alpha Vantage returned no daily data for {sym}")
        records = [
            {"timestamp": datetime.fromisoformat(d).replace(tzinfo=timezone.utc),
             "open":  float(v["1. open"]),  "high": float(v["2. high"]),
             "low":   float(v["3. low"]),   "close":float(v["4. close"]),
             "volume":float(v["6. volume"])}
            for d, v in ts.items()
        ]
    else:
        av_tf = _AV_TF_INTRA.get(timeframe, "5min")
        url   = "https://www.alphavantage.co/query"
        params = {"function":"TIME_SERIES_INTRADAY","symbol":sym,"interval":av_tf,
                  "outputsize":"full","extended_hours":"true","apikey":key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        key_name = f"Time Series ({av_tf})"
        ts = data.get(key_name, {})
        if not ts:
            raise RuntimeError(f"Alpha Vantage intraday: no data for {sym} {av_tf}")
        records = [
            {"timestamp": datetime.fromisoformat(d).replace(tzinfo=timezone.utc),
             "open":  float(v["1. open"]),  "high": float(v["2. high"]),
             "low":   float(v["3. low"]),   "close":float(v["4. close"]),
             "volume":float(v["5. volume"])}
            for d, v in ts.items()
        ]

    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    # Filter to requested date range
    s = pd.Timestamp(start_date, tz="UTC")
    e = pd.Timestamp(end_date,   tz="UTC") + timedelta(days=1)
    df = df[(df.index >= s) & (df.index <= e)]
    df.attrs["source"] = "alpha_vantage"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Source 4: Finnhub
# ══════════════════════════════════════════════════════════════════════════════

_FINNHUB_RES = {
    "1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1d":"D","1w":"W",
}

def _fetch_finnhub(
    symbol:    str,
    timeframe: str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    key = _secret("finnhub", "api_key", "FINNHUB_API_KEY")
    if not key:
        raise ValueError("Finnhub API key not configured")

    sym        = _FINNHUB_SYMS.get(symbol.upper(), symbol.upper())
    resolution = _FINNHUB_RES.get(timeframe, "D")
    ts_from    = int(datetime.fromisoformat(start_date).timestamp())
    ts_to      = int(datetime.fromisoformat(end_date).timestamp()) + 86400

    r = requests.get(
        "https://finnhub.io/api/v1/stock/candle",
        params={"symbol":sym,"resolution":resolution,"from":ts_from,"to":ts_to,"token":key},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    if data.get("s") != "ok" or not data.get("t"):
        raise RuntimeError(f"Finnhub: no data for {sym} {timeframe} — status: {data.get('s')}")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["t"], unit="s", utc=True),
        "open":  data["o"], "high": data["h"], "low": data["l"],
        "close": data["c"], "volume": data["v"],
    }).set_index("timestamp").sort_index()
    df.attrs["source"] = "finnhub"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Source 5: yfinance (always-available fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_yfinance(
    symbol:    str,
    timeframe: str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    import yfinance as yf
    yf_sym = _YFINANCE_SYMS.get(symbol.upper(), symbol.upper())
    t = yf.Ticker(yf_sym)
    df = t.history(start=start_date, end=end_date, interval=timeframe, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"yfinance: no data for {yf_sym} {timeframe}")
    df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df.index = pd.to_datetime(df.index)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df[["open","high","low","close","volume"]].dropna(subset=["close"])
    df.attrs["source"] = "yfinance"
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Master router — priority chain with automatic fallback
# ══════════════════════════════════════════════════════════════════════════════

class MultiAPILoader:
    """
    Tries each data source in priority order, returns the first successful
    non-empty OHLCV DataFrame.  All source errors are captured and included
    in the returned metadata dict for display in the terminal.
    """

    def __init__(self, sources: Optional[list[str]] = None):
        """
        sources : ordered list of sources to try.
                  Defaults to full priority chain.
        """
        self.sources = sources or [
            "tastytrade", "polygon", "alpha_vantage", "finnhub", "yfinance"
        ]

    def fetch(
        self,
        symbol:     str,
        timeframe:  str  = "1d",
        start_date: str  = "",
        end_date:   str  = "",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Fetch OHLCV with fallback.

        Returns
        -------
        (df, report)
            df     : OHLCV DataFrame (empty if all sources failed)
            report : dict with 'source', 'tried', 'errors', 'bars'
        """
        if not end_date:
            end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")

        tried  = []
        errors = {}

        for source in self.sources:
            tried.append(source)
            try:
                df = self._call(source, symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    return df, {
                        "source":     source,
                        "tried":      tried,
                        "errors":     errors,
                        "bars":       len(df),
                        "start":      str(df.index.min()),
                        "end":        str(df.index.max()),
                        "symbol":     symbol,
                        "timeframe":  timeframe,
                    }
                else:
                    errors[source] = "returned empty DataFrame"
            except Exception as exc:
                errors[source] = str(exc)
                logger.warning(f"[{source}] {symbol} {timeframe}: {exc}")

        return pd.DataFrame(), {
            "source":    None,
            "tried":     tried,
            "errors":    errors,
            "bars":      0,
            "symbol":    symbol,
            "timeframe": timeframe,
        }

    @staticmethod
    def _call(
        source:     str,
        symbol:     str,
        timeframe:  str,
        start_date: str,
        end_date:   str,
    ) -> Optional[pd.DataFrame]:
        if source == "tastytrade":
            from tastytrade_loader import fetch_tastytrade_ohlcv
            return fetch_tastytrade_ohlcv(symbol, timeframe, start_date, end_date)
        elif source == "polygon":
            return _fetch_polygon(symbol, timeframe, start_date, end_date)
        elif source == "alpha_vantage":
            return _fetch_alpha_vantage(symbol, timeframe, start_date, end_date)
        elif source == "finnhub":
            return _fetch_finnhub(symbol, timeframe, start_date, end_date)
        elif source == "yfinance":
            return _fetch_yfinance(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {source}")

    def status(self) -> dict[str, bool]:
        """Quick connectivity check for each configured source."""
        results = {}
        for s in self.sources:
            try:
                if s == "tastytrade":
                    from tastytrade_loader import get_client
                    results[s] = get_client().ping()
                elif s == "polygon":
                    key = _secret("polygon", "api_key", "POLYGON_API_KEY")
                    r   = requests.get("https://api.polygon.io/v2/aggs/ticker/SPY/prev",
                                       params={"apiKey": key}, timeout=8)
                    results[s] = r.status_code == 200
                elif s == "alpha_vantage":
                    key = _secret("alpha_vantage", "api_key", "ALPHAVANTAGE_API_KEY")
                    results[s] = bool(key)
                elif s == "finnhub":
                    key = _secret("finnhub", "api_key", "FINNHUB_API_KEY")
                    results[s] = bool(key)
                elif s == "yfinance":
                    results[s] = True
                else:
                    results[s] = False
            except Exception:
                results[s] = False
        return results


# ── Module-level singleton ────────────────────────────────────────────────────
_loader: Optional[MultiAPILoader] = None

def get_loader(sources: Optional[list[str]] = None) -> MultiAPILoader:
    global _loader
    if _loader is None or sources:
        _loader = MultiAPILoader(sources)
    return _loader


def fetch_ohlcv(
    symbol:     str,
    timeframe:  str  = "1d",
    start_date: str  = "",
    end_date:   str  = "",
    sources:    Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """Top-level convenience function used throughout the terminal."""
    return get_loader(sources).fetch(symbol, timeframe, start_date, end_date)
