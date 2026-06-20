"""
tastytrade_loader.py — Tastytrade OAuth2 + historical OHLCV pipeline.

Auth priority:
  1. Refresh token (long-lived, no password needed)
  2. OAuth2 client_credentials
  3. Session login fallback

Credentials from st.secrets["tastytrade"] or environment variables.
"""
from __future__ import annotations
import logging, time
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROD_BASE    = "https://api.tastytrade.com"
SANDBOX_BASE = "https://api.cert.tastyworks.com"

TF_MAP = {
    "1m":"Minute","5m":"FiveMinute","15m":"FifteenMinute",
    "30m":"ThirtyMinute","1h":"Hour","4h":"FourHour","1d":"Day","1w":"Week",
}
SYMBOL_MAP = {
    "NQ":"/NQ","ES":"/ES","MNQ":"/MNQ","MES":"/MES","SPY":"SPY","QQQ":"QQQ","SPX":"SPX",
}

def _load_credentials() -> dict:
    try:
        import streamlit as st
        sec = st.secrets.get("tastytrade", {})
        if sec.get("client_id"):
            return {
                "username":      sec.get("username",""),
                "client_id":     sec.get("client_id",""),
                "client_secret": sec.get("client_secret",""),
                "refresh_token": sec.get("refresh_token",""),
                "base_url":      sec.get("base_url", PROD_BASE),
            }
    except Exception:
        pass
    import os
    return {
        "username":      os.getenv("TT_USERNAME",""),
        "client_id":     os.getenv("TT_CLIENT_ID",""),
        "client_secret": os.getenv("TT_CLIENT_SECRET",""),
        "refresh_token": os.getenv("TT_REFRESH_TOKEN",""),
        "base_url":      os.getenv("TT_BASE_URL", PROD_BASE),
    }


class TastytradeClient:
    def __init__(self, sandbox: bool = False):
        creds = _load_credentials()
        self._base          = SANDBOX_BASE if sandbox else creds.get("base_url", PROD_BASE)
        self._client_id     = creds.get("client_id","")
        self._secret        = creds.get("client_secret","")
        self._username      = creds.get("username","")
        self._refresh_token = creds.get("refresh_token","")
        self._token: Optional[str] = None
        self._token_exp: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept":       "application/json",
        })

    def _valid(self) -> bool:
        return self._token is not None and time.time() < self._token_exp - 60

    def authenticate(self) -> bool:
        if self._valid(): return True

        # ── 1. Refresh token (preferred — long-lived, no password) ───────────
        if self._refresh_token:
            try:
                r = self._session.post(f"{self._base}/oauth/token", json={
                    "grant_type":    "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id":     self._client_id,
                }, timeout=15)
                if r.status_code == 200:
                    d = r.json()
                    self._token     = d.get("access_token") or d.get("session-token","")
                    self._token_exp = time.time() + d.get("expires_in", 86400)
                    self._session.headers["Authorization"] = self._token
                    logger.info("Tastytrade: refresh_token auth OK")
                    return True
                else:
                    logger.warning(f"Tastytrade refresh_token HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                logger.warning(f"Tastytrade refresh_token: {e}")

        # ── 2. OAuth2 client_credentials ─────────────────────────────────────
        if self._client_id and self._secret:
            try:
                r = self._session.post(f"{self._base}/oauth/token", json={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._secret,
                }, timeout=15)
                if r.status_code == 200:
                    d = r.json()
                    self._token     = d.get("access_token") or d.get("session-token","")
                    self._token_exp = time.time() + d.get("expires_in", 86400)
                    self._session.headers["Authorization"] = self._token
                    logger.info("Tastytrade: client_credentials auth OK")
                    return True
            except Exception as e:
                logger.warning(f"Tastytrade client_credentials: {e}")

        # ── 3. Session login fallback ─────────────────────────────────────────
        if self._username:
            try:
                r = self._session.post(f"{self._base}/sessions",
                    json={"login": self._username, "remember-me": True}, timeout=15)
                if r.status_code == 201:
                    d = r.json().get("data",{})
                    self._token     = d.get("session-token","")
                    self._token_exp = time.time() + 86400
                    self._session.headers["Authorization"] = self._token
                    logger.info("Tastytrade: session auth OK")
                    return True
            except Exception as e:
                logger.warning(f"Tastytrade session: {e}")

        logger.error("Tastytrade: all auth methods failed")
        return False

    def get_history(self, symbol:str, timeframe:str="1m", bar_count:int=500,
                    end_time:Optional[datetime]=None) -> pd.DataFrame:
        if not self.authenticate():
            raise RuntimeError("Tastytrade auth failed — check st.secrets['tastytrade']")
        tt_sym = SYMBOL_MAP.get(symbol.upper(), symbol)
        tt_tf  = TF_MAP.get(timeframe, "Minute")
        end_dt = end_time or datetime.now(timezone.utc)
        params = {"timeframe": tt_tf, "bar-count": min(bar_count, 2000),
                  "end-time":  end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")}
        url = f"{self._base}/market-data/history/{requests.utils.quote(tt_sym, safe='')}"
        try:
            r = self._session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return self._parse(r.json(), symbol)
        except requests.HTTPError as e:
            raise RuntimeError(f"Tastytrade HTTP {e.response.status_code} for {tt_sym}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Tastytrade fetch: {e}") from e

    def get_history_range(self, symbol:str, timeframe:str,
                          start_date:str, end_date:str) -> pd.DataFrame:
        end_dt   = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        frames, cursor = [], end_dt
        for _ in range(25):
            df = self.get_history(symbol, timeframe, 2000, cursor)
            if df.empty: break
            frames.append(df)
            earliest = df.index.min()
            if pd.Timestamp(earliest) <= pd.Timestamp(start_dt): break
            cursor = earliest.to_pydatetime() - timedelta(seconds=1)
            time.sleep(0.3)
        if not frames: return pd.DataFrame()
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out[out.index >= pd.Timestamp(start_dt, tz="UTC")]

    @staticmethod
    def _parse(raw: dict, symbol: str) -> pd.DataFrame:
        items = (raw.get("data",{}).get("items") or raw.get("data",{}).get("candles")
                 or raw.get("candles") or raw.get("items") or [])
        if not items: return pd.DataFrame()
        recs = []
        for c in items:
            ts = c.get("time") or c.get("timestamp") or c.get("dateTime")
            if ts is None: continue
            try:
                dt = (datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                      if isinstance(ts,(int,float))
                      else datetime.fromisoformat(str(ts).replace("Z","+00:00")))
            except Exception: continue
            recs.append({"timestamp":dt,"open":float(c.get("open",0)),
                         "high":float(c.get("high",0)),"low":float(c.get("low",0)),
                         "close":float(c.get("close",0)),"volume":float(c.get("volume",0))})
        if not recs: return pd.DataFrame()
        df = pd.DataFrame(recs).set_index("timestamp").sort_index()
        df.attrs.update({"source":"tastytrade","symbol":symbol})
        return df[df["close"]>0]

    def ping(self) -> bool:
        return self.authenticate()

    def get_accounts(self) -> list:
        if not self.authenticate(): return []
        try:
            r = self._session.get(f"{self._base}/customers/me/accounts", timeout=10)
            return r.json().get("data",{}).get("items",[])
        except Exception: return []


_client: Optional[TastytradeClient] = None

def get_client() -> TastytradeClient:
    global _client
    if _client is None: _client = TastytradeClient()
    return _client

def fetch_tastytrade_ohlcv(symbol:str, timeframe:str="1m",
                            start_date:Optional[str]=None, end_date:Optional[str]=None,
                            bar_count:int=500) -> pd.DataFrame:
    c = get_client()
    if start_date and end_date:
        return c.get_history_range(symbol, timeframe, start_date, end_date)
    return c.get_history(symbol, timeframe, bar_count=bar_count)
