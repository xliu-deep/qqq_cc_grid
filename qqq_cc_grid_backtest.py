"""
QQQ Covered Call Grid Search Backtest
======================================
Period: January 2021 – December 2025 (60 months)

Finds the optimal (DTE, Delta, Profit_Taking, Stop_Loss) combination that
maximizes Sharpe Ratio while keeping "called away" frequency low.

Strategy:
  1. Buy QQQ on 2021-01-04 with $100,000
  2. Sell 1 OTM Covered Call when the previous position closes
  3. Grid search across DTE × Delta × Profit Taking × Stop Loss
  4. When called away: cash-settle and re-buy at market price

Tax Model:
  - CC premiums → ordinary income (37% marginal)
  - Stock LTCG → 20% + 3.8% NIIT
  - QQQ dividends → ordinary income
  - QQQI comparison → Section 1256 (60/40 rule)

Data Sources:
  - Stock prices & dividends: Yahoo Finance (yfinance)
  - Option pricing: Black-Scholes with historical volatility + skew
  - ThetaData Terminal API (localhost:25503) for calibration when available
"""

import os
import json
import time
import math
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from itertools import product
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
THETADATA_URL = "http://127.0.0.1:25503"
SYMBOL = "QQQ"
QQQI_SYMBOL = "QQQI"

INITIAL_INVESTMENT = 100_000
START_DATE = date(2021, 1, 4)
END_DATE = date(2025, 12, 31)
RISK_FREE_RATE = 0.04
SLIPPAGE_PCT = 0.01
COMMISSION_PER_CONTRACT = 0.65

DTE_GRID = [7, 14, 21, 30, 45]
DELTA_GRID = [0.10, 0.15, 0.20, 0.25]
PROFIT_TAKING_GRID = [None, 0.50, 0.60]
STOP_LOSS_GRID = [None, 2.0, 3.0]

ORDINARY_TAX_RATE = 0.37
NIIT_RATE = 0.038
LTCG_TAX_RATE = 0.20 + NIIT_RATE
SECTION_1256_RATE = 0.60 * LTCG_TAX_RATE + 0.40 * ORDINARY_TAX_RATE
QQQ_DIVIDEND_TAX_RATE = ORDINARY_TAX_RATE

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(WORK_DIR, "qqq_cc_grid_cache.json")
CACHE_FILE_BACKUP = "/tmp/qqq_cc_grid_cache.json"
REPORT_FILE = os.path.join(WORK_DIR, "QQQ_CC_Grid_Search_Report.html")


# ═══════════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class BlackScholesEngine:
    """European option pricing with volatility skew."""

    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def call_delta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 1.0 if S >= K else 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d1))

    @staticmethod
    def call_theta(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
        return theta / 365.0

    @staticmethod
    def call_vega(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * math.sqrt(T) * norm.pdf(d1) / 100.0

    @classmethod
    def strike_for_delta(cls, S, target_delta, T, r, sigma):
        if T <= 0 or sigma <= 0 or target_delta <= 0 or target_delta >= 1:
            return S * 1.02
        d1_target = norm.ppf(target_delta)
        K_raw = S * math.exp(
            (r + 0.5 * sigma ** 2) * T - d1_target * sigma * math.sqrt(T)
        )
        return cls._round_strike(S, K_raw)

    @staticmethod
    def _round_strike(S, K):
        if S < 50:
            return round(K * 2) / 2
        elif S < 200:
            return round(K)
        elif S < 500:
            return round(K / 2) * 2
        else:
            return round(K / 5) * 5

    @classmethod
    def price_with_skew(cls, S, K, T, r, base_sigma):
        moneyness = K / S
        skew_adj = base_sigma * 0.12 * (1.0 - moneyness)
        adj_sigma = max(base_sigma + skew_adj, 0.05)
        return cls.call_price(S, K, T, r, adj_sigma), adj_sigma


# ═══════════════════════════════════════════════════════════════════════════════
#  THETADATA API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
SESSION = requests.Session()


def api_get(url, retries=2, timeout=15):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            txt = r.text.strip()
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if "<html>" in txt.lower():
                return None
            return txt
        except Exception:
            time.sleep(3)
    return None


def parse_csv(text):
    if not text:
        return []
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    hdr = [h.strip().strip('"') for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        vals = [v.strip().strip('"') for v in line.split(",")]
        if len(vals) >= len(hdr):
            rows.append(dict(zip(hdr, vals)))
    return rows


def check_thetadata():
    try:
        r = SESSION.get(
            f"{THETADATA_URL}/v3/option/list/expirations?symbol=SPY", timeout=5
        )
        return r.status_code == 200 and "<html>" not in r.text.lower()
    except Exception:
        return False


def fetch_call_chain_greeks(expiration, query_date):
    url = (
        f"{THETADATA_URL}/v3/option/history/greeks/eod"
        f"?symbol={SYMBOL}&expiration={expiration}&right=C"
        f"&start_date={query_date}&end_date={query_date}"
    )
    time.sleep(0.3)
    txt = api_get(url)
    if not txt:
        return []
    rows = parse_csv(txt)
    chain = []
    for r in rows:
        try:
            strike_raw = float(r.get("strike", 0))
            strike = strike_raw / 1000.0 if strike_raw > 5000 else strike_raw
            delta = float(r.get("delta", 0))
            bid = float(r.get("bid", 0))
            ask = float(r.get("ask", 0))
            iv = float(r.get("implied_vol", r.get("iv", 0)))
            close_px = float(r.get("close", 0))
            underlying = float(r.get("underlying_price", 0))
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else close_px
            if 0.01 < delta < 0.90 and mid > 0.01 and strike > 0:
                chain.append(dict(
                    strike=strike, delta=delta, bid=bid, ask=ask,
                    mid=mid, close=close_px, iv=iv, underlying=underlying,
                ))
        except (ValueError, TypeError):
            continue
    return chain


def fetch_option_eod(expiration, strike, start_date, end_date):
    url = (
        f"{THETADATA_URL}/v3/option/history/eod"
        f"?symbol={SYMBOL}&expiration={expiration}&strike={strike}&right=C"
        f"&start_date={start_date}&end_date={end_date}"
    )
    time.sleep(0.3)
    txt = api_get(url)
    if not txt:
        return []
    rows = parse_csv(txt)
    result = []
    for r in rows:
        try:
            d = r.get("date", r.get("ms_of_day", ""))
            close = float(r.get("close", 0))
            bid = float(r.get("bid", 0))
            ask = float(r.get("ask", 0))
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else close
            result.append(dict(date=d, close=close, bid=bid, ask=ask, mid=mid))
        except (ValueError, TypeError):
            continue
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class DataManager:
    """Handles stock price data and implied volatility estimation.

    When ThetaData Terminal is available (localhost:25503), fetches real
    option chains with greeks for accurate strike selection and premium
    pricing.  All ThetaData responses are cached to disk so subsequent
    runs are instant even if the Terminal is offline.
    """

    API_SLEEP = 0.20

    def __init__(self, symbol=SYMBOL, start=START_DATE, end=END_DATE):
        self.symbol = symbol
        self.start_date = start
        self.end_date = end
        self.prices_df = None
        self.dividends = None
        self.vix_df = None
        self._vol_cache = {}
        self.thetadata_live = False
        self._expirations: List[str] = []
        self._chain_cache: Dict[str, list] = {}
        self._disk_cache: Dict = {}
        self._disk_cache_path = os.path.join(WORK_DIR, CACHE_FILE)
        self._api_calls = 0
        self._cache_hits = 0

    # ── stock & vix data ──────────────────────────────────────────────

    def fetch_all(self):
        print(f"[Data] Fetching {self.symbol} prices from Yahoo Finance...")
        tkr = yf.Ticker(self.symbol)
        hist = tkr.history(
            start=(self.start_date - timedelta(days=120)).strftime("%Y-%m-%d"),
            end=(self.end_date + timedelta(days=5)).strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        if hist.empty:
            raise RuntimeError(f"No price data for {self.symbol}")
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        hist.index = hist.index.normalize()
        self.prices_df = hist
        self.dividends = hist["Dividends"][hist["Dividends"] > 0]
        print(f"  → {len(hist)} trading days, {len(self.dividends)} dividend events")

        print("[Data] Fetching VIX for implied volatility calibration...")
        vix = yf.download("^VIX",
                          start=(self.start_date - timedelta(days=120)).strftime("%Y-%m-%d"),
                          end=(self.end_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                          progress=False)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            if vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)
            vix.index = vix.index.normalize()
            self.vix_df = vix
            print(f"  → {len(vix)} VIX data points")
        else:
            print("  → VIX data unavailable, using historical vol estimate")

        self._load_disk_cache()
        cached_chains = sum(1 for k in self._disk_cache if k.startswith("chain_"))
        if cached_chains > 500:
            self.thetadata_live = False
            print(f"[Data] Using {cached_chains} cached chains (skipping live API)")
        else:
            self.thetadata_live = check_thetadata()
        if self.thetadata_live:
            print("[Data] ThetaData Terminal detected — will use REAL option data")
            self._fetch_expirations()
        else:
            cached_chains = sum(1 for k in self._disk_cache if k.startswith("chain_"))
            if cached_chains > 0:
                print(f"[Data] ThetaData offline — {cached_chains} cached chains available on disk")
                self._extract_expirations_from_cache()
            else:
                print("[Data] ThetaData not available, no cache — Black-Scholes fallback")

    # ── disk cache ────────────────────────────────────────────────────

    def _load_disk_cache(self):
        best_path = self._disk_cache_path
        best_size = 0
        for path in [self._disk_cache_path, CACHE_FILE_BACKUP]:
            try:
                sz = os.path.getsize(path)
                if sz > best_size:
                    best_size = sz
                    best_path = path
            except OSError:
                pass
        if best_size > 0:
            try:
                with open(best_path) as f:
                    self._disk_cache = json.load(f)
                print(f"[Cache] Loaded {len(self._disk_cache)} entries from {best_path} ({best_size:,} bytes)")
            except Exception:
                self._disk_cache = {}
        else:
            self._disk_cache = {}

    def _save_disk_cache(self):
        try:
            with open(self._disk_cache_path, "w") as f:
                json.dump(self._disk_cache, f)
        except Exception:
            pass

    def _extract_expirations_from_cache(self):
        """Extract unique expiration dates from cached chain keys."""
        exps = set()
        prefix = f"chain_{self.symbol}_"
        for key in self._disk_cache:
            if key.startswith(prefix):
                parts = key[len(prefix):].split("_")
                if parts:
                    exps.add(parts[0])
        self._expirations = sorted(exps)
        if self._expirations:
            print(f"  → {len(self._expirations)} expirations extracted from cache")

    # ── ThetaData: expirations ────────────────────────────────────────

    def _fetch_expirations(self):
        cache_key = f"expirations_{self.symbol}"
        if cache_key in self._disk_cache:
            self._expirations = self._disk_cache[cache_key]
            print(f"  → {len(self._expirations)} expirations (cached)")
            return

        url = f"{THETADATA_URL}/v3/option/list/expirations?symbol={self.symbol}"
        txt = api_get(url, retries=5)
        if not txt:
            return
        rows = parse_csv(txt)
        exps = set()
        for r in rows:
            for key in ["expiration", "date"]:
                v = r.get(key, "").strip()
                if v and len(v) >= 8:
                    exps.add(v)
                    break
        self._expirations = sorted(exps)
        self._disk_cache[cache_key] = self._expirations
        self._save_disk_cache()
        print(f"  → {len(self._expirations)} expirations fetched")

    # ── ThetaData: real call chain ────────────────────────────────────

    def get_call_chain(self, expiration_str: str, query_date_str: str,
                       stock_price: float) -> list:
        """Get real call chain with greeks for a given expiration and date.

        Returns a list of dicts with keys: strike, delta, bid, ask, mid,
        close, iv, underlying.  Falls back to Black-Scholes if ThetaData
        is unavailable and no cached data exists.
        """
        cache_key = f"chain_{self.symbol}_{expiration_str}_{query_date_str}"

        if cache_key in self._chain_cache:
            self._cache_hits += 1
            return self._chain_cache[cache_key]
        if cache_key in self._disk_cache:
            chain = self._disk_cache[cache_key]
            self._chain_cache[cache_key] = chain
            self._cache_hits += 1
            return chain

        chain = []
        if self.thetadata_live:
            chain = self._fetch_chain_full(expiration_str, query_date_str)
            if not chain:
                chain = self._fetch_chain_by_strikes(
                    expiration_str, query_date_str, stock_price
                )
            self._api_calls += 1

        if chain:
            self._chain_cache[cache_key] = chain
            self._disk_cache[cache_key] = chain
            if self._api_calls % 20 == 0:
                self._save_disk_cache()
        return chain

    def _fetch_chain_full(self, expiration, query_date):
        url = (
            f"{THETADATA_URL}/v3/option/history/greeks/eod"
            f"?symbol={self.symbol}&expiration={expiration}&right=C"
            f"&start_date={query_date}&end_date={query_date}"
        )
        time.sleep(self.API_SLEEP)
        txt = api_get(url)
        return self._parse_call_chain(txt)

    def _fetch_chain_by_strikes(self, expiration, query_date, stock_price):
        step = 1.0 if stock_price < 200 else (2.0 if stock_price < 500 else 5.0)
        chain = []
        k = round(stock_price * 1.00 / step) * step
        while k <= stock_price * 1.15:
            url = (
                f"{THETADATA_URL}/v3/option/history/greeks/eod"
                f"?symbol={self.symbol}&expiration={expiration}"
                f"&strike={k}&right=C"
                f"&start_date={query_date}&end_date={query_date}"
            )
            time.sleep(self.API_SLEEP)
            txt = api_get(url)
            chain.extend(self._parse_call_chain(txt))
            k += step
        return chain

    @staticmethod
    def _parse_call_chain(txt):
        if not txt:
            return []
        rows = parse_csv(txt)
        chain = []
        for r in rows:
            try:
                strike_raw = float(r.get("strike", 0))
                strike = strike_raw / 1000.0 if strike_raw > 5000 else strike_raw
                delta = float(r.get("delta", 0))
                bid = float(r.get("bid", 0))
                ask = float(r.get("ask", 0))
                iv = float(r.get("implied_vol", r.get("iv", 0)))
                close_px = float(r.get("close", 0))
                underlying = float(r.get("underlying_price", 0))
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else close_px
                if 0.01 < delta < 0.90 and mid > 0.01 and strike > 0:
                    chain.append(dict(
                        strike=strike, delta=delta, bid=bid, ask=ask,
                        mid=mid, close=close_px, iv=iv, underlying=underlying,
                    ))
            except (ValueError, TypeError):
                continue
        return chain

    # ── expiration lookup ─────────────────────────────────────────────

    def find_expiration_real(self, entry_date, target_dte):
        """Find the nearest real expiration from ThetaData's list.

        Falls back to synthetic Friday calculation when no expiration
        list is available.
        """
        if not self._expirations:
            return self.find_expiration(entry_date, target_dte)

        target = entry_date + timedelta(days=target_dte)
        best_exp = None
        best_diff = 999
        for exp_str in self._expirations:
            try:
                exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            actual_dte = (exp_d - entry_date).days
            if actual_dte < 1:
                continue
            diff = abs(actual_dte - target_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp_d
        return best_exp if best_exp else self.find_expiration(entry_date, target_dte)

    def has_real_data(self):
        """True if we have ThetaData live or cached chain data."""
        if self.thetadata_live:
            return True
        return any(k.startswith("chain_") for k in self._disk_cache)

    def print_data_stats(self):
        print(f"[Data Stats] API calls: {self._api_calls}, "
              f"Cache hits: {self._cache_hits}, "
              f"Disk entries: {len(self._disk_cache)}")

    def fetch_qqqi(self):
        """Fetch QQQI (Neos QQQ High Income ETF) data for comparison."""
        try:
            tkr = yf.Ticker(QQQI_SYMBOL)
            hist = tkr.history(start="2023-01-01", end="2026-01-15", auto_adjust=False)
            if hist.empty or len(hist) < 50:
                return None, None
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            hist.index = hist.index.normalize()
            divs = hist["Dividends"][hist["Dividends"] > 0]
            print(f"[Data] QQQI: {len(hist)} days, {len(divs)} distributions")
            return hist, divs
        except Exception as e:
            print(f"[Data] QQQI fetch failed: {e}")
            return None, None

    def get_trading_dates(self):
        mask = (
            (self.prices_df.index >= pd.Timestamp(self.start_date))
            & (self.prices_df.index <= pd.Timestamp(self.end_date))
        )
        return [d.date() for d in self.prices_df.index[mask]]

    def get_price(self, target_date):
        ts = pd.Timestamp(target_date)
        if ts in self.prices_df.index:
            return float(self.prices_df.loc[ts, "Close"])
        mask = self.prices_df.index <= ts
        if mask.any():
            return float(self.prices_df.loc[mask].iloc[-1]["Close"])
        return None

    def get_implied_vol(self, as_of_date, moneyness=1.0):
        cache_key = (as_of_date, round(moneyness, 3))
        if cache_key in self._vol_cache:
            return self._vol_cache[cache_key]

        vol = self._vix_based_vol(as_of_date)
        if vol is None:
            vol = self._hist_vol(as_of_date)

        otm_adj = vol * 0.10 * (moneyness - 1.0)
        vol = max(vol + otm_adj, 0.08)
        self._vol_cache[cache_key] = vol
        return vol

    def _vix_based_vol(self, as_of_date):
        if self.vix_df is None:
            return None
        ts = pd.Timestamp(as_of_date)
        mask = self.vix_df.index <= ts
        if not mask.any():
            return None
        row = self.vix_df.loc[mask].iloc[-1]
        vix_val = float(row["Close"]) if "Close" in row.index else None
        if vix_val is None or vix_val <= 0:
            return None
        qqq_vol = vix_val / 100.0 * 1.05
        return qqq_vol

    def _hist_vol(self, as_of_date, lookback=60):
        ts = pd.Timestamp(as_of_date)
        idx = self.prices_df.index.get_indexer([ts], method="ffill")[0]
        if idx < 20:
            idx = min(lookback, len(self.prices_df) - 1)
        window = self.prices_df["Close"].iloc[max(0, idx - lookback): idx + 1]
        if len(window) < 20:
            return 0.25
        rets = np.log(window / window.shift(1)).dropna()
        return float(rets.std() * np.sqrt(252))

    def find_expiration(self, entry_date, target_dte):
        target = entry_date + timedelta(days=target_dte)
        days_to_fri = (4 - target.weekday()) % 7
        exp = target + timedelta(days=days_to_fri)
        actual_dte = (exp - entry_date).days
        if actual_dte < target_dte * 0.6:
            exp += timedelta(days=7)
        if exp.weekday() != 4:
            exp += timedelta(days=(4 - exp.weekday()) % 7)
        return exp


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class OptionPosition:
    entry_date: date
    expiration: date
    strike: float
    entry_premium: float
    entry_stock_price: float
    delta_at_entry: float
    iv_at_entry: float
    contracts: int


@dataclass
class TradeResult:
    entry_date: date
    exit_date: date
    expiration: date
    strike: float
    entry_premium: float
    buyback_cost: float
    entry_stock_price: float
    exit_stock_price: float
    pnl_per_share: float
    called_away: bool
    exit_reason: str
    holding_days: int
    delta_at_entry: float
    contracts: int
    otm_pct: float


# ═══════════════════════════════════════════════════════════════════════════════
#  COVERED CALL BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class CoveredCallBacktest:
    """Single-parameter-set covered call backtest with daily simulation."""

    def __init__(self, dm: DataManager, dte: int, delta: float,
                 profit_taking: Optional[float], stop_loss: Optional[float]):
        self.dm = dm
        self.dte = dte
        self.delta = delta
        self.profit_taking = profit_taking
        self.stop_loss = stop_loss
        self.bs = BlackScholesEngine()

        self.trades: List[TradeResult] = []
        self.equity_curve: Dict[date, float] = {}
        self.summary: Dict = {}

    def run(self) -> Dict:
        trading_dates = self.dm.get_trading_dates()
        if not trading_dates:
            return {}

        first_price = self.dm.get_price(trading_dates[0])
        shares = (INITIAL_INVESTMENT // (first_price * 100)) * 100
        if shares == 0:
            shares = 100
        contracts = shares // 100
        stock_cost_basis = shares * first_price
        cash = INITIAL_INVESTMENT - stock_cost_basis

        position: Optional[OptionPosition] = None
        total_premium = 0.0
        total_buyback_cost = 0.0
        called_away_loss = 0.0
        dividend_income = 0.0
        prev_equity = INITIAL_INVESTMENT

        for today in trading_dates:
            price = self.dm.get_price(today)
            if price is None:
                continue

            ts = pd.Timestamp(today)
            if ts in self.dm.dividends.index:
                div = float(self.dm.dividends[ts]) * shares
                cash += div
                dividend_income += div

            if position is not None:
                exit_info = self._check_exit(position, today, price, shares)
                if exit_info:
                    reason, buyback = exit_info
                    called = reason == "called_away"

                    if called:
                        rebuy_penalty = (price - position.strike) * shares
                        cash -= rebuy_penalty
                        called_away_loss += rebuy_penalty
                        pnl_ps = position.entry_premium - (price - position.strike)
                    else:
                        cash -= buyback * shares
                        total_buyback_cost += buyback * shares
                        pnl_ps = position.entry_premium - buyback

                    commission = COMMISSION_PER_CONTRACT * contracts * 2
                    cash -= commission

                    self.trades.append(TradeResult(
                        entry_date=position.entry_date,
                        exit_date=today,
                        expiration=position.expiration,
                        strike=position.strike,
                        entry_premium=position.entry_premium,
                        buyback_cost=buyback if not called else price - position.strike,
                        entry_stock_price=position.entry_stock_price,
                        exit_stock_price=price,
                        pnl_per_share=pnl_ps,
                        called_away=called,
                        exit_reason=reason,
                        holding_days=(today - position.entry_date).days,
                        delta_at_entry=position.delta_at_entry,
                        contracts=contracts,
                        otm_pct=(position.strike / position.entry_stock_price - 1) * 100,
                    ))
                    position = None

            if position is None:
                cutoff = self.dm.end_date - timedelta(days=max(self.dte // 2, 3))
                if today <= cutoff:
                    position = self._open_position(today, price, contracts)
                    if position:
                        cash += position.entry_premium * shares
                        total_premium += position.entry_premium * shares
                        cash -= COMMISSION_PER_CONTRACT * contracts

            option_liability = 0.0
            if position is not None:
                rem_dte = max((position.expiration - today).days, 0)
                T = max(rem_dte / 365.0, 1e-6)
                mkt_vol = self.dm.get_implied_vol(today, position.strike / price)
                sigma = position.iv_at_entry * 0.3 + mkt_vol * 0.7 if position.iv_at_entry > 0 else mkt_vol
                option_liability = self.bs.call_price(
                    price, position.strike, T, RISK_FREE_RATE, sigma
                ) * shares

            equity = shares * price + cash - option_liability
            self.equity_curve[today] = equity

        self._compute_summary(
            total_premium, total_buyback_cost, called_away_loss,
            dividend_income, shares, stock_cost_basis
        )
        return self.summary

    def _check_exit(self, pos: OptionPosition, today: date, price: float,
                    shares: int) -> Optional[Tuple[str, float]]:
        if today >= pos.expiration:
            if price > pos.strike:
                return ("called_away", 0.0)
            else:
                return ("expired_otm", 0.0)

        rem_dte = (pos.expiration - today).days
        T = max(rem_dte / 365.0, 1e-6)
        market_vol = self.dm.get_implied_vol(today, pos.strike / price)
        sigma = pos.iv_at_entry * 0.3 + market_vol * 0.7 if pos.iv_at_entry > 0 else market_vol
        current_value = self.bs.call_price(price, pos.strike, T, RISK_FREE_RATE, sigma)

        if self.profit_taking is not None:
            target_buyback = pos.entry_premium * (1.0 - self.profit_taking)
            if current_value <= target_buyback:
                return ("profit_taking", current_value * (1 + SLIPPAGE_PCT))

        if self.stop_loss is not None:
            max_cost = pos.entry_premium * self.stop_loss
            if current_value >= max_cost:
                return ("stop_loss", current_value * (1 + SLIPPAGE_PCT))

        return None

    def _open_position(self, today: date, price: float,
                       contracts: int) -> Optional[OptionPosition]:
        exp = self.dm.find_expiration_real(today, self.dte)
        actual_dte = (exp - today).days
        T = actual_dte / 365.0
        if T <= 0:
            return None

        exp_str = exp.strftime("%Y-%m-%d")
        query_str = today.strftime("%Y-%m-%d")
        chain = self.dm.get_call_chain(exp_str, query_str, price)

        if chain:
            otm = [c for c in chain if c["strike"] > price]
            if not otm:
                otm = chain
            best = min(otm, key=lambda c: abs(c["delta"] - self.delta))
            if abs(best["delta"] - self.delta) <= 0.20:
                strike = best["strike"]
                premium = best["mid"] * (1 - SLIPPAGE_PCT)
                actual_delta = best["delta"]
                iv = best["iv"] if best["iv"] > 0 else self.dm.get_implied_vol(today)
                if premium < 0.01:
                    return None
                return OptionPosition(
                    entry_date=today, expiration=exp, strike=strike,
                    entry_premium=premium, entry_stock_price=price,
                    delta_at_entry=actual_delta, iv_at_entry=iv,
                    contracts=contracts,
                )

        sigma = self.dm.get_implied_vol(today)
        strike = self.bs.strike_for_delta(price, self.delta, T, RISK_FREE_RATE, sigma)

        if strike <= price:
            step = 1 if price < 200 else (2 if price < 500 else 5)
            strike = price + step

        premium, adj_sigma = self.bs.price_with_skew(
            price, strike, T, RISK_FREE_RATE, sigma
        )
        premium *= (1 - SLIPPAGE_PCT)
        actual_delta = self.bs.call_delta(price, strike, T, RISK_FREE_RATE, adj_sigma)

        if premium < 0.01:
            return None

        return OptionPosition(
            entry_date=today, expiration=exp, strike=strike,
            entry_premium=premium, entry_stock_price=price,
            delta_at_entry=actual_delta, iv_at_entry=adj_sigma,
            contracts=contracts,
        )

    def _compute_summary(self, total_premium, total_buyback, called_away_loss,
                         dividend_income, shares, stock_cost_basis):
        dates = sorted(self.equity_curve.keys())
        if len(dates) < 10:
            self.summary = {}
            return

        values = pd.Series(
            [self.equity_curve[d] for d in dates],
            index=pd.DatetimeIndex(dates),
        )
        years = (dates[-1] - dates[0]).days / 365.25
        final_val = float(values.iloc[-1])

        total_return = (final_val / INITIAL_INVESTMENT - 1) * 100
        cagr = ((final_val / INITIAL_INVESTMENT) ** (1.0 / years) - 1) * 100

        daily_rets = values.pct_change().dropna()
        ann_vol = float(daily_rets.std() * np.sqrt(252) * 100)
        sharpe = (
            (daily_rets.mean() * 252 - RISK_FREE_RATE)
            / (daily_rets.std() * np.sqrt(252))
            if daily_rets.std() > 0 else 0.0
        )

        running_max = values.cummax()
        drawdown = (values - running_max) / running_max
        max_dd = float(drawdown.min() * 100)

        n_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_per_share > 0)
        win_rate = wins / n_trades * 100 if n_trades > 0 else 0
        n_called = sum(1 for t in self.trades if t.called_away)
        called_pct = n_called / n_trades * 100 if n_trades > 0 else 0

        net_premium = total_premium - total_buyback
        premium_tax = net_premium * ORDINARY_TAX_RATE
        stock_gain = max(final_val - INITIAL_INVESTMENT, 0)
        stock_tax = stock_gain * LTCG_TAX_RATE
        div_tax = dividend_income * QQQ_DIVIDEND_TAX_RATE
        total_tax = premium_tax + div_tax
        after_tax_value = final_val - total_tax

        sec1256_tax = net_premium * SECTION_1256_RATE
        sec1256_savings = premium_tax - sec1256_tax

        avg_prem_per_trade = total_premium / n_trades if n_trades > 0 else 0
        ann_premium_yield = (total_premium / years / INITIAL_INVESTMENT * 100) if years > 0 else 0
        avg_hold = np.mean([t.holding_days for t in self.trades]) if self.trades else 0
        avg_otm = np.mean([t.otm_pct for t in self.trades]) if self.trades else 0

        self.summary = {
            "dte": self.dte,
            "delta": self.delta,
            "profit_taking": self.profit_taking,
            "stop_loss": self.stop_loss,
            "label": self._label(),
            "total_return": round(total_return, 2),
            "cagr": round(cagr, 2),
            "ann_volatility": round(ann_vol, 2),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 1),
            "num_trades": n_trades,
            "called_away": n_called,
            "called_away_pct": round(called_pct, 1),
            "total_premium": round(total_premium, 2),
            "net_premium": round(net_premium, 2),
            "total_buyback": round(total_buyback, 2),
            "called_away_loss": round(called_away_loss, 2),
            "dividend_income": round(dividend_income, 2),
            "final_value": round(final_val, 2),
            "after_tax_value": round(after_tax_value, 2),
            "premium_tax_ordinary": round(premium_tax, 2),
            "premium_tax_1256": round(sec1256_tax, 2),
            "sec1256_savings": round(sec1256_savings, 2),
            "ann_premium_yield": round(ann_premium_yield, 2),
            "avg_premium_per_trade": round(avg_prem_per_trade, 2),
            "avg_holding_days": round(avg_hold, 1),
            "avg_otm_pct": round(avg_otm, 2),
        }

    def _label(self):
        pt = f"PT{int(self.profit_taking*100)}%" if self.profit_taking else "NoPT"
        sl = f"SL{self.stop_loss:.0f}x" if self.stop_loss else "NoSL"
        return f"DTE{self.dte}_Δ{self.delta:.2f}_{pt}_{sl}"


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID SEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class GridSearchEngine:
    """Runs all parameter combinations and collects results."""

    def __init__(self, dm: DataManager):
        self.dm = dm
        self.backtests: List[CoveredCallBacktest] = []
        self.results_df: Optional[pd.DataFrame] = None
        self.buy_hold: Dict = {}
        self.qqqi: Dict = {}

    def run_buy_and_hold(self):
        trading_dates = self.dm.get_trading_dates()
        first_price = self.dm.get_price(trading_dates[0])
        shares = (INITIAL_INVESTMENT // (first_price * 100)) * 100
        if shares == 0:
            shares = 100
        cash = INITIAL_INVESTMENT - shares * first_price
        dividend_income = 0.0

        eq = {}
        for d in trading_dates:
            p = self.dm.get_price(d)
            ts = pd.Timestamp(d)
            if ts in self.dm.dividends.index:
                div = float(self.dm.dividends[ts]) * shares
                cash += div
                dividend_income += div
            eq[d] = shares * p + cash

        dates = sorted(eq.keys())
        values = pd.Series([eq[d] for d in dates], index=pd.DatetimeIndex(dates))
        years = (dates[-1] - dates[0]).days / 365.25
        final_val = float(values.iloc[-1])
        total_ret = (final_val / INITIAL_INVESTMENT - 1) * 100
        cagr = ((final_val / INITIAL_INVESTMENT) ** (1 / years) - 1) * 100

        daily_rets = values.pct_change().dropna()
        ann_vol = float(daily_rets.std() * np.sqrt(252) * 100)
        sharpe = (
            (daily_rets.mean() * 252 - RISK_FREE_RATE)
            / (daily_rets.std() * np.sqrt(252))
            if daily_rets.std() > 0 else 0.0
        )
        max_dd = float(((values - values.cummax()) / values.cummax()).min() * 100)

        stock_gain_tax = max(final_val - INITIAL_INVESTMENT, 0) * LTCG_TAX_RATE
        div_tax = dividend_income * QQQ_DIVIDEND_TAX_RATE

        self.buy_hold = {
            "equity_curve": eq,
            "final_value": round(final_val, 2),
            "total_return": round(total_ret, 2),
            "cagr": round(cagr, 2),
            "ann_volatility": round(ann_vol, 2),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(max_dd, 2),
            "shares": shares,
            "entry_price": first_price,
            "dividend_income": round(dividend_income, 2),
            "after_tax_value": round(final_val - stock_gain_tax - div_tax, 2),
        }
        print(f"[B&H] QQQ Buy & Hold: {total_ret:+.1f}% | ${final_val:,.0f} | "
              f"Sharpe {float(sharpe):.3f}")

    def run_qqqi_comparison(self):
        """Run QQQI (Neos QQQ High Income ETF) total return comparison.

        Computes total return = price appreciation + reinvested dividends.
        Also computes QQQ total return over the exact same date range for
        an apples-to-apples comparison.
        """
        qqqi_hist, qqqi_divs = self.dm.fetch_qqqi()
        if qqqi_hist is None:
            print("[QQQI] No QQQI data available")
            return

        start_ts = qqqi_hist.index[0]
        end_ts = qqqi_hist.index[-1]
        start_d = start_ts.date() if hasattr(start_ts, 'date') else start_ts
        end_d = end_ts.date() if hasattr(end_ts, 'date') else end_ts
        first_price = float(qqqi_hist.iloc[0]["Close"])
        shares_qqqi = INITIAL_INVESTMENT / first_price
        cash_qqqi = 0.0
        div_total = 0.0

        eq_qqqi = {}
        for row in qqqi_hist.itertuples():
            d = row.Index.date() if hasattr(row.Index, 'date') else row.Index
            price = float(row.Close)
            ts = row.Index
            if qqqi_divs is not None and ts in qqqi_divs.index:
                dv = float(qqqi_divs[ts])
                div_cash = shares_qqqi * dv
                reinv_shares = div_cash / price if price > 0 else 0
                shares_qqqi += reinv_shares
                div_total += div_cash
            eq_qqqi[d] = shares_qqqi * price + cash_qqqi

        qqqi_dates = sorted(eq_qqqi.keys())
        qqqi_values = pd.Series([eq_qqqi[d] for d in qqqi_dates],
                                index=pd.DatetimeIndex(qqqi_dates))
        qqqi_final = float(qqqi_values.iloc[-1])
        qqqi_years = (qqqi_dates[-1] - qqqi_dates[0]).days / 365.25
        qqqi_total_ret = (qqqi_final / INITIAL_INVESTMENT - 1) * 100
        qqqi_cagr = ((qqqi_final / INITIAL_INVESTMENT) ** (1 / qqqi_years) - 1) * 100
        qqqi_price_ret = (float(qqqi_hist.iloc[-1]["Close"]) / first_price - 1) * 100

        qqqi_daily = qqqi_values.pct_change().dropna()
        qqqi_vol = float(qqqi_daily.std() * np.sqrt(252) * 100)
        qqqi_sharpe = (
            (qqqi_daily.mean() * 252 - RISK_FREE_RATE) / (qqqi_daily.std() * np.sqrt(252))
            if qqqi_daily.std() > 0 else 0.0
        )
        qqqi_dd = float(((qqqi_values - qqqi_values.cummax()) / qqqi_values.cummax()).min() * 100)

        qqqi_income_tax = div_total * SECTION_1256_RATE
        qqqi_price_gain = max(qqqi_final - INITIAL_INVESTMENT - div_total, 0)
        qqqi_cap_tax = qqqi_price_gain * LTCG_TAX_RATE
        qqqi_after_tax = qqqi_final - qqqi_income_tax - qqqi_cap_tax

        # QQQ over the exact same period (for fair comparison)
        qqq_mask = (
            (self.dm.prices_df.index >= start_ts)
            & (self.dm.prices_df.index <= end_ts)
        )
        qqq_sub = self.dm.prices_df[qqq_mask]
        if len(qqq_sub) > 0:
            qqq_first = float(qqq_sub.iloc[0]["Close"])
            qqq_shares = INITIAL_INVESTMENT / qqq_first
            qqq_div_total = 0.0
            eq_qqq = {}
            for row in qqq_sub.itertuples():
                d = row.Index.date() if hasattr(row.Index, 'date') else row.Index
                p = float(row.Close)
                ts2 = row.Index
                if ts2 in self.dm.dividends.index:
                    dv2 = float(self.dm.dividends[ts2])
                    reinv2 = qqq_shares * dv2 / p if p > 0 else 0
                    qqq_shares += reinv2
                    qqq_div_total += qqq_shares * dv2
                eq_qqq[d] = qqq_shares * p

            qqq_dates = sorted(eq_qqq.keys())
            qqq_final = eq_qqq[qqq_dates[-1]]
            qqq_same_ret = (qqq_final / INITIAL_INVESTMENT - 1) * 100
            qqq_same_cagr = ((qqq_final / INITIAL_INVESTMENT) ** (1 / qqqi_years) - 1) * 100
            qqq_vals = pd.Series([eq_qqq[d] for d in qqq_dates], index=pd.DatetimeIndex(qqq_dates))
            qqq_daily = qqq_vals.pct_change().dropna()
            qqq_same_vol = float(qqq_daily.std() * np.sqrt(252) * 100)
            qqq_same_sharpe = (
                (qqq_daily.mean() * 252 - RISK_FREE_RATE) / (qqq_daily.std() * np.sqrt(252))
                if qqq_daily.std() > 0 else 0.0
            )
            qqq_same_dd = float(((qqq_vals - qqq_vals.cummax()) / qqq_vals.cummax()).min() * 100)
        else:
            qqq_same_ret = qqq_same_cagr = qqq_same_vol = qqq_same_sharpe = 0
            qqq_same_dd = 0
            eq_qqq = {}

        # Best CC over same period
        best_cc_same = None
        for bt in self.backtests:
            if not bt.summary:
                continue
            ec = bt.equity_curve
            overlap_dates = [d for d in ec if start_d <= d <= end_d]
            if len(overlap_dates) < 10:
                continue
            v0 = ec[overlap_dates[0]]
            vf = ec[overlap_dates[-1]]
            same_ret = (vf / v0 - 1) * 100
            if best_cc_same is None or same_ret > best_cc_same["same_ret"]:
                vals = pd.Series([ec[d] for d in overlap_dates],
                                 index=pd.DatetimeIndex(overlap_dates))
                dr = vals.pct_change().dropna()
                cc_vol = float(dr.std() * np.sqrt(252) * 100)
                cc_sh = (
                    (dr.mean() * 252 - RISK_FREE_RATE) / (dr.std() * np.sqrt(252))
                    if dr.std() > 0 else 0.0
                )
                cc_dd = float(((vals - vals.cummax()) / vals.cummax()).min() * 100)
                best_cc_same = {
                    "label": bt.summary["label"],
                    "same_ret": round(same_ret, 2),
                    "same_cagr": round(((vf/v0) ** (1/qqqi_years) - 1) * 100, 2),
                    "same_vol": round(cc_vol, 2),
                    "same_sharpe": round(float(cc_sh), 3),
                    "same_dd": round(cc_dd, 2),
                    "equity_curve": {d: ec[d] for d in overlap_dates},
                }

        self.qqqi = {
            "available": True,
            "start": str(start_d), "end": str(end_d),
            "days": len(qqqi_hist),
            "first_price": round(first_price, 2),
            "last_price": round(float(qqqi_hist.iloc[-1]["Close"]), 2),
            "price_return": round(qqqi_price_ret, 2),
            "total_dividends": round(div_total, 2),
            "dividend_yield_ann": round(div_total / qqqi_years / INITIAL_INVESTMENT * 100, 2),
            "total_return": round(qqqi_total_ret, 2),
            "cagr": round(qqqi_cagr, 2),
            "ann_volatility": round(qqqi_vol, 2),
            "sharpe": round(float(qqqi_sharpe), 3),
            "max_drawdown": round(qqqi_dd, 2),
            "final_value": round(qqqi_final, 2),
            "after_tax_value": round(qqqi_after_tax, 2),
            "income_tax": round(qqqi_income_tax, 2),
            "equity_curve": eq_qqqi,
            "qqq_same_period": {
                "total_return": round(qqq_same_ret, 2),
                "cagr": round(qqq_same_cagr, 2),
                "ann_volatility": round(qqq_same_vol, 2),
                "sharpe": round(float(qqq_same_sharpe), 3),
                "max_drawdown": round(qqq_same_dd, 2),
                "final_value": round(qqq_final, 2) if eq_qqq else 0,
                "equity_curve": eq_qqq,
            },
            "best_cc_same": best_cc_same,
        }
        print(f"[QQQI] {start_d}→{end_d}: Price {qqqi_price_ret:+.1f}% + "
              f"Div ${div_total:,.0f} = Total {qqqi_total_ret:+.1f}% | "
              f"Sharpe {float(qqqi_sharpe):.3f}")

    def run_grid_search(self):
        combos = list(product(DTE_GRID, DELTA_GRID, PROFIT_TAKING_GRID, STOP_LOSS_GRID))
        total = len(combos)
        src = "ThetaData + BS" if self.dm.has_real_data() else "Black-Scholes"
        print(f"\n[Grid] Running {total} parameter combinations (data: {src})...")

        for i, (dte, delta, pt, sl) in enumerate(combos):
            bt = CoveredCallBacktest(self.dm, dte, delta, pt, sl)
            bt.run()
            self.backtests.append(bt)

            if (i + 1) % 20 == 0 or i == total - 1:
                s = bt.summary
                if s:
                    print(f"  [{i+1}/{total}] {s['label']}: "
                          f"Return={s['total_return']:+.1f}% "
                          f"Sharpe={s['sharpe']:.3f} "
                          f"Called={s['called_away']} "
                          f"[API:{self.dm._api_calls}]")

        rows = [bt.summary for bt in self.backtests if bt.summary]
        self.results_df = pd.DataFrame(rows)
        print(f"\n[Grid] Complete: {len(rows)} valid results "
              f"(API calls: {self.dm._api_calls}, cache hits: {self.dm._cache_hits})")
        self.dm._save_disk_cache()
        return self.results_df

    def get_best(self, metric="sharpe", ascending=False):
        if self.results_df is None or self.results_df.empty:
            return None
        df = self.results_df.sort_values(metric, ascending=ascending)
        return df.iloc[0]

    def get_top_n(self, n=10, metric="sharpe"):
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        return self.results_df.nlargest(n, metric)

    def get_heatmap_data(self, metric="sharpe", pt=None, sl=None):
        if self.results_df is None:
            return pd.DataFrame()
        df = self.results_df.copy()
        if pt is not None:
            df = df[df["profit_taking"] == pt] if pt else df[df["profit_taking"].isna()]
        if sl is not None:
            df = df[df["stop_loss"] == sl] if sl else df[df["stop_loss"].isna()]
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="dte", columns="delta", values=metric, aggfunc="mean")


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class ReportGenerator:
    """Generates a comprehensive HTML report."""

    def __init__(self, engine: GridSearchEngine, dm: DataManager):
        self.engine = engine
        self.dm = dm

    def generate(self, filepath=REPORT_FILE):
        df = self.engine.results_df
        bh = self.engine.buy_hold
        best_sharpe = self.engine.get_best("sharpe")
        best_return = self.engine.get_best("total_return")
        top10 = self.engine.get_top_n(10, "sharpe")

        best_bt = None
        for bt in self.engine.backtests:
            if bt.summary and bt.summary.get("label") == best_sharpe["label"]:
                best_bt = bt
                break

        html = self._build_html(df, bh, best_sharpe, best_return, top10, best_bt)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n[Report] Saved → {filepath}")

    def _build_html(self, df, bh, best_sharpe, best_return, top10, best_bt):
        qqqi = self.engine.qqqi
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QQQ Covered Call Grid Search Backtest</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
{self._css()}
</style>
</head>
<body>
<div class="container">
{self._hero(bh, best_sharpe, df)}
{self._experiment_purpose()}
{self._methodology()}
{self._data_sources()}
{self._kpi_cards(bh, best_sharpe, best_return, df)}
{self._heatmap_section(df)}
{self._top10_table(top10, bh)}
{self._equity_chart(bh, best_bt, df)}
{self._dte_analysis(df)}
{self._delta_analysis(df)}
{self._management_analysis(df)}
{self._called_away_section(df)}
{self._qqqi_comparison(qqqi, bh, best_sharpe)}
{self._tax_section(df, bh, best_sharpe, qqqi)}
{self._full_results_table(df, bh)}
{self._trade_log(best_bt, best_sharpe)}
{self._insights(df, bh, best_sharpe)}
</div>
</body>
</html>"""

    def _css(self):
        return """:root {
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --text: #e2e8f0; --dim: #94a3b8; --accent: #3b82f6;
    --green: #22c55e; --red: #ef4444; --orange: #f59e0b; --purple: #a855f7;
    --cyan: #06b6d4; --pink: #ec4899;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.hero { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border: 1px solid var(--surface2); border-radius: 16px;
        padding: 40px; margin-bottom: 30px; text-align: center; }
.hero h1 { font-size: 2em; margin-bottom: 8px; }
.hero .subtitle { color: var(--dim); font-size: 1.1em; }
.hero .config { display: flex; justify-content: center; gap: 20px;
                margin-top: 20px; flex-wrap: wrap; }
.hero .config span { background: var(--surface); padding: 6px 16px;
                     border-radius: 8px; font-size: 0.9em; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 30px; }
.kpi { background: var(--surface); border-radius: 12px; padding: 20px;
       border-left: 4px solid var(--accent); }
.kpi.green { border-left-color: var(--green); }
.kpi.red { border-left-color: var(--red); }
.kpi.orange { border-left-color: var(--orange); }
.kpi.purple { border-left-color: var(--purple); }
.kpi.cyan { border-left-color: var(--cyan); }
.kpi .label { font-size: 0.8em; color: var(--dim); text-transform: uppercase; letter-spacing: 0.5px; }
.kpi .value { font-size: 1.7em; font-weight: 700; margin: 4px 0; }
.kpi .detail { font-size: 0.82em; color: var(--dim); }
.section { background: var(--surface); border-radius: 12px;
           padding: 28px; margin-bottom: 24px; }
.section h2 { font-size: 1.3em; margin-bottom: 16px;
              padding-bottom: 10px; border-bottom: 1px solid var(--surface2); }
.section h3 { font-size: 1.05em; margin: 16px 0 8px; color: var(--accent); }
table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
th { background: var(--surface2); color: var(--dim); text-transform: uppercase;
     font-size: 0.72em; letter-spacing: 0.5px; padding: 10px 8px;
     text-align: right; white-space: nowrap; }
th:first-child { text-align: left; }
td { padding: 9px 8px; text-align: right;
     border-bottom: 1px solid rgba(255,255,255,0.05); }
td:first-child { text-align: left; font-weight: 600; }
tr:hover { background: rgba(59,130,246,0.06); }
.best-row { background: rgba(59,130,246,0.12) !important; }
.positive { color: var(--green); }
.negative { color: var(--red); }
.chart-wrap { position: relative; height: 420px; margin: 10px 0; }
.chart-sm { height: 300px; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .three-col { grid-template-columns: 1fr; } }
.heatmap { display: grid; gap: 3px; margin: 16px 0; }
.heatmap-cell { padding: 12px 8px; text-align: center; border-radius: 6px;
                font-weight: 600; font-size: 0.9em; }
.heatmap-header { background: var(--surface2); color: var(--dim);
                  font-size: 0.75em; text-transform: uppercase; }
.insight { background: rgba(59,130,246,0.08); border-left: 3px solid var(--accent);
           padding: 14px 18px; border-radius: 0 8px 8px 0; margin: 12px 0;
           font-size: 0.95em; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 12px;
       font-size: 0.78em; font-weight: 600; margin: 2px; }
.tag-green { background: rgba(34,197,94,0.2); color: var(--green); }
.tag-red { background: rgba(239,68,68,0.2); color: var(--red); }
.tag-blue { background: rgba(59,130,246,0.2); color: var(--accent); }
.tag-orange { background: rgba(245,158,11,0.2); color: var(--orange); }
.scroll-x { overflow-x: auto; }
.footer { text-align: center; color: var(--dim); font-size: 0.85em;
          margin-top: 30px; padding: 20px; }
.flow { display: flex; align-items: center; gap: 0; flex-wrap: wrap;
        justify-content: center; margin: 20px 0; }
.flow-step { background: var(--surface2); border-radius: 10px; padding: 14px 18px;
             text-align: center; min-width: 140px; font-size: 0.88em; }
.flow-step strong { display: block; color: var(--accent); font-size: 0.78em;
                    text-transform: uppercase; margin-bottom: 4px; }
.flow-arrow { color: var(--dim); font-size: 1.5em; padding: 0 6px; }
.source-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
               gap: 16px; margin-top: 16px; }
.source-card { background: var(--surface2); border-radius: 10px; padding: 18px; }
.source-card h4 { color: var(--accent); margin-bottom: 8px; font-size: 0.95em; }
.source-card p { font-size: 0.88em; color: var(--dim); }
.source-card ul { font-size: 0.85em; color: var(--dim); margin-top: 6px; padding-left: 18px; }
.obj-list { list-style: none; padding: 0; }
.obj-list li { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
               font-size: 0.92em; }
.obj-list li::before { content: '→ '; color: var(--accent); font-weight: 700; }"""

    def _hero(self, bh, best, df):
        n = len(df) if df is not None else 0
        return f"""<div class="hero">
    <h1>QQQ Covered Call Grid Search</h1>
    <p class="subtitle">Backtest Period: {START_DATE} to {END_DATE} | {n} Parameter Combinations</p>
    <div class="config">
        <span>Capital: ${INITIAL_INVESTMENT:,.0f}</span>
        <span>DTE: {DTE_GRID}</span>
        <span>Delta: {DELTA_GRID}</span>
        <span>Profit Taking: {[str(int(x*100))+"%" if x else "None" for x in PROFIT_TAKING_GRID]}</span>
        <span>Stop Loss: {[str(x)+"x" if x else "None" for x in STOP_LOSS_GRID]}</span>
    </div>
</div>"""

    def _experiment_purpose(self):
        return f"""<div class="section">
<h2>Experiment Purpose</h2>
<p style="margin-bottom:14px;">This study aims to answer a core question for income-oriented investors holding QQQ:</p>
<ul class="obj-list">
<li><strong>Primary Goal:</strong> Find the optimal (DTE, Delta, Profit Taking, Stop Loss) combination
    that maximizes the <strong>Sharpe Ratio</strong> (risk-adjusted return) while keeping the "called away" frequency low
    — preserving long-term equity upside.</li>
<li><strong>Secondary Goal:</strong> Compare the after-tax total return of a DIY covered call strategy
    on QQQ versus buying <strong>QQQI</strong> (NEOS QQQ High Income ETF), which implements a similar
    covered call overlay but with Section 1256 tax treatment on its distributions.</li>
<li><strong>Tertiary Goal:</strong> Quantify the <strong>tax drag</strong> of covered call premiums
    taxed as ordinary income (37%) vs Section 1256 treatment (60/40 rule, ~{SECTION_1256_RATE*100:.1f}% blended),
    and evaluate whether the tax structure of QQQI justifies its management fee and potential tracking error.</li>
<li><strong>Practical Question:</strong> In a predominantly <em>bullish</em> market regime (2021–2025),
    does writing covered calls add value or does the capped upside offset the premium income?
    Under what parameter combinations does CC outperform pure Buy &amp; Hold?</li>
</ul>
</div>"""

    def _methodology(self):
        return f"""<div class="section">
<h2>Experiment Design &amp; Methodology</h2>

<h3>1. Backtest Framework</h3>
<div class="flow">
<div class="flow-step"><strong>Step 1</strong>Buy QQQ<br>${INITIAL_INVESTMENT:,.0f}<br>{START_DATE}</div>
<div class="flow-arrow">→</div>
<div class="flow-step"><strong>Step 2</strong>Sell OTM Call<br>Select by Δ target<br>Nearest expiry to DTE</div>
<div class="flow-arrow">→</div>
<div class="flow-step"><strong>Step 3</strong>Daily Monitor<br>Mark-to-market<br>Check PT / SL</div>
<div class="flow-arrow">→</div>
<div class="flow-step"><strong>Step 4</strong>Exit Event<br>Expiry / PT hit /<br>SL hit</div>
<div class="flow-arrow">→</div>
<div class="flow-step"><strong>Step 5</strong>If Called Away<br>Cash-settle<br>Re-buy at market</div>
<div class="flow-arrow">→</div>
<div class="flow-step"><strong>Step 6</strong>Repeat<br>Sell new CC<br>Until {END_DATE}</div>
</div>

<h3>2. Grid Search Parameters</h3>
<div class="scroll-x"><table>
<tr><th>Parameter</th><th>Values</th><th>Description</th></tr>
<tr><td>DTE (Days to Expiry)</td><td>{DTE_GRID}</td>
    <td>Target days until option expiration. Mapped to nearest Friday expiry.</td></tr>
<tr><td>Delta</td><td>{DELTA_GRID}</td>
    <td>Target call delta at entry. Lower Δ = more OTM = less premium but more upside preservation.</td></tr>
<tr><td>Profit Taking</td><td>{[str(int(x*100))+"% of premium" if x else "None (hold to expiry)" for x in PROFIT_TAKING_GRID]}</td>
    <td>Buy back when option decays to (1 − PT%) of initial premium. E.g., PT 50%: buy back at 50% of entry price.</td></tr>
<tr><td>Stop Loss</td><td>{[str(x)+"× initial premium" if x else "None (no stop)" for x in STOP_LOSS_GRID]}</td>
    <td>Buy back when option rises to SL× the initial premium received.</td></tr>
</table></div>
<p style="color:var(--dim);margin-top:8px;">Total combinations: {len(DTE_GRID)} × {len(DELTA_GRID)} × {len(PROFIT_TAKING_GRID)} × {len(STOP_LOSS_GRID)} = <strong>{len(DTE_GRID)*len(DELTA_GRID)*len(PROFIT_TAKING_GRID)*len(STOP_LOSS_GRID)}</strong></p>

<h3>3. Position Management Rules</h3>
<div class="scroll-x"><table>
<tr><th>Event</th><th>Action</th><th>Cash Impact</th></tr>
<tr><td>Option sold</td><td>Receive premium</td><td>+Premium × Shares (after 1% slippage)</td></tr>
<tr><td>Profit Taking triggered</td><td>Buy back option at reduced price</td><td>−Buyback cost (1% slippage added)</td></tr>
<tr><td>Stop Loss triggered</td><td>Buy back option at elevated price</td><td>−Buyback cost (1% slippage added)</td></tr>
<tr><td>Expired OTM</td><td>Option expires worthless</td><td>$0 (full premium retained)</td></tr>
<tr><td>Called Away (ITM at expiry)</td><td>Cash-settle: sell at strike, re-buy at market</td><td>−(Market − Strike) × Shares</td></tr>
<tr><td>Commission</td><td>${COMMISSION_PER_CONTRACT:.2f} per contract per trade</td><td>−Commission on open &amp; close</td></tr>
</table></div>

<h3>4. Option Pricing Model</h3>
<p style="color:var(--dim);font-size:0.92em;margin:8px 0;">
Options are priced daily using the <strong>Black-Scholes model</strong> with:<br>
• <strong>Implied Volatility</strong>: Calibrated from CBOE VIX (×1.05 scaling for QQQ beta) with real-time daily updates<br>
• <strong>Volatility Skew</strong>: Applied as σ<sub>adj</sub> = σ<sub>base</sub> + 0.12 × σ<sub>base</sub> × (1 − K/S) for OTM calls<br>
• <strong>Risk-Free Rate</strong>: {RISK_FREE_RATE*100:.0f}% (constant, approximate Fed Funds 2021–2025 average)<br>
• <strong>Strike Selection</strong>: Analytic inversion of BS delta formula, rounded to realistic listed strike intervals
</p>

<h3>5. Tax Model</h3>
<div class="scroll-x"><table>
<tr><th>Income Type</th><th>Source</th><th>Tax Treatment</th><th>Rate</th></tr>
<tr><td>CC Premiums (DIY)</td><td>Short call gains</td><td>Ordinary income</td><td>{ORDINARY_TAX_RATE*100:.0f}%</td></tr>
<tr><td>CC Premiums (QQQI)</td><td>Section 1256 contracts</td><td>60% LTCG + 40% ordinary</td><td>~{SECTION_1256_RATE*100:.1f}%</td></tr>
<tr><td>Stock Appreciation</td><td>QQQ capital gains (>1yr)</td><td>Long-term capital gains + NIIT</td><td>{LTCG_TAX_RATE*100:.1f}%</td></tr>
<tr><td>QQQ Dividends</td><td>Mostly non-qualified</td><td>Ordinary income</td><td>{QQQ_DIVIDEND_TAX_RATE*100:.0f}%</td></tr>
</table></div>
</div>"""

    def _data_sources(self):
        n_days = len(self.dm.get_trading_dates())
        n_divs = len(self.dm.dividends)
        vix_n = len(self.dm.vix_df) if self.dm.vix_df is not None else 0
        real_chains = sum(1 for k in self.dm._disk_cache if k.startswith("chain_"))
        if self.dm.thetadata_live:
            td_status = f"Connected — {self.dm._api_calls} API calls, {real_chains} chains cached"
        elif real_chains > 0:
            td_status = f"Offline — using {real_chains} cached real chains from previous runs"
        else:
            td_status = "Not available — full Black-Scholes fallback mode"
        return f"""<div class="section">
<h2>Data Sources</h2>
<div class="source-grid">
<div class="source-card">
    <h4>QQQ Stock Prices</h4>
    <p><strong>Source:</strong> Yahoo Finance (yfinance API)</p>
    <ul>
        <li>Ticker: {SYMBOL} (Invesco QQQ Trust, tracks Nasdaq-100)</li>
        <li>Period: {self.dm.start_date} to {self.dm.end_date}</li>
        <li>Trading Days: {n_days}</li>
        <li>Dividend Events: {n_divs}</li>
        <li>Fields: Open, High, Low, Close, Volume, Dividends</li>
        <li>Adjustment: unadjusted (auto_adjust=False), dividends tracked separately</li>
    </ul>
</div>
<div class="source-card">
    <h4>Implied Volatility (VIX)</h4>
    <p><strong>Source:</strong> CBOE VIX Index via Yahoo Finance (^VIX)</p>
    <ul>
        <li>Data Points: {vix_n}</li>
        <li>Usage: Daily IV proxy for QQQ options</li>
        <li>Scaling: VIX × 1.05 (QQQ beta adjustment)</li>
        <li>Skew Model: σ_adj = σ_base × (1 + 0.12 × (1 − K/S))</li>
        <li>Fallback: 60-day historical realized volatility if VIX unavailable</li>
    </ul>
</div>
<div class="source-card">
    <h4>Option Pricing</h4>
    <p><strong>Model:</strong> Black-Scholes European Call</p>
    <ul>
        <li>Strike Selection: Analytic BS delta inversion</li>
        <li>Daily Mark-to-Market: Full BS repricing with updated S, T, σ</li>
        <li>Slippage: {SLIPPAGE_PCT*100:.0f}% adverse on premium (entry &amp; exit)</li>
        <li>Commission: ${COMMISSION_PER_CONTRACT:.2f} per contract per leg</li>
        <li>Expiration: Weekly (nearest Friday to target DTE)</li>
    </ul>
</div>
<div class="source-card">
    <h4>QQQI Comparison Data</h4>
    <p><strong>Source:</strong> Yahoo Finance (QQQI — NEOS QQQ High Income ETF)</p>
    <ul>
        <li>Inception: January 2024</li>
        <li>Includes: Daily close prices + all distributions</li>
        <li>Total Return: Price appreciation + reinvested dividends</li>
        <li>Tax Model: Distributions taxed as Section 1256 (60/40)</li>
        <li>Comparison: Same-period QQQ and best CC strategy overlaid</li>
    </ul>
</div>
</div>
<div class="insight" style="margin-top:16px;">
<strong>ThetaData Terminal API:</strong> {td_status}.<br>
When ThetaData is connected (localhost:25503), real historical option chains with Greeks (EOD) are used
for strike selection and premium calibration. Otherwise, Black-Scholes with VIX-calibrated IV provides
option pricing throughout.
</div>
</div>"""

    def _kpi_cards(self, bh, best_sharpe, best_return, df):
        bs = best_sharpe
        br = best_return
        avg_sharpe = df["sharpe"].mean() if df is not None and not df.empty else 0
        return f"""<div class="kpi-grid">
    <div class="kpi green">
        <div class="label">Best Sharpe Ratio</div>
        <div class="value">{bs['sharpe']:.3f}</div>
        <div class="detail">{bs['label']}<br>Return: {bs['total_return']:+.1f}%</div>
    </div>
    <div class="kpi purple">
        <div class="label">Best Total Return</div>
        <div class="value">{br['total_return']:+.1f}%</div>
        <div class="detail">{br['label']}<br>Sharpe: {br['sharpe']:.3f}</div>
    </div>
    <div class="kpi orange">
        <div class="label">Buy & Hold Return</div>
        <div class="value">{bh['total_return']:+.1f}%</div>
        <div class="detail">Sharpe: {bh['sharpe']:.3f}<br>Max DD: {bh['max_drawdown']:.1f}%</div>
    </div>
    <div class="kpi cyan">
        <div class="label">Best Premium Yield</div>
        <div class="value">{bs['ann_premium_yield']:.1f}%/yr</div>
        <div class="detail">Net Premium: ${bs['net_premium']:,.0f}<br>{bs['num_trades']} trades</div>
    </div>
    <div class="kpi red">
        <div class="label">Best Called Away Rate</div>
        <div class="value">{bs['called_away_pct']:.0f}%</div>
        <div class="detail">{bs['called_away']}/{bs['num_trades']} trades<br>Avg Hold: {bs['avg_holding_days']:.0f}d</div>
    </div>
    <div class="kpi">
        <div class="label">Avg Grid Sharpe</div>
        <div class="value">{avg_sharpe:.3f}</div>
        <div class="detail">{len(df)} combinations tested</div>
    </div>
</div>"""

    def _heatmap_section(self, df):
        sections = []
        mgmt_combos = [
            (None, None, "No Management"),
            (0.50, None, "PT 50%"),
            (0.60, None, "PT 60%"),
            (None, 2.0, "SL 2x"),
            (None, 3.0, "SL 3x"),
            (0.50, 2.0, "PT 50% + SL 2x"),
        ]
        for pt, sl, label in mgmt_combos:
            sub = df.copy()
            if pt is None:
                sub = sub[sub["profit_taking"].isna()]
            else:
                sub = sub[sub["profit_taking"] == pt]
            if sl is None:
                sub = sub[sub["stop_loss"].isna()]
            else:
                sub = sub[sub["stop_loss"] == sl]
            if sub.empty:
                continue

            pivot = sub.pivot_table(index="dte", columns="delta", values="sharpe", aggfunc="mean")
            if pivot.empty:
                continue

            global_min = df["sharpe"].min()
            global_max = df["sharpe"].max()

            cols = sorted(pivot.columns)
            rows_html = ""
            header = '<div class="heatmap-cell heatmap-header">DTE \\ Delta</div>'
            for c in cols:
                header += f'<div class="heatmap-cell heatmap-header">Δ {c:.2f}</div>'

            for dte in sorted(pivot.index):
                rows_html += f'<div class="heatmap-cell heatmap-header">{dte}d</div>'
                for c in cols:
                    val = pivot.loc[dte, c] if c in pivot.columns else 0
                    color = self._sharpe_color(val, global_min, global_max)
                    rows_html += (
                        f'<div class="heatmap-cell" style="background:{color};color:#fff">'
                        f'{val:.3f}</div>'
                    )

            ncols = len(cols) + 1
            sections.append(f"""<h3>{label}</h3>
<div class="heatmap" style="grid-template-columns: repeat({ncols}, 1fr);">
{header}{rows_html}
</div>""")

        heatmap_ret_html = self._build_return_heatmaps(df)

        return f"""<div class="section">
<h2>Sharpe Ratio Heatmaps (DTE × Delta)</h2>
<p style="color:var(--dim);margin-bottom:12px;">Green = higher Sharpe (better risk-adjusted return). Each heatmap shows a different management rule combination.</p>
<div class="three-col">{''.join(f'<div>{s}</div>' for s in sections[:3])}</div>
{('<div class="three-col">' + ''.join(f'<div>{s}</div>' for s in sections[3:6]) + '</div>') if len(sections) > 3 else ''}
</div>
<div class="section">
<h2>Total Return Heatmaps (DTE × Delta)</h2>
{heatmap_ret_html}
</div>"""

    def _build_return_heatmaps(self, df):
        mgmt_combos = [
            (None, None, "No Management"),
            (0.50, None, "PT 50%"),
            (0.60, None, "PT 60%"),
        ]
        sections = []
        for pt, sl, label in mgmt_combos:
            sub = df.copy()
            if pt is None:
                sub = sub[sub["profit_taking"].isna()]
            else:
                sub = sub[sub["profit_taking"] == pt]
            if sl is None:
                sub = sub[sub["stop_loss"].isna()]
            else:
                sub = sub[sub["stop_loss"] == sl]
            if sub.empty:
                continue

            pivot = sub.pivot_table(index="dte", columns="delta", values="total_return", aggfunc="mean")
            if pivot.empty:
                continue
            global_min = df["total_return"].min()
            global_max = df["total_return"].max()
            cols = sorted(pivot.columns)
            header = '<div class="heatmap-cell heatmap-header">DTE \\ Delta</div>'
            for c in cols:
                header += f'<div class="heatmap-cell heatmap-header">Δ {c:.2f}</div>'
            rows_html = ""
            for dte in sorted(pivot.index):
                rows_html += f'<div class="heatmap-cell heatmap-header">{dte}d</div>'
                for c in cols:
                    val = pivot.loc[dte, c] if c in pivot.columns else 0
                    color = self._return_color(val, global_min, global_max)
                    rows_html += (
                        f'<div class="heatmap-cell" style="background:{color};color:#fff">'
                        f'{val:+.1f}%</div>'
                    )
            ncols = len(cols) + 1
            sections.append(f"""<h3>{label}</h3>
<div class="heatmap" style="grid-template-columns: repeat({ncols}, 1fr);">
{header}{rows_html}
</div>""")
        return f'<div class="three-col">{"".join(f"<div>{s}</div>" for s in sections)}</div>'

    @staticmethod
    def _sharpe_color(val, vmin, vmax):
        if vmax == vmin:
            return "rgba(59,130,246,0.4)"
        t = max(0, min(1, (val - vmin) / (vmax - vmin)))
        if t < 0.5:
            r, g, b = 220, int(50 + 150 * t * 2), int(50 + 50 * t * 2)
        else:
            r, g, b = int(220 - 180 * (t - 0.5) * 2), int(200 + 30 * (t - 0.5) * 2), int(80)
        return f"rgba({r},{g},{b},0.7)"

    @staticmethod
    def _return_color(val, vmin, vmax):
        if vmax == vmin:
            return "rgba(59,130,246,0.4)"
        t = max(0, min(1, (val - vmin) / (vmax - vmin)))
        if t < 0.33:
            r, g, b = 220, 80, 80
        elif t < 0.66:
            r, g, b = 230, 180, 60
        else:
            r, g, b = 60, 200, 100
        return f"rgba({r},{g},{b},0.65)"

    def _top10_table(self, top10, bh):
        rows_html = ""
        rows_html += f"""<tr style="background:rgba(245,158,11,0.1)">
<td>Buy & Hold</td><td>—</td><td>—</td><td>—</td><td>—</td>
<td class="{'positive' if bh['total_return'] > 0 else 'negative'}">{bh['total_return']:+.1f}%</td>
<td>{bh['cagr']:+.1f}%</td><td>{bh['ann_volatility']:.1f}%</td>
<td><strong>{bh['sharpe']:.3f}</strong></td><td>{bh['max_drawdown']:.1f}%</td>
<td>—</td><td>—</td><td>—</td>
<td>${bh['final_value']:,.0f}</td></tr>"""

        for i, row in top10.iterrows():
            pt_str = f"{int(row['profit_taking']*100)}%" if pd.notna(row['profit_taking']) else "None"
            sl_str = f"{row['stop_loss']:.0f}x" if pd.notna(row['stop_loss']) else "None"
            css = ' class="best-row"' if i == top10.index[0] else ""
            rows_html += f"""<tr{css}>
<td>#{top10.index.get_loc(i)+1}</td>
<td>{int(row['dte'])}d</td><td>Δ {row['delta']:.2f}</td>
<td>{pt_str}</td><td>{sl_str}</td>
<td class="{'positive' if row['total_return'] > 0 else 'negative'}">{row['total_return']:+.1f}%</td>
<td>{row['cagr']:+.1f}%</td><td>{row['ann_volatility']:.1f}%</td>
<td><strong>{row['sharpe']:.3f}</strong></td><td>{row['max_drawdown']:.1f}%</td>
<td>{row['win_rate']:.0f}%</td>
<td>{int(row['called_away'])}/{int(row['num_trades'])} ({row['called_away_pct']:.0f}%)</td>
<td>{row['ann_premium_yield']:.1f}%</td>
<td>${row['final_value']:,.0f}</td></tr>"""

        return f"""<div class="section">
<h2>Top 10 Strategies by Sharpe Ratio</h2>
<div class="scroll-x"><table>
<tr><th>Rank</th><th>DTE</th><th>Delta</th><th>PT</th><th>SL</th>
<th>Total Return</th><th>CAGR</th><th>Ann Vol</th><th>Sharpe</th><th>Max DD</th>
<th>Win Rate</th><th>Called Away</th><th>Prem Yield</th><th>Final Value</th></tr>
{rows_html}
</table></div></div>"""

    def _equity_chart(self, bh, best_bt, df):
        bh_dates = sorted(bh["equity_curve"].keys())
        bh_labels = [d.strftime("%Y-%m-%d") for d in bh_dates]
        bh_vals = [round(bh["equity_curve"][d], 0) for d in bh_dates]

        datasets = [
            {"label": "Buy & Hold", "data": bh_vals,
             "borderColor": "#f59e0b", "borderWidth": 2, "pointRadius": 0, "fill": False},
        ]

        top_bts = []
        if best_bt:
            top_bts.append(("Best Sharpe", best_bt, "#22c55e"))

        for label_name, metric, color in [
            ("Best Return", "total_return", "#3b82f6"),
            ("Lowest DD", "max_drawdown", "#a855f7"),
        ]:
            row = self.engine.get_best(metric, ascending=(metric == "max_drawdown"))
            if row is not None:
                for bt in self.engine.backtests:
                    if bt.summary and bt.summary.get("label") == row["label"]:
                        if bt != best_bt:
                            top_bts.append((label_name, bt, color))
                        break

        for lbl, bt, clr in top_bts:
            bt_dates = sorted(bt.equity_curve.keys())
            bt_vals = [round(bt.equity_curve[d], 0) for d in bt_dates]
            padding = [None] * (len(bh_dates) - len(bt_dates))
            datasets.append({
                "label": f"{lbl} ({bt.summary['label']})",
                "data": padding + bt_vals if len(bt_vals) < len(bh_vals) else bt_vals,
                "borderColor": clr, "borderWidth": 1.8, "pointRadius": 0, "fill": False,
            })

        ds_json = json.dumps(datasets)
        labels_json = json.dumps(bh_labels)

        return f"""<div class="section">
<h2>Equity Curves: Top Strategies vs Buy & Hold</h2>
<div class="chart-wrap"><canvas id="equityChart"></canvas></div>
<script>
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{ labels: {labels_json}, datasets: {ds_json} }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
            x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 20 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
            y: {{ ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
                  grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }},
        interaction: {{ mode: 'index', intersect: false }},
    }}
}});
</script></div>"""

    def _dte_analysis(self, df):
        dte_agg = df.groupby("dte").agg(
            avg_return=("total_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_vol=("ann_volatility", "mean"),
            avg_called=("called_away_pct", "mean"),
            avg_premium=("ann_premium_yield", "mean"),
            avg_win=("win_rate", "mean"),
            avg_dd=("max_drawdown", "mean"),
        ).round(2)

        rows_html = ""
        for dte in sorted(dte_agg.index):
            r = dte_agg.loc[dte]
            rows_html += f"""<tr>
<td>{dte}d</td>
<td class="{'positive' if r['avg_return'] > 0 else 'negative'}">{r['avg_return']:+.1f}%</td>
<td><strong>{r['avg_sharpe']:.3f}</strong></td>
<td>{r['avg_vol']:.1f}%</td><td>{r['avg_dd']:.1f}%</td>
<td>{r['avg_win']:.0f}%</td><td>{r['avg_called']:.0f}%</td>
<td>{r['avg_premium']:.1f}%</td></tr>"""

        labels = json.dumps([f"{d}d" for d in sorted(dte_agg.index)])
        sharpe_data = json.dumps([round(dte_agg.loc[d, "avg_sharpe"], 3) for d in sorted(dte_agg.index)])
        return_data = json.dumps([round(dte_agg.loc[d, "avg_return"], 1) for d in sorted(dte_agg.index)])

        return f"""<div class="section">
<h2>Analysis by DTE</h2>
<div class="two-col">
<div>
<div class="scroll-x"><table>
<tr><th>DTE</th><th>Avg Return</th><th>Avg Sharpe</th><th>Avg Vol</th>
<th>Avg Max DD</th><th>Avg Win Rate</th><th>Avg Called Away</th><th>Avg Prem Yield</th></tr>
{rows_html}
</table></div>
</div>
<div>
<div class="chart-wrap chart-sm"><canvas id="dteChart"></canvas></div>
<script>
new Chart(document.getElementById('dteChart'), {{
    type: 'bar',
    data: {{
        labels: {labels},
        datasets: [
            {{ label: 'Avg Sharpe', data: {sharpe_data}, backgroundColor: 'rgba(59,130,246,0.7)', yAxisID: 'y' }},
            {{ label: 'Avg Return %', data: {return_data}, backgroundColor: 'rgba(34,197,94,0.5)', yAxisID: 'y1' }}
        ]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
            y: {{ position: 'left', ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }},
                  title: {{ display: true, text: 'Sharpe', color: '#94a3b8' }} }},
            y1: {{ position: 'right', ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }},
                   title: {{ display: true, text: 'Return %', color: '#94a3b8' }} }},
            x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
    }}
}});
</script>
</div></div></div>"""

    def _delta_analysis(self, df):
        delta_agg = df.groupby("delta").agg(
            avg_return=("total_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_called=("called_away_pct", "mean"),
            avg_premium=("ann_premium_yield", "mean"),
            avg_otm=("avg_otm_pct", "mean"),
            avg_win=("win_rate", "mean"),
        ).round(2)

        rows_html = ""
        for delta in sorted(delta_agg.index):
            r = delta_agg.loc[delta]
            rows_html += f"""<tr>
<td>Δ {delta:.2f}</td>
<td class="{'positive' if r['avg_return'] > 0 else 'negative'}">{r['avg_return']:+.1f}%</td>
<td><strong>{r['avg_sharpe']:.3f}</strong></td>
<td>{r['avg_called']:.0f}%</td><td>{r['avg_premium']:.1f}%</td>
<td>{r['avg_otm']:.1f}%</td><td>{r['avg_win']:.0f}%</td></tr>"""

        labels = json.dumps([f"Δ{d:.2f}" for d in sorted(delta_agg.index)])
        sharpe_data = json.dumps([round(delta_agg.loc[d, "avg_sharpe"], 3) for d in sorted(delta_agg.index)])
        called_data = json.dumps([round(delta_agg.loc[d, "avg_called"], 1) for d in sorted(delta_agg.index)])

        return f"""<div class="section">
<h2>Analysis by Delta</h2>
<div class="two-col">
<div class="scroll-x"><table>
<tr><th>Delta</th><th>Avg Return</th><th>Avg Sharpe</th><th>Avg Called Away</th>
<th>Avg Prem Yield</th><th>Avg OTM %</th><th>Avg Win Rate</th></tr>
{rows_html}
</table></div>
<div>
<div class="chart-wrap chart-sm"><canvas id="deltaChart"></canvas></div>
<script>
new Chart(document.getElementById('deltaChart'), {{
    type: 'bar',
    data: {{
        labels: {labels},
        datasets: [
            {{ label: 'Avg Sharpe', data: {sharpe_data}, backgroundColor: 'rgba(168,85,247,0.7)', yAxisID: 'y' }},
            {{ label: 'Called Away %', data: {called_data}, backgroundColor: 'rgba(239,68,68,0.5)', yAxisID: 'y1' }}
        ]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
            y: {{ position: 'left', ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }},
                  title: {{ display: true, text: 'Sharpe', color: '#94a3b8' }} }},
            y1: {{ position: 'right', ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }},
                   title: {{ display: true, text: 'Called %', color: '#94a3b8' }} }},
            x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
    }}
}});
</script>
</div></div></div>"""

    def _management_analysis(self, df):
        def label_mgmt(row):
            pt = f"PT{int(row['profit_taking']*100)}%" if pd.notna(row['profit_taking']) else "NoPT"
            sl = f"SL{row['stop_loss']:.0f}x" if pd.notna(row['stop_loss']) else "NoSL"
            return f"{pt}+{sl}"

        df2 = df.copy()
        df2["mgmt"] = df2.apply(label_mgmt, axis=1)
        mgmt_agg = df2.groupby("mgmt").agg(
            avg_return=("total_return", "mean"),
            avg_sharpe=("sharpe", "mean"),
            avg_called=("called_away_pct", "mean"),
            avg_premium=("ann_premium_yield", "mean"),
            avg_win=("win_rate", "mean"),
            avg_dd=("max_drawdown", "mean"),
        ).round(2).sort_values("avg_sharpe", ascending=False)

        rows_html = ""
        for mgmt in mgmt_agg.index:
            r = mgmt_agg.loc[mgmt]
            rows_html += f"""<tr>
<td>{mgmt}</td>
<td class="{'positive' if r['avg_return'] > 0 else 'negative'}">{r['avg_return']:+.1f}%</td>
<td><strong>{r['avg_sharpe']:.3f}</strong></td>
<td>{r['avg_dd']:.1f}%</td><td>{r['avg_win']:.0f}%</td>
<td>{r['avg_called']:.0f}%</td><td>{r['avg_premium']:.1f}%</td></tr>"""

        return f"""<div class="section">
<h2>Analysis by Management Rules (Profit Taking × Stop Loss)</h2>
<div class="scroll-x"><table>
<tr><th>Management</th><th>Avg Return</th><th>Avg Sharpe</th><th>Avg Max DD</th>
<th>Avg Win Rate</th><th>Avg Called Away</th><th>Avg Prem Yield</th></tr>
{rows_html}
</table></div></div>"""

    def _called_away_section(self, df):
        ca_pivot = df.pivot_table(
            index="dte", columns="delta", values="called_away_pct", aggfunc="mean"
        ).round(1)
        cols = sorted(ca_pivot.columns) if not ca_pivot.empty else []
        header = '<div class="heatmap-cell heatmap-header">DTE \\ Delta</div>'
        for c in cols:
            header += f'<div class="heatmap-cell heatmap-header">Δ {c:.2f}</div>'
        rows_html = ""
        for dte in sorted(ca_pivot.index):
            rows_html += f'<div class="heatmap-cell heatmap-header">{dte}d</div>'
            for c in cols:
                val = ca_pivot.loc[dte, c] if c in ca_pivot.columns else 0
                intensity = min(val / 60, 1.0)
                r = int(60 + 180 * intensity)
                g = int(200 - 130 * intensity)
                b = int(100 - 20 * intensity)
                rows_html += (
                    f'<div class="heatmap-cell" style="background:rgba({r},{g},{b},0.65);color:#fff">'
                    f'{val:.0f}%</div>'
                )
        ncols = len(cols) + 1
        return f"""<div class="section">
<h2>Called Away Frequency (% of Trades)</h2>
<p style="color:var(--dim);margin-bottom:12px;">
Averaged across all management rules. Red = more frequent calling away.</p>
<div class="heatmap" style="grid-template-columns: repeat({ncols}, 1fr);">
{header}{rows_html}
</div>
<div class="insight">
Lower delta and shorter DTE reduce called-away risk. Called-away events reduce
upside participation in a bull market but generate additional premium income from
more frequent re-entry.
</div></div>"""

    def _qqqi_comparison(self, qqqi, bh, best):
        if not qqqi or not qqqi.get("available"):
            return """<div class="section"><h2>QQQI Comparison</h2>
<p style="color:var(--dim);">QQQI data not available. QQQI (NEOS QQQ High Income ETF) launched Jan 2024.</p></div>"""

        qsp = qqqi["qqq_same_period"]
        bcc = qqqi.get("best_cc_same", {}) or {}

        qqqi_eq = qqqi.get("equity_curve", {})
        qqq_eq = qsp.get("equity_curve", {})
        cc_eq = bcc.get("equity_curve", {}) if bcc else {}

        all_dates = sorted(set(list(qqqi_eq.keys()) + list(qqq_eq.keys()) + list(cc_eq.keys())))
        labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in all_dates]
        qqqi_vals = [round(qqqi_eq.get(d, None) or 0, 0) if d in qqqi_eq else None for d in all_dates]
        qqq_vals = [round(qqq_eq.get(d, None) or 0, 0) if d in qqq_eq else None for d in all_dates]
        cc_vals = [round(cc_eq.get(d, None) or 0, 0) if d in cc_eq else None for d in all_dates]

        datasets = [
            {"label": "QQQI (Total Return)", "data": qqqi_vals,
             "borderColor": "#ec4899", "borderWidth": 2.2, "pointRadius": 0, "fill": False, "spanGaps": True},
            {"label": f"QQQ Buy & Hold (same period)", "data": qqq_vals,
             "borderColor": "#f59e0b", "borderWidth": 2, "pointRadius": 0, "fill": False, "spanGaps": True},
        ]
        if cc_vals and any(v for v in cc_vals if v):
            datasets.append(
                {"label": f"Best CC ({bcc.get('label', '')})", "data": cc_vals,
                 "borderColor": "#22c55e", "borderWidth": 1.8, "pointRadius": 0, "fill": False, "spanGaps": True}
            )

        ds_json = json.dumps(datasets)
        labels_json = json.dumps(labels)

        price_chg = qqqi["last_price"] - qqqi["first_price"]
        price_pct = qqqi["price_return"]
        div_pct = (qqqi["total_dividends"] / INITIAL_INVESTMENT) * 100

        return f"""<div class="section">
<h2>QQQI vs QQQ vs Covered Call — Same Period Comparison</h2>
<p style="color:var(--dim);margin-bottom:16px;">
Apples-to-apples comparison over <strong>{qqqi['start']} to {qqqi['end']}</strong>
({qqqi['days']} trading days, QQQI inception period).
All strategies start with ${INITIAL_INVESTMENT:,.0f}. QQQI dividends are reinvested.</p>

<div class="kpi-grid" style="margin-bottom:20px;">
<div class="kpi" style="border-left-color:var(--pink);">
    <div class="label">QQQI Total Return</div>
    <div class="value">{qqqi['total_return']:+.1f}%</div>
    <div class="detail">
        Price: ${qqqi['first_price']:.2f} → ${qqqi['last_price']:.2f} ({price_pct:+.1f}%)<br>
        Dividends: ${qqqi['total_dividends']:,.0f} ({div_pct:.1f}% of capital)<br>
        Yield: {qqqi['dividend_yield_ann']:.1f}%/yr
    </div>
</div>
<div class="kpi orange">
    <div class="label">QQQ Same Period</div>
    <div class="value">{qsp['total_return']:+.1f}%</div>
    <div class="detail">Sharpe: {qsp['sharpe']:.3f}<br>Max DD: {qsp['max_drawdown']:.1f}%</div>
</div>
<div class="kpi green">
    <div class="label">Best CC Same Period</div>
    <div class="value">{bcc.get('same_ret', 0):+.1f}%</div>
    <div class="detail">{bcc.get('label', 'N/A')}<br>Sharpe: {bcc.get('same_sharpe', 0):.3f}</div>
</div>
</div>

<h3>Return Decomposition: QQQI</h3>
<div class="scroll-x"><table>
<tr><th>Component</th><th>Amount ($100K invested)</th><th>% of Capital</th><th>% of Total Return</th></tr>
<tr><td>Price Appreciation</td>
    <td class="{'positive' if price_chg > 0 else 'negative'}">${price_chg * INITIAL_INVESTMENT / qqqi['first_price']:,.0f}</td>
    <td>{price_pct:+.1f}%</td>
    <td>{price_pct / qqqi['total_return'] * 100 if qqqi['total_return'] != 0 else 0:.0f}%</td></tr>
<tr><td>Distributions (reinvested)</td>
    <td class="positive">${qqqi['total_dividends']:,.0f}</td>
    <td>{div_pct:.1f}%</td>
    <td>{div_pct / qqqi['total_return'] * 100 if qqqi['total_return'] != 0 else 0:.0f}%</td></tr>
<tr style="font-weight:700;border-top:2px solid var(--surface2)">
    <td>Total Return</td><td>${qqqi['final_value'] - INITIAL_INVESTMENT:,.0f}</td>
    <td>{qqqi['total_return']:+.1f}%</td><td>100%</td></tr>
</table></div>

<h3>Risk-Adjusted Comparison (Same Period)</h3>
<div class="scroll-x"><table>
<tr><th>Metric</th><th>QQQI</th><th>QQQ B&H</th><th>Best CC</th></tr>
<tr><td>Total Return</td>
    <td>{qqqi['total_return']:+.1f}%</td>
    <td>{qsp['total_return']:+.1f}%</td>
    <td>{bcc.get('same_ret', 0):+.1f}%</td></tr>
<tr><td>CAGR</td>
    <td>{qqqi['cagr']:+.1f}%</td>
    <td>{qsp['cagr']:+.1f}%</td>
    <td>{bcc.get('same_cagr', 0):+.1f}%</td></tr>
<tr><td>Ann. Volatility</td>
    <td>{qqqi['ann_volatility']:.1f}%</td>
    <td>{qsp['ann_volatility']:.1f}%</td>
    <td>{bcc.get('same_vol', 0):.1f}%</td></tr>
<tr><td>Sharpe Ratio</td>
    <td><strong>{qqqi['sharpe']:.3f}</strong></td>
    <td><strong>{qsp['sharpe']:.3f}</strong></td>
    <td><strong>{bcc.get('same_sharpe', 0):.3f}</strong></td></tr>
<tr><td>Max Drawdown</td>
    <td>{qqqi['max_drawdown']:.1f}%</td>
    <td>{qsp['max_drawdown']:.1f}%</td>
    <td>{bcc.get('same_dd', 0):.1f}%</td></tr>
<tr><td>After-Tax Value*</td>
    <td>${qqqi['after_tax_value']:,.0f}</td>
    <td>—</td><td>—</td></tr>
</table></div>
<p style="color:var(--dim);font-size:0.82em;margin-top:4px;">* QQQI after-tax assumes distributions taxed at Section 1256 rate ({SECTION_1256_RATE*100:.1f}%).</p>

<h3>Equity Curves — Same Period</h3>
<div class="chart-wrap"><canvas id="qqqiChart"></canvas></div>
<script>
new Chart(document.getElementById('qqqiChart'), {{
    type: 'line',
    data: {{ labels: {labels_json}, datasets: {ds_json} }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
            x: {{ ticks: {{ color: '#94a3b8', maxTicksLimit: 15 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
            y: {{ ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
                  grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }},
        interaction: {{ mode: 'index', intersect: false }},
    }}
}});
</script>

<div class="insight">
<strong>Key Takeaway:</strong> QQQI's total return includes both price change ({price_pct:+.1f}%)
and reinvested distributions (${qqqi['total_dividends']:,.0f}, ~{qqqi['dividend_yield_ann']:.1f}%/yr).
While QQQI provides steady income via distributions, its NAV growth is naturally suppressed
because premium income that would otherwise accrue to NAV is paid out as distributions.
QQQ's {qsp['total_return']:+.1f}% return over the same period captures full equity upside.
The "best" choice depends on whether you prioritize <em>income</em> (QQQI) or <em>total return</em> (QQQ/CC).
</div>
</div>"""

    def _tax_section(self, df, bh, best, qqqi=None):
        net_prem = best["net_premium"]
        ord_tax = net_prem * ORDINARY_TAX_RATE
        s1256_tax = net_prem * SECTION_1256_RATE
        savings = ord_tax - s1256_tax

        bh_gain = max(bh["final_value"] - INITIAL_INVESTMENT, 0)
        bh_tax = bh_gain * LTCG_TAX_RATE + bh["dividend_income"] * QQQ_DIVIDEND_TAX_RATE
        bh_after = bh["final_value"] - bh_tax

        cc_gain = max(best["final_value"] - INITIAL_INVESTMENT - net_prem, 0)
        cc_stock_tax = cc_gain * LTCG_TAX_RATE
        cc_prem_tax_ord = ord_tax
        cc_div_tax = best["dividend_income"] * QQQ_DIVIDEND_TAX_RATE
        cc_total_tax = cc_stock_tax + cc_prem_tax_ord + cc_div_tax
        cc_after = best["final_value"] - cc_total_tax

        cc_prem_tax_1256 = s1256_tax
        cc_total_tax_1256 = cc_stock_tax + cc_prem_tax_1256 + cc_div_tax
        cc_after_1256 = best["final_value"] - cc_total_tax_1256

        qqqi_col = ""
        qqqi_hdr = ""
        if qqqi and qqqi.get("available"):
            qqqi_hdr = "<th>QQQI (Actual)</th>"
            qqqi_income_tax = qqqi.get("income_tax", 0)
            qqqi_final = qqqi.get("final_value", 0)
            qqqi_after = qqqi.get("after_tax_value", 0)
            qqqi_total_tax = qqqi_income_tax
            qqqi_col = f"""<td>${qqqi_final:,.0f} <span class="tag tag-blue">since {qqqi['start']}</span></td>"""
            qqqi_col2 = f"<td>—</td>"
            qqqi_col3 = f'<td>${qqqi_income_tax:,.0f} <span class="tag tag-green">Sec 1256</span></td>'
            qqqi_col4 = f"<td>—</td>"
            qqqi_col5 = f"<td>${qqqi_total_tax:,.0f}</td>"
            qqqi_col6 = f'<td style="color:var(--green)">${qqqi_after:,.0f}</td>'
            qqqi_col7 = f"<td>—</td>"
        else:
            qqqi_col = qqqi_col2 = qqqi_col3 = qqqi_col4 = qqqi_col5 = qqqi_col6 = qqqi_col7 = ""

        return f"""<div class="section">
<h2>Tax Impact Analysis — Full Backtest Period</h2>
<p style="color:var(--dim);margin-bottom:16px;">
CC premiums are taxed as <strong>ordinary income</strong> (37% marginal).
QQQI uses Section 1256 contracts: 60% LTCG + 40% ordinary = ~{SECTION_1256_RATE*100:.1f}% blended rate.
Stock gains held &gt;1yr qualify for LTCG ({LTCG_TAX_RATE*100:.1f}%).</p>
<div class="scroll-x"><table>
<tr><th>Item</th><th>Buy & Hold</th><th>Best CC (Ordinary)</th><th>Best CC (if Sec 1256)</th>{qqqi_hdr}</tr>
<tr><td>Pre-Tax Value</td><td>${bh['final_value']:,.0f}</td><td>${best['final_value']:,.0f}</td><td>${best['final_value']:,.0f}</td>{qqqi_col}</tr>
<tr><td>Stock LTCG Tax</td><td>${bh_gain*LTCG_TAX_RATE:,.0f}</td><td>${cc_stock_tax:,.0f}</td><td>${cc_stock_tax:,.0f}</td>{qqqi_col2}</tr>
<tr><td>Premium / Dist. Tax</td><td>$0</td><td>${cc_prem_tax_ord:,.0f} <span class="tag tag-red">37%</span></td><td>${cc_prem_tax_1256:,.0f} <span class="tag tag-green">{SECTION_1256_RATE*100:.1f}%</span></td>{qqqi_col3}</tr>
<tr><td>Dividend Tax</td><td>${bh['dividend_income']*QQQ_DIVIDEND_TAX_RATE:,.0f}</td><td>${cc_div_tax:,.0f}</td><td>${cc_div_tax:,.0f}</td>{qqqi_col4}</tr>
<tr style="font-weight:700;border-top:2px solid var(--surface2)">
<td>Total Tax</td><td>${bh_tax:,.0f}</td><td>${cc_total_tax:,.0f}</td><td>${cc_total_tax_1256:,.0f}</td>{qqqi_col5}</tr>
<tr style="font-weight:700;color:var(--green)">
<td>After-Tax Value</td><td>${bh_after:,.0f}</td><td>${cc_after:,.0f}</td><td>${cc_after_1256:,.0f}</td>{qqqi_col6}</tr>
<tr><td>Sec 1256 Savings vs Ordinary</td><td>—</td><td>—</td><td class="positive">${savings:,.0f}</td>{qqqi_col7}</tr>
</table></div>
<div class="insight">
Section 1256 treatment (as used by QQQI) saves <strong>${savings:,.0f}</strong> on ${net_prem:,.0f} net premium
vs ordinary income. The blended 1256 rate is {SECTION_1256_RATE*100:.1f}% vs 37% ordinary —
a {(ORDINARY_TAX_RATE-SECTION_1256_RATE)*100:.1f}pp advantage per dollar of premium income.
For a $100K portfolio generating ~${net_prem:,.0f} in CC premium over {(END_DATE-START_DATE).days//365} years,
the annual tax savings would be ~${savings / ((END_DATE-START_DATE).days/365.25):,.0f}/yr.
</div></div>"""

    def _full_results_table(self, df, bh):
        sorted_df = df.sort_values("sharpe", ascending=False)
        rows_html = ""
        for idx, row in sorted_df.iterrows():
            pt_s = f"{int(row['profit_taking']*100)}%" if pd.notna(row['profit_taking']) else "—"
            sl_s = f"{row['stop_loss']:.0f}x" if pd.notna(row['stop_loss']) else "—"
            diff = row['total_return'] - bh['total_return']
            diff_cls = "positive" if diff >= 0 else "negative"
            rows_html += f"""<tr>
<td>{int(row['dte'])}d</td><td>Δ{row['delta']:.2f}</td><td>{pt_s}</td><td>{sl_s}</td>
<td class="{'positive' if row['total_return']>0 else 'negative'}">{row['total_return']:+.1f}%</td>
<td class="{diff_cls}">{diff:+.1f}%</td>
<td>{row['cagr']:+.1f}%</td><td>{row['ann_volatility']:.1f}%</td>
<td><strong>{row['sharpe']:.3f}</strong></td><td>{row['max_drawdown']:.1f}%</td>
<td>{row['win_rate']:.0f}%</td>
<td>{int(row['called_away'])}/{int(row['num_trades'])}</td>
<td>{row['ann_premium_yield']:.1f}%</td>
<td>${row['final_value']:,.0f}</td></tr>"""

        return f"""<div class="section">
<h2>Full Results Table ({len(sorted_df)} Combinations)</h2>
<p style="color:var(--dim);margin-bottom:10px;">Sorted by Sharpe Ratio (descending). "vs B&H" = return difference vs Buy & Hold ({bh['total_return']:+.1f}%).</p>
<div class="scroll-x" style="max-height:600px;overflow-y:auto;"><table>
<tr><th>DTE</th><th>Delta</th><th>PT</th><th>SL</th><th>Return</th><th>vs B&H</th>
<th>CAGR</th><th>Vol</th><th>Sharpe</th><th>Max DD</th><th>Win%</th>
<th>Called</th><th>Yield</th><th>Final $</th></tr>
{rows_html}
</table></div></div>"""

    def _trade_log(self, best_bt, best_sharpe):
        if best_bt is None or not best_bt.trades:
            return ""
        rows_html = ""
        for t in best_bt.trades[:60]:
            reason_cls = "negative" if t.called_away else ("positive" if t.exit_reason == "expired_otm" else "")
            reason_lbl = {
                "called_away": "CALLED",
                "expired_otm": "Expired OTM",
                "profit_taking": "Profit Take",
                "stop_loss": "Stop Loss",
            }.get(t.exit_reason, t.exit_reason)
            rows_html += f"""<tr>
<td>{t.entry_date}</td><td>{t.exit_date}</td><td>{t.expiration}</td>
<td>${t.entry_stock_price:.0f}</td><td>${t.strike:.0f}</td><td>{t.otm_pct:+.1f}%</td>
<td>${t.entry_premium:.2f}</td><td>{t.holding_days}d</td>
<td class="{'positive' if t.pnl_per_share > 0 else 'negative'}">${t.pnl_per_share:.2f}</td>
<td class="{reason_cls}">{reason_lbl}</td></tr>"""

        return f"""<div class="section">
<h2>Trade Log — Best Strategy: {best_sharpe['label']}</h2>
<p style="color:var(--dim);margin-bottom:10px;">Showing first 60 trades.</p>
<div class="scroll-x" style="max-height:500px;overflow-y:auto;"><table>
<tr><th>Entry</th><th>Exit</th><th>Expiry</th><th>Stock</th><th>Strike</th><th>OTM%</th>
<th>Premium</th><th>Hold</th><th>P&L/sh</th><th>Result</th></tr>
{rows_html}
</table></div></div>"""

    def _insights(self, df, bh, best):
        best_dte = df.groupby("dte")["sharpe"].mean().idxmax()
        best_delta = df.groupby("delta")["sharpe"].mean().idxmax()

        low_ca = df[df["called_away_pct"] < 15].nlargest(3, "sharpe")
        low_ca_text = ""
        for _, r in low_ca.iterrows():
            low_ca_text += f"<li>{r['label']}: Sharpe {r['sharpe']:.3f}, Called {r['called_away_pct']:.0f}%, Return {r['total_return']:+.1f}%</li>"

        return f"""<div class="section">
<h2>Key Insights & Recommendations</h2>
<div class="insight">
<strong>1. Optimal DTE:</strong> {best_dte}-day expirations deliver the best average risk-adjusted returns
across all delta/management combinations. Shorter DTEs collect more premium through higher theta decay
but incur more transaction costs and gamma risk.
</div>
<div class="insight">
<strong>2. Optimal Delta:</strong> Δ {best_delta:.2f} offers the best Sharpe ratio on average.
Lower deltas preserve more upside but collect less premium. Higher deltas generate more income
but get called away more frequently.
</div>
<div class="insight">
<strong>3. CC vs Buy & Hold:</strong> Over this {(END_DATE-START_DATE).days//365}-year period,
the best CC strategy returned {best['total_return']:+.1f}% vs Buy & Hold's {bh['total_return']:+.1f}%
({best['total_return']-bh['total_return']:+.1f}% difference). In a strong bull market, covered calls
typically underperform pure equity due to capped upside, but provide lower volatility
({best['ann_volatility']:.1f}% vs {bh['ann_volatility']:.1f}%) and better drawdown protection
({best['max_drawdown']:.1f}% vs {bh['max_drawdown']:.1f}%).
</div>
<div class="insight">
<strong>4. Best Low Called-Away Strategies (under 15%):</strong>
<ul>{low_ca_text}</ul>
These preserve equity upside while generating meaningful premium income.
</div>
<div class="insight">
<strong>5. Tax Efficiency:</strong> Using Section 1256 treatment (QQQI-style) saves
${best['sec1256_savings']:,.0f} on ${best['net_premium']:,.0f} net premium vs ordinary income.
For high-income investors, this {(ORDINARY_TAX_RATE-SECTION_1256_RATE)*100:.1f}pp rate difference
is significant and favors synthetic CC products over DIY covered calls.
</div>
</div>
<div class="footer">
<p>QQQ Covered Call Grid Search Backtest Report | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p>Option pricing: Black-Scholes with VIX-calibrated implied volatility and skew adjustment</p>
<p>Past performance does not guarantee future results. This is for educational purposes only.</p>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def print_console_report(engine: GridSearchEngine):
    df = engine.results_df
    bh = engine.buy_hold
    W = 120

    print(f"\n{'═' * W}")
    print(f"  QQQ COVERED CALL GRID SEARCH — BACKTEST RESULTS")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Capital: ${INITIAL_INVESTMENT:,.0f} | DTE: {DTE_GRID} | Delta: {DELTA_GRID}")
    print(f"{'═' * W}")

    print(f"\n  Buy & Hold: Return={bh['total_return']:+.1f}%  Sharpe={bh['sharpe']:.3f}  "
          f"Final=${bh['final_value']:,.0f}")

    print(f"\n  TOP 10 BY SHARPE RATIO:")
    print(f"  {'Rank':<5} {'Strategy':<35} {'Return':>8} {'Sharpe':>8} "
          f"{'MaxDD':>7} {'Win%':>6} {'Called':>8} {'Final$':>12}")
    print(f"  {'─' * 95}")

    top10 = engine.get_top_n(10, "sharpe")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {rank:<5} {row['label']:<35} {row['total_return']:>+7.1f}% "
              f"{row['sharpe']:>8.3f} {row['max_drawdown']:>+6.1f}% "
              f"{row['win_rate']:>5.0f}% {row['called_away']:>3.0f}/{row['num_trades']:>3.0f} "
              f"${row['final_value']:>11,.0f}")

    print(f"\n  AVERAGES BY DTE:")
    dte_agg = df.groupby("dte").agg(
        ret=("total_return", "mean"), sharpe=("sharpe", "mean"),
        called=("called_away_pct", "mean"), prem=("ann_premium_yield", "mean"),
    ).round(2)
    for dte in sorted(dte_agg.index):
        r = dte_agg.loc[dte]
        print(f"    DTE {dte:>2}d: Return={r['ret']:>+6.1f}%  Sharpe={r['sharpe']:.3f}  "
              f"Called={r['called']:.0f}%  PremYield={r['prem']:.1f}%")

    print(f"\n  AVERAGES BY DELTA:")
    delta_agg = df.groupby("delta").agg(
        ret=("total_return", "mean"), sharpe=("sharpe", "mean"),
        called=("called_away_pct", "mean"), prem=("ann_premium_yield", "mean"),
    ).round(2)
    for delta in sorted(delta_agg.index):
        r = delta_agg.loc[delta]
        print(f"    Δ {delta:.2f}: Return={r['ret']:>+6.1f}%  Sharpe={r['sharpe']:.3f}  "
              f"Called={r['called']:.0f}%  PremYield={r['prem']:.1f}%")

    print(f"\n{'═' * W}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  QQQ COVERED CALL GRID SEARCH BACKTEST")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Capital: ${INITIAL_INVESTMENT:,.0f}")
    print(f"  Grid: {len(DTE_GRID)}×{len(DELTA_GRID)}×{len(PROFIT_TAKING_GRID)}×{len(STOP_LOSS_GRID)} "
          f"= {len(DTE_GRID)*len(DELTA_GRID)*len(PROFIT_TAKING_GRID)*len(STOP_LOSS_GRID)} combinations")
    print("=" * 80)

    dm = DataManager()
    dm.fetch_all()

    engine = GridSearchEngine(dm)
    engine.run_buy_and_hold()
    engine.run_grid_search()
    engine.run_qqqi_comparison()

    dm.print_data_stats()
    dm._save_disk_cache()

    print_console_report(engine)

    report = ReportGenerator(engine, dm)
    report.generate()

    results_data = {
        "config": {
            "symbol": SYMBOL, "initial": INITIAL_INVESTMENT,
            "start": str(START_DATE), "end": str(END_DATE),
            "dte_grid": DTE_GRID, "delta_grid": DELTA_GRID,
            "pt_grid": [x if x else None for x in PROFIT_TAKING_GRID],
            "sl_grid": [x if x else None for x in STOP_LOSS_GRID],
            "data_source": "ThetaData real chains" if dm.has_real_data() else "Black-Scholes",
        },
        "buy_hold": {k: v for k, v in engine.buy_hold.items() if k != "equity_curve"},
        "top10": engine.get_top_n(10, "sharpe").to_dict("records"),
    }
    results_path = os.path.join(WORK_DIR, "qqq_cc_grid_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"[Results] JSON saved → {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
