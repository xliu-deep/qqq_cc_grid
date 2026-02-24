"""
Dry-run: identify all unique (entry_date, expiration) pairs needed for the
QQQ Covered Call grid search backtest. Outputs them as JSON for MCP fetching.
"""
import json
import math
import warnings
from datetime import date, timedelta
from itertools import product
from typing import Optional, List, Dict, Set

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

SYMBOL = "QQQ"
INITIAL_INVESTMENT = 100_000
START_DATE = date(2021, 1, 4)
END_DATE = date(2025, 12, 31)
RISK_FREE_RATE = 0.04
SLIPPAGE_PCT = 0.01

DTE_GRID = [7, 14, 21, 30, 45]
DELTA_GRID = [0.10, 0.15, 0.20, 0.25]
PROFIT_TAKING_GRID = [None, 0.50, 0.60]
STOP_LOSS_GRID = [None, 2.0, 3.0]

EXPIRATIONS_FILE = "qqq_expirations.json"
PAIRS_FILE = "qqq_cc_needed_pairs.json"


class BSEngine:
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

    @classmethod
    def strike_for_delta(cls, S, target_delta, T, r, sigma):
        if T <= 0 or sigma <= 0 or target_delta <= 0 or target_delta >= 1:
            return S * 1.02
        d1_target = norm.ppf(target_delta)
        K_raw = S * math.exp((r + 0.5 * sigma ** 2) * T - d1_target * sigma * math.sqrt(T))
        if S < 50:
            return round(K_raw * 2) / 2
        elif S < 200:
            return round(K_raw)
        elif S < 500:
            return round(K_raw / 2) * 2
        else:
            return round(K_raw / 5) * 5

    @classmethod
    def price_with_skew(cls, S, K, T, r, base_sigma):
        moneyness = K / S
        skew_adj = base_sigma * 0.12 * (1.0 - moneyness)
        adj_sigma = max(base_sigma + skew_adj, 0.05)
        return cls.call_price(S, K, T, r, adj_sigma), adj_sigma


def main():
    print("[DryRun] Fetching QQQ data from Yahoo...")
    tkr = yf.Ticker(SYMBOL)
    hist = tkr.history(
        start=(START_DATE - timedelta(days=120)).strftime("%Y-%m-%d"),
        end=(END_DATE + timedelta(days=5)).strftime("%Y-%m-%d"),
        auto_adjust=False,
    )
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist.index = hist.index.normalize()

    vix = yf.download("^VIX",
                       start=(START_DATE - timedelta(days=120)).strftime("%Y-%m-%d"),
                       end=(END_DATE + timedelta(days=5)).strftime("%Y-%m-%d"),
                       progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    vix.index = vix.index.normalize()

    mask = (hist.index >= pd.Timestamp(START_DATE)) & (hist.index <= pd.Timestamp(END_DATE))
    trading_dates = [d.date() for d in hist.index[mask]]
    print(f"  → {len(trading_dates)} trading days")

    expirations = json.load(open(EXPIRATIONS_FILE))
    exp_dates = sorted(set(e["expiration"] for e in expirations
                           if "2021" <= e["expiration"] <= "2026-12-31"))
    print(f"  → {len(exp_dates)} expirations in range")

    def get_price(d):
        ts = pd.Timestamp(d)
        if ts in hist.index:
            return float(hist.loc[ts, "Close"])
        m = hist.index <= ts
        if m.any():
            return float(hist.loc[m].iloc[-1]["Close"])
        return None

    vol_cache = {}

    def get_vol(d):
        if d in vol_cache:
            return vol_cache[d]
        ts = pd.Timestamp(d)
        m = vix.index <= ts
        if m.any():
            v = float(vix.loc[m].iloc[-1]["Close"]) / 100.0 * 1.05
        else:
            v = 0.25
        vol_cache[d] = v
        return v

    def find_expiration(entry_date, target_dte):
        target = entry_date + timedelta(days=target_dte)
        best_exp = None
        best_diff = 999
        for exp_str in exp_dates:
            try:
                exp_d = date.fromisoformat(exp_str)
            except ValueError:
                continue
            actual_dte = (exp_d - entry_date).days
            if actual_dte < 1:
                continue
            diff = abs(actual_dte - target_dte)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp_d
        return best_exp

    needed_pairs: Set[tuple] = set()
    bs = BSEngine()

    combos = list(product(DTE_GRID, DELTA_GRID, PROFIT_TAKING_GRID, STOP_LOSS_GRID))
    print(f"[DryRun] Simulating {len(combos)} combos to find needed pairs...")

    for idx, (dte, delta, pt, sl) in enumerate(combos):
        first_price = get_price(trading_dates[0])
        shares = max((INITIAL_INVESTMENT // (first_price * 100)) * 100, 100)
        contracts = shares // 100

        position = None  # (entry_date, expiration, strike, premium, entry_price, iv)

        for today in trading_dates:
            price = get_price(today)
            if price is None:
                continue

            if position is not None:
                entry_d, exp, strike, premium, entry_px, iv = position

                if today >= exp:
                    position = None
                else:
                    rem_dte = (exp - today).days
                    T = max(rem_dte / 365.0, 1e-6)
                    sigma = iv
                    cv = bs.call_price(price, strike, T, RISK_FREE_RATE, sigma)

                    if pt is not None:
                        target_bb = premium * (1.0 - pt)
                        if cv <= target_bb:
                            position = None

                    if position is not None and sl is not None:
                        max_cost = premium * sl
                        if cv >= max_cost:
                            position = None

            if position is None:
                cutoff = END_DATE - timedelta(days=max(dte // 2, 3))
                if today <= cutoff:
                    exp = find_expiration(today, dte)
                    if exp is None:
                        continue
                    actual_dte = (exp - today).days
                    T = actual_dte / 365.0
                    if T <= 0:
                        continue

                    exp_str = exp.strftime("%Y-%m-%d")
                    query_str = today.strftime("%Y-%m-%d")
                    needed_pairs.add((query_str, exp_str))

                    sigma = get_vol(today)
                    strike = bs.strike_for_delta(price, delta, T, RISK_FREE_RATE, sigma)
                    if strike <= price:
                        step = 1 if price < 200 else (2 if price < 500 else 5)
                        strike = price + step
                    prem, adj_sig = bs.price_with_skew(price, strike, T, RISK_FREE_RATE, sigma)
                    prem *= (1 - SLIPPAGE_PCT)
                    if prem >= 0.01:
                        position = (today, exp, strike, prem, price, adj_sig)

        if (idx + 1) % 30 == 0:
            print(f"  → {idx+1}/{len(combos)} done, {len(needed_pairs)} unique pairs so far")

    pairs_list = sorted([{"query_date": q, "expiration": e} for q, e in needed_pairs],
                        key=lambda x: (x["query_date"], x["expiration"]))

    with open(PAIRS_FILE, "w") as f:
        json.dump({"total": len(pairs_list), "pairs": pairs_list}, f, indent=2)

    print(f"\n[DryRun] DONE: {len(pairs_list)} unique (date, expiration) pairs")
    print(f"  → Saved to {PAIRS_FILE}")

    by_exp = {}
    for p in pairs_list:
        by_exp.setdefault(p["expiration"], []).append(p["query_date"])
    print(f"  → {len(by_exp)} unique expirations")
    print(f"  → Dates per expiration: min={min(len(v) for v in by_exp.values())}, "
          f"max={max(len(v) for v in by_exp.values())}, "
          f"avg={sum(len(v) for v in by_exp.values())/len(by_exp):.1f}")


if __name__ == "__main__":
    main()
