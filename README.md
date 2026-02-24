# QQQ Covered Call Grid Search

## Motivation

This project is interest-driven research on one practical question:

- Can a covered call (CC) overlay on long QQQ generate meaningful cash flow?
- After assignment drag and taxes, is the strategy still worthwhile versus buy-and-hold?
- What parameter combination is most effective in a full market cycle?

The goal is to validate CC cash-flow feasibility with backtesting and find robust parameters, not rely on generic rules.

## Methods

The strategy framework:

1. Buy QQQ on `2021-01-04` with `$100,000`.
2. Continuously sell OTM covered calls.
3. If called away, cash-settle and re-buy QQQ to maintain exposure.
4. Run a grid search across DTE, delta, profit-taking, and stop-loss.
5. Evaluate each parameter set by return, Sharpe, drawdown, premium income, and called-away frequency.

Model and data:

- Underlying prices/dividends: `yfinance`
- Option chain calibration: `ThetaData` 
- Pricing engine: Black-Scholes with skew adjustment
- Backtest range: `2021-01-04` to `2025-12-31`
- Baseline: QQQ buy-and-hold over the same period

## Parameters

Grid used in this project (`5 x 4 x 3 x 3 = 180` combinations):

- DTE grid: `[7, 14, 21, 30, 45]`
- Delta grid: `[0.10, 0.15, 0.20, 0.25]`
- Profit-taking grid: `[None, 0.50, 0.60]`
- Stop-loss grid: `[None, 2.0, 3.0]`

Core assumptions:

- Risk-free rate: `4.0%`
- Slippage: `1%`
- Commission: `$0.65` per contract
- Tax model included (ordinary income, LTCG + NIIT, section 1256 comparison)

## Key Results

Buy-and-hold benchmark (QQQ):

- Total return: `+95.1%`
- CAGR: `14.34%`
- Sharpe: `0.554`
- Max drawdown: `-32.97%`
- Final value: `$195,103`

Best overall strategy in this run (top Sharpe + strong total return):

- Label: `DTE45_D0.10_PT50%_NoSL`
- Total return: `+104.5%`
- CAGR: `15.42%`
- Sharpe: `0.669`
- Max drawdown: `-27.75%`
- Called away: `7 / 99` trades (`7.1%`)
- Net premium: `$23,626.51`
- Final value: `$204,497.35`

Additional useful observation:

- A higher-delta setup (`DTE45_D0.25_PT60%_NoSL`) produced much higher net premium (`$66,540.01`) but weaker total return (`+91.43%`), showing the classic trade-off between cash-flow intensity and upside retention.

## Conclusion

✅ In this backtest window, CC cash-flow generation on QQQ is feasible and can outperform buy-and-hold when parameters are calibrated.

The evidence from this run suggests:

- 🎯 Lower delta + longer DTE + moderate profit-taking gave the best risk-adjusted result.
- ⚠️ Chasing maximum premium alone tends to increase assignment drag and can reduce total return.
- 🧩 Parameter selection matters more than using a one-size-fits-all CC rule.

## Files

- `qqq_cc_grid_backtest.py`: Main grid-search engine
- `qqq_cc_fetch_thetadata.py`: ThetaData fetch helper
- `qqq_cc_dryrun.py`: Dry-run utility
- `qqq_cc_grid_results.json`: Full result dataset
- `qqq_cc_grid_cache.json`: Cache data
- `QQQ_CC_Grid_Search_Report.html`: Visual report

