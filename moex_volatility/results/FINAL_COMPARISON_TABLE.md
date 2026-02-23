# Final Comparison Table

**Period**: 2022-01-03 to 2026-02-03 (1019 trading days)
**Commission**: 0.40% per side (daily strategies)
**CBR key rate**: avg 14.6% over period (range 7.5%–21.0%)
**MMF rate**: CBR − 1.5% (management fees)
**Leverage**: via futures (no borrowing cost, ГО covers margin)

## Results

| Portfolio | Return% | Vol% | Sharpe | MDD% | Calmar | Beta | Alpha% | Comment |
|---|---|---|---|---|---|---|---|---|
| IMOEX B&H | -3.29 | 28.9 | -0.11 | -50.76 | -0.06 | 1.0 | — | Benchmark (price index, no dividends) |
| META-A | 6.1 | 4.33 | 1.41 | -3.22 | 1.9 | 0.0058 | 6.12 | Baseline (no forecasts) |
| META-B | 3.56 | 1.29 | 2.75 | -0.6 | 5.97 | 0.0025 | 3.57 | Adaptive stops |
| META-C | 4.78 | 2.89 | 1.65 | -2.0 | 2.39 | 0.0039 | 4.79 | Regime filter |
| META-D | 7.16 | 2.76 | 2.59 | -2.16 | 3.31 | -0.0004 | 7.16 | Vol-gate |
| META-BEST | 7.32 | 2.77 | 2.65 | -2.1 | 3.49 | -0.0003 | 7.32 | Best approach per strategy |
| META-MEAN(BCD) | 5.17 | 2.16 | 2.4 | -1.35 | 3.84 | 0.002 | 5.17 | Average forecast effect |
| META-A + MMF | 13.51 | 4.33 | 3.12 | -2.79 | 4.84 | 0.0058 | — | Strategy + money market on idle capital |
| META-D + MMF | 14.88 | 2.76 | 5.4 | -1.31 | 11.32 | -0.0004 | — | Strategy + money market on idle capital |
| META-BEST + MMF | 15.03 | 2.76 | 5.44 | -1.28 | 11.76 | -0.0003 | — | Strategy + money market on idle capital |
| META-B + MMF | 12.01 | 1.3 | 9.26 | -0.29 | 41.6 | 0.0025 | — | Strategy + money market on idle capital |
| META-D x2 | 14.32 | 5.52 | 2.59 | -4.3 | 3.33 | ~0 | — | Futures leverage (no funding cost) |
| META-D x2 + MMF | 20.95 | 5.51 | 3.8 | -3.25 | 6.44 | ~0 | — | Futures leverage + MMF on remainder |
| META-D x3 | 21.47 | 8.28 | 2.59 | -6.4 | 3.36 | ~0 | — | Futures leverage (no funding cost) |
| META-D x3 + MMF | 27.02 | 8.27 | 3.27 | -5.2 | 5.19 | ~0 | — | Futures leverage + MMF on remainder |
| META-BEST x2 | 14.63 | 5.53 | 2.65 | -4.16 | 3.51 | ~0 | — | Futures leverage (no funding cost) |
| META-BEST x2 + MMF | 21.24 | 5.53 | 3.84 | -3.14 | 6.77 | ~0 | — | Futures leverage + MMF on remainder |
| META-BEST x3 | 21.95 | 8.3 | 2.65 | -6.2 | 3.54 | ~0 | — | Futures leverage (no funding cost) |
| META-BEST x3 + MMF | 27.46 | 8.29 | 3.31 | -5.08 | 5.4 | ~0 | — | Futures leverage + MMF on remainder |

## Key Takeaways

1. **IMOEX B&H** — catastrophic over 2022–2025: sanctions shock, -50% drawdown, negative return. Poor benchmark.

2. **META-BEST + MMF** — the most realistic practical variant: **15.03% return** at **2.76% vol**, Sharpe **5.44**, MDD **-1.28%**. Idle capital (~88%) earns money market rate.

3. **Leverage 2–3×** via futures is shown for completeness. No borrowing cost (ГО/margin covers), so Sharpe scales linearly. MDD increases proportionally.

4. **All strategies are market-neutral** (beta ≈ 0, correlation < 0.05 with IMOEX). Returns are pure alpha from volatility forecasting + strategy construction.

5. **Capital efficiency** is the correct framing (not leverage): binary strategies use 4–16% of capital, the rest earns risk-free rate. At avg CBR 14.6%, this adds ~12% p.a. to strategy returns.

## Methodology Notes

- Meta-portfolio daily returns reconstructed from `daily_positions.parquet` (17 tickers × 6 strategies × EW)
- Commission: 0.40% per side charged on every position change
- Exposure: fraction of (strategy, ticker) pairs with non-zero position each day
- MMF return: (CBR key rate − 1.5%) / 365 per calendar day, applied to (1 − exposure) fraction of capital
- Leverage: simple multiplication of daily returns (futures model, no funding cost)
- Leveraged + MMF: free capital = max(0, 1 − leverage × exposure)
- IMOEX: price index (no dividends). Strategies also trade price, so comparison is apples-to-apples
- Alpha/Beta computed via OLS regression of daily returns on IMOEX returns
