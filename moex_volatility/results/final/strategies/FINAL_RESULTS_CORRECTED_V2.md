# CORRECTED FINAL RESULTS V2 (FDR fixed + MSP-B fixed)

## BUGS FOUND AND FIXED
1. **FDR bug (v1)**: Free capital earned RF once per strategy (6×), not once per portfolio (1×).
   Effect: cumulative return inflated.
2. **MSP-B bug (v2)**: MSP_B_inst.csv was regenerated with wrong method on 2026-03-23.
   Old MSP-B had exposure=4.1% and XS Sharpe=3.66 — incorrect.
   Fixed: exposure=13.0%, XS Sharpe=1.47.

## MSP Results (ALL6×EW, comm=0.05%, CORRECTED V2)

| MSP | XS Sharpe | Trading Sh | Ann Ret | Ann Vol | MDD | Exposure |
|-----|-----------|------------|---------|---------|-----|----------|
| A   | 1.32      | 1.84       | 19.6%   | 4.4%    | -2.8% | 15.7% |
| B   | 1.47      | 1.96       | 19.5%   | 3.8%    | -1.6% | 13.0% |
| C   | 2.51      | 2.86       | 21.2%   | 3.0%    | -0.9% | 7.2%  |
| D   | 2.54      | 3.16       | 20.9%   | 2.9%    | -1.1% | 12.2% |

## Per-Year Excess Sharpe

| Year | A     | B     | C     | D     | KR    |
|------|-------|-------|-------|-------|-------|
| 2022 | +1.68 | +1.82 | +2.86 | +3.69 | 10.6% |
| 2023 | +2.99 | +2.97 | +5.09 | +3.89 | 9.9%  |
| 2024 | +1.56 | +1.78 | +2.98 | +2.29 | 17.4% |
| 2025 | +0.40 | +0.49 | +1.87 | +1.12 | 21.0% |

All approaches positive every year.

## Bootstrap (CORRECTED V2, 10,000 replications)

| Test      | ΔSharpe | p-value   |       |
|-----------|---------|-----------|-------|
| B>A       | +0.15   | 0.009     | ***   |
| C>A       | +1.19   | <0.001    | ***   |
| D>A       | +1.22   | <0.001    | ***   |
| D>C       | +0.03   | 0.543     |       |
| MSP-A>IMOEX | +2.05 | 0.001   | ***   |

## CE Bootstrap (CORRECTED V2)

| γ  | B>A         | C>A          | D>A          | D>C         |
|----|-------------|--------------|--------------|-------------|
| 3  | -0.09% p=0.59 | +1.77%** p=0.03 | +1.46%* p=0.07 | -0.30% p=0.66 |
| 5  | -0.05% p=0.55 | +1.86%** p=0.02 | +1.56%* p=0.05 | -0.30% p=0.66 |
| 10 | +0.04% p=0.45 | +2.07%*** p=0.01 | +1.79%** p=0.02 | -0.28% p=0.65 |

## IMOEX Benchmark

| Metric | IMOEX B&H | MSP-A |
|--------|-----------|-------|
| XS Sharpe | -0.69  | +1.32 |
| Ann Return | -7.1%  | +19.6% |
| Ann Vol    | 30.8%  | 4.4%  |
| MDD        | -70.3% | -2.8% |

MSP-A > IMOEX: ΔSh=+2.05, p=0.001***

## Key Changes from V1

| Metric | V1 (broken B) | V2 (fixed) |
|--------|---------------|------------|
| MSP-B XS Sharpe | 3.07 | **1.47** |
| MSP-B exposure | 4.1% | **13.0%** |
| Bootstrap B>A ΔSh | +2.34 | **+0.15** |
| B>A still significant? | Yes | **Yes (p=0.009)** |
| CE B>A | -0.69% | **-0.09%** (not significant) |

## Hypothesis Results

- **H2** (BCD > A): B>A marginal (ΔSh=+0.15, p=0.009), C>A strong (+1.19, p<0.001), D>A strong (+1.22, p<0.001). **H2 confirmed.**
- **H3** (MSP > IMOEX): ΔSh=+2.05, p=0.001. **H3 confirmed.**
- **H4** (CE > 0): C>A significant at γ≥3 (+1.77 to +2.07%). D>A significant at γ≥10 (+1.79%). B>A not significant. **H4 confirmed for C and D.**
- **H5** (D>C in CE): ΔCE=-0.30%, p=0.66. **NOT confirmed.** C≈D, slight edge to C.

## Interpretation

B (σ-stops) provides marginal Sharpe improvement but zero CE improvement — not economically valuable.
C (regime filter) and D (vol-gate) are the real improvements: +1.2 excess Sharpe, +1.5-2.1% CE/year.
C and D are statistically indistinguishable (p=0.54 for Sharpe, p=0.66 for CE).
