# Data Inventory for Trading Strategies Block

Generated: 2026-02-09 12:41

---

## 1. Tickers

**17 model tickers** (all have 10-min candles, features, and WF predictions):
  AFLT, ALRS, HYDR, IRAO, LKOH, LSRG, MGNT, MOEX, MTLR, MTSS, NVTK, OGKB, PHOR, RTKM, SBER, TATN, VTBR

---

## 2. Daily OHLCV

**File:** `data/ohlcv_daily.parquet`
**Rows:** 31,349
**Columns:** date, ticker, open, high, low, close, volume
**Period:** 2019-01-03 — 2026-02-03
**Tickers:** 17

| Ticker | Days | Date Range | Close Range |
|--------|------|------------|-------------|
| AFLT | 1,846 | 2019-01-03 — 2026-02-03 | 22.44 — 120.30 |
| ALRS | 1,846 | 2019-01-03 — 2026-02-03 | 37.82 — 150.30 |
| HYDR | 1,846 | 2019-01-03 — 2026-02-03 | 0.37 — 1.03 |
| IRAO | 1,844 | 2019-01-03 — 2026-02-03 | 1.99 — 6.50 |
| LKOH | 1,846 | 2019-01-03 — 2026-02-03 | 3539.50 — 8152.00 |
| LSRG | 1,836 | 2019-01-03 — 2026-02-03 | 410.20 — 1148.00 |
| MGNT | 1,846 | 2019-01-03 — 2026-02-03 | 2337.00 — 8444.00 |
| MOEX | 1,846 | 2019-01-03 — 2026-02-03 | 73.19 — 251.94 |
| MTLR | 1,844 | 2019-01-03 — 2026-02-03 | 54.15 — 332.88 |
| MTSS | 1,845 | 2019-01-03 — 2026-02-03 | 163.90 — 349.55 |
| NVTK | 1,842 | 2019-01-03 — 2026-02-03 | 715.80 — 1993.00 |
| OGKB | 1,840 | 2019-01-03 — 2026-02-03 | 0.28 — 0.84 |
| PHOR | 1,844 | 2019-01-03 — 2026-02-03 | 2099.00 — 8908.00 |
| RTKM | 1,846 | 2019-01-03 — 2026-02-03 | 50.42 — 110.98 |
| SBER | 1,846 | 2019-01-03 — 2026-02-03 | 101.50 — 387.60 |
| TATN | 1,844 | 2019-01-03 — 2026-02-03 | 309.80 — 830.00 |
| VTBR | 1,842 | 2019-01-03 — 2026-02-03 | 64.21 — 287.00 |

---

## 3. Hourly OHLCV

**File:** `data/ohlcv_hourly.parquet`
**Rows:** 281,612
**Columns:** datetime, ticker, open, high, low, close, volume
**Period:** 2019-01-03 10:00:00 — 2026-02-03 18:00:00
**Tickers:** 17
**Session:** 10:00 — 18:50 MSK
**Expected bars/day:** 9 (hours 10-18)

**Bars/day stats:** mean=9.0, median=9, min=2, max=9

| Bars/Day | Count | % |
|----------|-------|---|
| 2 | 1 | 0.0% |
| 4 | 79 | 0.3% |
| 5 | 15 | 0.0% |
| 7 | 17 | 0.1% |
| 8 | 33 | 0.1% |
| 9 | 31,204 | 99.5% |

| Ticker | Hourly Bars | Days (approx) |
|--------|-------------|---------------|
| AFLT | 16,581 | ~1,842 |
| ALRS | 16,581 | ~1,842 |
| HYDR | 16,581 | ~1,842 |
| IRAO | 16,563 | ~1,840 |
| LKOH | 16,581 | ~1,842 |
| LSRG | 16,505 | ~1,833 |
| MGNT | 16,581 | ~1,842 |
| MOEX | 16,581 | ~1,842 |
| MTLR | 16,573 | ~1,841 |
| MTSS | 16,565 | ~1,840 |
| NVTK | 16,545 | ~1,838 |
| OGKB | 16,541 | ~1,837 |
| PHOR | 16,563 | ~1,840 |
| RTKM | 16,581 | ~1,842 |
| SBER | 16,581 | ~1,842 |
| TATN | 16,563 | ~1,840 |
| VTBR | 16,546 | ~1,838 |

---

## 4. Walk-Forward Predictions (Hybrid V1_Adaptive)

**File:** `data/predictions_aligned.parquet`
**Rows:** 31,332
**Columns:** date, ticker, rv_pred_h1, rv_actual_h1, rv_pred_h5, rv_actual_h5, rv_pred_h22, rv_actual_h22
**Period:** 2019-01-03 — 2026-02-02
**Tickers:** 17

**Hourly scaling rule:** sigma_hour = sigma_day / sqrt(9)

**h=1:** 31,328 predictions, mean=0.000420, std=0.000603, min=0.000054, max=0.017849
**h=5:** 31,260 predictions, mean=0.000389, std=0.000312, min=0.000115, max=0.008239
**h=22:** 30,971 predictions, mean=0.000367, std=0.000224, min=0.000112, max=0.003156

---

## 5. Date Alignment (OHLCV vs Predictions, 2020+)

| Ticker | OHLCV days | Pred days | Inner | OHLCV only | Pred only |
|--------|-----------|-----------|-------|------------|-----------|
| AFLT | 1594 | 1593 | 1593 | 1 | 0 |
| ALRS | 1594 | 1593 | 1593 | 1 | 0 |
| HYDR | 1594 | 1593 | 1593 | 1 | 0 |
| IRAO | 1592 | 1591 | 1591 | 1 | 0 |
| LKOH | 1594 | 1593 | 1593 | 1 | 0 |
| LSRG | 1584 | 1583 | 1583 | 1 | 0 |
| MGNT | 1594 | 1593 | 1593 | 1 | 0 |
| MOEX | 1594 | 1593 | 1593 | 1 | 0 |
| MTLR | 1592 | 1591 | 1591 | 1 | 0 |
| MTSS | 1593 | 1592 | 1592 | 1 | 0 |
| NVTK | 1590 | 1589 | 1589 | 1 | 0 |
| OGKB | 1588 | 1587 | 1587 | 1 | 0 |
| PHOR | 1592 | 1591 | 1591 | 1 | 0 |
| RTKM | 1594 | 1593 | 1593 | 1 | 0 |
| SBER | 1594 | 1593 | 1593 | 1 | 0 |
| TATN | 1592 | 1591 | 1591 | 1 | 0 |
| VTBR | 1590 | 1589 | 1589 | 1 | 0 |

---

## 6. Gaps > 5 Business Days (2020+)

| Ticker | From | To | Calendar Days | Reason |
|--------|------|----|---------------|--------|
| AFLT | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| ALRS | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| HYDR | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| IRAO | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| LKOH | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| LSRG | 2022-02-25 | 2022-03-28 | 31 | MOEX trading halt |
| MGNT | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| MOEX | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| MTLR | 2022-02-25 | 2022-03-28 | 31 | MOEX trading halt |
| MTSS | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| NVTK | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| OGKB | 2022-02-25 | 2022-03-28 | 31 | MOEX trading halt |
| PHOR | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| RTKM | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| SBER | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| TATN | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |
| VTBR | 2022-02-25 | 2022-03-24 | 27 | MOEX trading halt |

---

## 7. Quality Checks

- Negative/zero prices in daily: PASS
- NaN in daily OHLCV: PASS
- NaN in hourly OHLCV: PASS
- Zero volume days (daily): 0
- Zero volume bars (hourly): 0
- NaN in rv_pred_h1: 4
- NaN in rv_pred_h5: 72
- NaN in rv_pred_h22: 361
