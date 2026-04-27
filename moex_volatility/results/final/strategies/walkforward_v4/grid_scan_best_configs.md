# Grid Scan: лучшие конфигурации per Strategy × Ticker

**Метрика:** `calc_trading_sharpe` (compound return - compound RF, без FDR, vol ALL days, comm=0.05%)
**Период:** OOS 2022-01-03 — 2026-02-03
**Источник:** Phase 1 (signal grid) + Phase 2 (RM refinement) + Pyramiding test

---

## S1 MeanReversion (mean=+1.15, 9/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +1.29 | ma=hull w=5 th=0.7 | none | {} | be=1.0 mh=10 sl_p=0.03 tp_a=2.0 | 0 |
| ALRS | +0.96 | ma=hull w=5 th=0.3 | volume | {} | be=2.0 chand=5.0 mh=30 rr=0.5 | 0 |
| HYDR | +1.12 | ma=hull w=80 th=0.3 | volume | {} | be=2.0 mh=10 sl_a=4.0 trail=5.0 | 0 |
| IRAO | +0.76 | ma=hull w=10 th=0.3 | none | {} | be=1.5 mh=30 rr=3.0 sl_p=0.07 | 0 |
| LKOH | +0.85 | ma=hull w=10 th=0.3 | none | {} | be=2.0 mh=20 rr=2.5 sl_a=3.0 | 0 |
| LSRG | +1.53 | ma=ema w=150 th=0.5 | volume | adx_th=25 atr_lo=25 | default | 0 |
| MGNT | +1.17 | ma=hull w=50 th=1.3 | none | {} | be=2.0 rr=3.0 sl_p=0.07 | 0 |
| MOEX | +0.86 | ma=hull w=10 th=0.3 | none | {} | mh=30 sl_p=0.07 trail=5.0 | 0 |
| MTLR | +1.84 | ma=hull w=5 th=0.5 | none | {} | be=1.0 rr=0.5 sl_a=5.0 | 0 |
| MTSS | +1.08 | ma=sma w=200 th=4.0 | volume | {} | rr=3.0 sl_p=0.07 | 0 |
| NVTK | +0.82 | ma=hull w=5 th=0.3 | volume | {} | sl_p=0.1 tp_a=2.0 | 0 |
| OGKB | +1.79 | ma=hull w=5 th=0.5 | none | {} | chand=5.0 mh=20 rr=3.0 | 0 |
| PHOR | +1.29 | ma=hull w=80 th=0.3 | volume | {} | mh=10 rr=1.5 sl_a=2.5 | 0 |
| RTKM | +1.40 | ma=hull w=60 th=1.3 | none | {} | be=0.5 sl_a=2.0 tp_p=0.15 | 0 |
| SBER | +0.97 | ma=hull w=60 th=1.3 | none | {} | be=1.5 sl_a=0.5 tp_a=8.0 | 0 |
| TATN | +0.94 | ma=hull w=14 th=0.5 | none | mom_n=3 | be=1.5 mh=20 sl_a=1.0 trail=2.5 | 0 |
| VTBR | +0.89 | ma=hull w=5 th=0.3 | volume | {} | chand=4.0 mh=30 rr=4.0 | 0 |

**Паттерны S1:** Hull MA доминирует (14/17). Короткие окна (w=5-10) для волатильных, длинные (w=60-200) для стабильных тикеров. Breakeven почти всегда. RR ratio — основной exit.

---

## S2 Bollinger (mean=+1.20, 13/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +1.03 | ma=sma w=5 bs=1.5 | none | {} | be=1.5 mh=30 rr=2.0 sl_a=5.0 | 0 |
| ALRS | +1.03 | ma=hull w=10 bs=0.5 | none | {} | be=2.0 mh=15 rr=2.5 sl_p=0.1 | 0 |
| HYDR | +1.23 | ma=sma w=10 bs=1.5 | none | {} | be=1.5 chand=4.0 mh=30 rr=0.5 | 0 |
| IRAO | +0.85 | ma=sma w=5 bs=1.5 | none | {} | be=2.0 mh=20 rr=3.0 sl_a=5.0 | 0 |
| LKOH | +1.15 | ma=sma w=7 bs=1.3 | none | {} | rr=1.5 sl_a=1.5 | 0 |
| LSRG | +1.12 | ma=sma w=5 bs=1.3 | volume | adx_th=40 | mh=15 sl_a=2.5 tp_p=0.15 | 0 |
| MGNT | +0.96 | ma=dema w=10 bs=0.7 | none | {} | sl_a=1.5 trail=4.0 | 0 |
| MOEX | +0.99 | ma=dema w=10 bs=0.7 | none | {} | chand=5.0 mh=30 rr=2.0 | 0 |
| MTLR | +1.50 | ma=hull w=5 bs=0.5 | none | {} | be=1.5 mh=10 sl_a=0.5 tp_p=0.1 | 0 |
| MTSS | +0.85 | ma=hull w=10 bs=0.5 | none | {} | sl_a=2.5 tp_p=0.15 | 0 |
| NVTK | +1.37 | ma=ema w=7 bs=1.3 | none | {} | chand=5.0 rr=1.5 | 0 |
| OGKB | +1.61 | ma=hull w=5 bs=0.5 | volume | {} | be=1.0 chand=5.0 tp_a=1.5 | 0 |
| PHOR | +1.13 | ma=sma w=10 bs=1.5 | none | {} | be=1.0 mh=30 rr=2.0 sl_a=5.0 | 0 |
| RTKM | +1.56 | ma=dema w=10 bs=1.5 | none | atr_lo=10 | mh=15 rr=2.5 sl_a=5.0 | 0 |
| SBER | +1.08 | ma=sma w=5 bs=1.5 | none | {} | be=1.0 chand=4.0 rr=2.0 | 0 |
| TATN | +1.33 | ma=dema w=10 bs=0.7 | none | {} | be=2.0 rr=1.5 sl_a=2.0 | 0 |
| VTBR | +1.51 | ma=sma w=10 bs=1.5 | none | {} | be=1.0 mh=20 sl_p=0.05 tp_a=8.0 | 0 |

**Паттерны S2:** SMA/DEMA/Hull — все работают. Окна 5-10. bs=0.5-1.5. RR ratio доминирует. Chandelier 4-5 для SL. Breakeven почти всегда.

---

## S3 Donchian (mean=+1.31 с пирамидингом, 12/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +2.17 | w=3 | rsi30 | {} | be=1.5 mh=20 sl_a=1.5 trail=5.0 | 3×1.0 |
| ALRS | +1.67 | w=7 | rsi30 | {} | be=1.5 chand=3.0 mh=20 trail=3.0 | 3×1.0 |
| HYDR | +1.37 | w=5 | rsi20 | {} | be=1.5 mh=20 sl_a=1.5 trail=5.0 | 3×1.0 |
| IRAO | +1.44 | w=3 | volume | {} | be=1.5 mh=50 sl_a=5.0 trail=5.0 | 3×1.0 |
| LKOH | +0.24 | w=10 | rsi20 | {} | be=2.0 mh=20 sl_a=1.5 trail=3.0 | 0 |
| LSRG | +0.73 | w=14 | volume | {} | be=1.5 mh=50 sl_a=5.0 trail=5.0 | 2×1.0 |
| MGNT | +0.31 | w=3 | volume | {} | be=2.0 chand=3.0 mh=20 trail=5.0 | 0 |
| MOEX | +2.33 | w=3 | volume | {} | be=2.0 mh=20 sl_a=1.5 trail=3.0 | 3×1.0 |
| MTLR | +1.74 | w=10 | none | {} | be=2.0 mh=20 sl_a=1.5 trail=3.0 | 3×1.0 |
| MTSS | +1.36 | w=7 | rsi20 | {} | be=1.5 mh=20 sl_a=1.5 trail=5.0 | 3×1.0 |
| NVTK | +0.31 | w=10 | volume | {} | be=1.5 mh=50 sl_a=5.0 trail=5.0 | 3×1.0 |
| OGKB | +1.48 | w=7 | none | {} | be=1.0 sl_a=2.0 trail=5.0 | 3×1.0 |
| PHOR | -0.14 | w=7 | rsi20 | {} | be=1.0 sl_a=2.5 trail=5.0 | 0 |
| RTKM | +2.59 | w=5 | rsi20 | {} | be=2.0 mh=20 sl_a=1.5 trail=3.0 | 3×1.0 |
| SBER | +1.38 | w=3 | none | {} | be=1.0 sl_a=2.5 trail=5.0 | 3×1.0 |
| TATN | +0.40 | w=10 | none | {} | be=1.5 chand=3.0 mh=20 trail=3.0 | 3×1.0 |
| VTBR | +2.15 | w=7 | rsi20 | {} | be=1.5 mh=50 sl_a=5.0 trail=5.0 | 3×1.0 |

**Паттерны S3:** Короткие окна w=3-7. Trailing ОБЯЗАТЕЛЕН (3.0-5.0). Пирамидинг 3×1.0 критичен для Sh>1.0. Breakeven 1.0-2.0. RSI20/RSI30/volume как подтверждение. Без пирамидинга max=0.84.

---

## S4 Supertrend (mean=+0.95, 7/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +0.91 | ap=5 mult=1.25 | rsi30 | {} | be=1.5 mh=100 sl_a=4.0 trail=5.0 | 0 |
| ALRS | +0.85 | ap=5 mult=1.5 | volume | {} | be=2.0 chand=3.0 mh=100 trail=2.0 | 0 |
| HYDR | +0.76 | ap=3 mult=1.25 | rsi30 | {} | be=1.5 chand=4.0 mh=50 trail=4.0 | 0 |
| IRAO | +0.08 | ap=3 mult=1.75 | volume | {} | be=1.5 chand=4.0 mh=50 trail=2.5 | 0 |
| LKOH | +1.12 | ap=5 mult=1.0 | volume | {} | mh=50 sl_a=4.0 trail=4.0 | 0 |
| LSRG | +0.84 | ap=12 mult=1.75 | rsi30 | {} | be=0.5 sl_a=1.5 trail=1.5 | 0 |
| MGNT | +0.98 | ap=10 mult=1.25 | volume | mom_n=10 | mh=30 sl_a=2.0 tp_p=0.1 | 0 |
| MOEX | +1.35 | ap=10 mult=1.5 | rsi30 | vol_th=1.0 | be=1.5 chand=3.0 mh=50 tp_a=4.0 | 0 |
| MTLR | +2.18 | ap=7 mult=1.25 | none | rsi_th=30 | mh=10 rr=1.5 sl_p=0.1 | 0 |
| MTSS | +1.41 | ap=3 mult=1.5 | volume | {} | be=2.0 chand=3.0 mh=50 trail=4.0 | 0 |
| NVTK | +1.00 | ap=3 mult=0.75 | rsi30 | mom_n=5 | be=1.5 mh=10 rr=2.0 sl_a=2.5 | 0 |
| OGKB | +0.63 | ap=5 mult=1.25 | rsi30 | {} | be=2.0 chand=2.0 mh=100 trail=4.0 | 0 |
| PHOR | +0.00 | ap=20 mult=1.0 | volume | adx_th=20 | default | 0 |
| RTKM | +1.14 | ap=35 mult=0.5 | rsi20 | mom_n=7 | be=1.0 chand=2.5 mh=20 tp_a=5.0 | 0 |
| SBER | +0.70 | ap=14 mult=1.5 | rsi30 | {} | sl_a=2.5 trail=5.0 | 0 |
| TATN | +1.09 | ap=5 mult=1.0 | volume | dow=3 mom_n=3 | be=2.0 mh=30 rr=3.0 sl_a=3.0 | 0 |
| VTBR | +1.05 | ap=5 mult=1.0 | volume | {} | chand=5.0 mh=50 trail=5.0 | 0 |

**Паттерны S4:** ap=3-10, mult=0.75-1.5. Chandelier 2-4 + trailing 2-5. Длинное удержание mh=50-100. RSI30/volume подтверждение. MTLR — исключение: короткий mh=10 + RR ratio.

---

## S5 PivotPoints (mean=+1.14, 10/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +0.88 | pt=camarilla el=1 | none | {} | be=1.0 mh=15 rr=4.0 sl_a=5.0 | 0 |
| ALRS | +0.69 | pt=classic el=1 | none | {} | be=2.0 mh=20 sl_a=2.0 tp_a=1.5 | 0 |
| HYDR | +0.98 | pt=classic el=1 | none | {} | be=0.5 mh=15 rr=1.0 sl_a=2.0 | 0 |
| IRAO | +0.91 | pt=woodie el=1 | volume | {} | be=1.5 rr=2.5 sl_p=0.1 | 0 |
| LKOH | +1.13 | pt=fibonacci el=2 | none | adx_th=35 atr_lo=15 | be=2.0 chand=4.0 mh=30 trail=3.0 | 0 |
| LSRG | +1.76 | pt=fibonacci el=1 | none | {} | be=2.0 chand=5.0 mh=5 rr=1.0 | 0 |
| MGNT | +0.87 | pt=fibonacci el=1 | none | {} | chand=5.0 mh=15 rr=1.0 | 0 |
| MOEX | +1.14 | pt=camarilla el=1 | none | {} | mh=30 sl_p=0.07 trail=5.0 | 0 |
| MTLR | +2.46 | pt=fibonacci el=1 | none | {} | be=1.0 mh=30 sl_a=3.0 trail=2.0 | 0 |
| MTSS | +1.13 | pt=camarilla el=1 | none | {} | be=2.0 mh=15 sl_p=0.03 trail=4.0 | 0 |
| NVTK | +0.80 | pt=classic el=1 | none | {} | be=1.0 chand=2.5 rr=2.5 | 0 |
| OGKB | +1.15 | pt=woodie el=1 | volume | {} | be=2.0 sl_p=0.03 tp_a=8.0 | 0 |
| PHOR | +1.06 | pt=camarilla el=1 | none | {} | mh=20 rr=4.0 sl_a=1.5 | 0 |
| RTKM | +1.26 | pt=fibonacci el=2 | none | {} | be=2.0 chand=5.0 mh=10 rr=3.0 | 0 |
| SBER | +0.66 | pt=fibonacci el=1 | none | {} | sl_a=1.0 tp_a=8.0 | 0 |
| TATN | +1.03 | pt=camarilla el=1 | none | {} | be=1.0 rr=1.5 sl_a=4.0 | 0 |
| VTBR | +1.55 | pt=classic el=1 | none | {} | be=1.5 chand=4.0 mh=10 rr=3.0 | 0 |

**Паттерны S5:** el=1 доминирует (15/17). Fibonacci (7) и camarilla (5) — лучшие типы. RR ratio + chandelier. Без фильтров (15/17 conf=none). Breakeven обязателен.

---

## S6 VWAP (mean=+1.28, 13/17 > 1.0)

| Ticker | Sharpe | Signal | Conf | Filter | RM | Pyr |
|--------|--------|--------|------|--------|----|-----|
| AFLT | +1.98 | vw=7 dm=1.0 | none | {} | be=1.5 chand=5.0 mh=20 rr=1.5 | 0 |
| ALRS | +0.80 | vw=3 dm=2.0 | none | {} | chand=5.0 rr=1.0 | 0 |
| HYDR | +1.16 | vw=5 dm=1.75 | none | {} | be=1.0 mh=30 sl_a=1.5 tp_a=3.0 | 0 |
| IRAO | +1.19 | vw=5 dm=2.0 | none | {} | mh=30 sl_p=0.07 trail=5.0 | 0 |
| LKOH | +0.71 | vw=3 dm=0.5 | none | {} | be=1.5 mh=15 sl_a=1.5 tp_p=0.1 | 0 |
| LSRG | +1.23 | vw=3 dm=1.0 | none | {} | mh=20 sl_a=5.0 tp_a=8.0 | 0 |
| MGNT | +1.36 | vw=3 dm=1.0 | none | {} | be=2.0 mh=30 rr=3.0 sl_a=1.0 | 0 |
| MOEX | +0.91 | vw=3 dm=2.0 | none | {} | be=1.5 mh=30 rr=3.0 sl_p=0.1 | 0 |
| MTLR | +1.90 | vw=3 dm=0.75 | none | {} | be=1.5 rr=2.5 sl_a=5.0 | 0 |
| MTSS | +1.16 | vw=5 dm=1.75 | none | {} | be=2.0 mh=30 rr=4.0 sl_a=2.5 | 0 |
| NVTK | +1.01 | vw=3 dm=0.75 | none | atr_lo=20 | default | 0 |
| OGKB | +1.56 | vw=5 dm=1.75 | none | {} | be=2.0 chand=5.0 mh=20 rr=2.5 | 0 |
| PHOR | +0.94 | vw=7 dm=1.0 | none | {} | be=1.5 chand=3.0 mh=30 rr=2.0 | 0 |
| RTKM | +1.68 | vw=5 dm=0.25 | none | {} | mh=10 sl_a=0.5 trail=5.0 | 0 |
| SBER | +1.07 | vw=3 dm=2.0 | none | {} | be=1.5 mh=20 rr=2.5 sl_a=3.0 | 0 |
| TATN | +1.59 | vw=3 dm=0.75 | none | atr_lo=20 | be=1.5 mh=20 sl_a=1.0 trail=2.5 | 0 |
| VTBR | +1.49 | vw=3 dm=2.0 | none | {} | be=2.0 rr=1.5 sl_a=4.0 | 0 |

**Паттерны S6:** vw=3-5 (короткий VWAP). dm=0.25-2.0 (широкий диапазон). RR ratio + chandelier. Без подтверждений (17/17 conf=none). Breakeven 1.0-2.0.

---

## Сводка

| Strategy | Mean Sh | Median | Max | N>0 | N>1.0 | Best ticker |
|----------|---------|--------|-----|-----|-------|-------------|
| S1 MeanRev | +1.15 | +1.08 | +1.84 | 17/17 | 9/17 | MTLR |
| S2 Bollinger | +1.20 | +1.13 | +1.61 | 17/17 | 13/17 | OGKB |
| S3 Donchian | +1.31 | +1.37 | +2.59 | 16/17 | 12/17 | RTKM |
| S4 Supertrend | +0.95 | +0.98 | +2.18 | 17/17 | 7/17 | MTLR |
| S5 PivotPoints | +1.14 | +1.06 | +2.46 | 17/17 | 10/17 | MTLR |
| S6 VWAP | +1.28 | +1.19 | +1.98 | 17/17 | 13/17 | AFLT |

### Общие RM паттерны

- **Breakeven (be):** используется в 80%+ конфигов, типично 1.0-2.0 ATR
- **RR ratio:** доминирует для контр-трендовых (S1/S2/S5/S6), типично 1.0-3.0
- **Trailing:** доминирует для трендовых (S3/S4), типично 3.0-5.0 ATR
- **Chandelier:** популярен для SL, типично 3.0-5.0 ATR
- **Max hold:** 10-30 для контр-трендовых, 50-100 для трендовых
- **Пирамидинг:** критичен для S3 (3×1.0, cond=profit_1atr)
