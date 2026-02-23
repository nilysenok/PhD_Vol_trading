# MOEX Volatility Dataset

## Период
**2014-08-26 — 2026-02-03** (2,874 торговых дней)

---

## Структура

```
dataset_final/
│
├── 01_stocks/                      ← Акции РФ (17 тикеров)
│   ├── rv_daily.parquet            ← RV из 10-мин свечей (long)
│   ├── rv_wide.parquet             ← RV (wide format)
│   └── candles_10m/                ← 10-мин свечи (17 файлов)
│
├── 02_external/                    ← Внешние факторы
│   ├── rv_moex.parquet             ← RV индексов MOEX (long)
│   ├── rv_moex_wide.parquet        ← RV индексов (wide)
│   ├── yahoo.parquet               ← Yahoo Finance (10 инструментов)
│   ├── macro.parquet               ← Макро-факторы (8 индикаторов)
│   └── candles_10m/                ← 10-мин свечи индексов (11 файлов)
│
├── 03_master/                      ← Объединённые данные
│   ├── master_long.parquet         ← Всё в одном (long format)
│   ├── master_wide.parquet         ← Всё в одном (wide format)
│   └── dates.csv                   ← Список 2,874 торговых дат
│
└── README.md
```

---

## 01_stocks/ — Акции РФ

### Тикеры (17)
SBER, LKOH, TATN, NVTK, VTBR, ALRS, AFLT, HYDR, MGNT, MOEX, RTKM, MTSS, MTLR, IRAO, OGKB, PHOR, LSRG

### Файлы

| Файл | Формат | Shape | Описание |
|------|--------|-------|----------|
| rv_daily.parquet | long | (48808, 6) | date, ticker, rv_daily, rv_annualized, n_bars, close |
| rv_wide.parquet | wide | (2874, 35) | date, rv_SBER, ..., close_SBER, ... |
| candles_10m/*.parquet | — | ~200K баров | 10-мин OHLCV свечи |

### Формат RV
- `rv_daily` — дневная сумма квадратов log-returns
- `rv_annualized` — в десятичном формате (0.50 = 50% годовых)

---

## 02_external/ — Внешние факторы

### rv_moex.parquet — Индексы MOEX (11)

| Тикер | Описание |
|-------|----------|
| IMOEX | Индекс Московской биржи |
| MOEXOG | Нефть и газ |
| MOEXFN | Финансы |
| MOEXMM | Металлы и добыча |
| MOEXEU | Электроэнергетика |
| MOEXTL | Телекоммуникации |
| MOEXCN | Потребительский сектор |
| MOEXCH | Химия и нефтехимия |
| RGBI | Индекс гос. облигаций |
| RVI | Индекс волатильности |
| CNYRUB_TOM | Юань/рубль |

### yahoo.parquet — Глобальные рынки (10)

| Колонка | Тикер | Описание |
|---------|-------|----------|
| brent | BZ=F | Brent Crude Oil |
| vix | ^VIX | CBOE Volatility Index |
| sp500 | ^GSPC | S&P 500 |
| gold | GC=F | Gold Futures |
| dxy | DX-Y.NYB | US Dollar Index |
| eem | EEM | Emerging Markets ETF |
| wti | CL=F | WTI Crude Oil |
| tbill_3m | ^IRX | 3-Month T-Bill |
| treasury_10y | ^TNX | 10-Year Treasury |
| treasury_5y | ^FVX | 5-Year Treasury |

### macro.parquet — Макро-факторы (8)

| Колонка | Описание | Источник |
|---------|----------|----------|
| cpi_ru | CPI Russia (%) | TradingView |
| cpi_us | CPI USA (index) | TradingView |
| unemp_us | US Unemployment (%) | TradingView |
| m2_ru | M2 Money Supply Russia | TradingView |
| pmi_ru | PMI Russia Manufacturing | Cbonds.ru |
| key_rate_cbr | Ключевая ставка ЦБ (%) | Cbonds.ru |
| fed_rate | Fed Funds Rate (%) | FRED |
| si_fut | USD/RUB (фьючерс Si) | TradingView |

---

## 03_master/ — Объединённые данные

### master_long.parquet
- **Формат:** Long (одна строка = один тикер в один день)
- **Shape:** (48808, 46)
- **Использование:** Для панельных моделей

### master_wide.parquet
- **Формат:** Wide (одна строка = один день)
- **Shape:** (2874, 75)
- **Использование:** Для временных рядов

---

## Использование

```python
import pandas as pd

# === Master dataset ===
df = pd.read_parquet('dataset_final/03_master/master_long.parquet')
sber = df[df['ticker'] == 'SBER']

# === По отдельности ===
stocks = pd.read_parquet('dataset_final/01_stocks/rv_daily.parquet')
yahoo = pd.read_parquet('dataset_final/02_external/yahoo.parquet')
macro = pd.read_parquet('dataset_final/02_external/macro.parquet')

# === Merge ===
merged = stocks.merge(yahoo, on='date', how='left')
merged = merged.merge(macro, on='date', how='left')

# === Wide format ===
stocks_wide = pd.read_parquet('dataset_final/01_stocks/rv_wide.parquet')
rv_moex = pd.read_parquet('dataset_final/02_external/rv_moex_wide.parquet')
```

---

## Примечания

1. **Формат RV:** Все значения в десятичном формате (0.50 = 50%)
2. **Пропуск 2022-02-24 — 2022-03-23:** Торги MOEX остановлены
3. **NaN:** <1% в макро-данных (начало периода)
4. **WTI -$37:** 20.04.2020 — реальное событие

---

## Источники данных

| Источник | Данные |
|----------|--------|
| MOEX ISS API | Акции 10-мин, индексы 10-мин, валюта |
| Yahoo Finance | Глобальные рынки (дневные) |
| TradingView | CPI, M2, PMI, Si фьючерс |
| Cbonds.ru | Ключевая ставка ЦБ |
| FRED | Fed Funds Rate |
