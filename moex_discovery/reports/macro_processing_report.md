# Macro Data Processing Report

**Дата:** 2026-02-06 09:10

## Источники данных

| Источник | Файлы |
|----------|-------|
| TradingView | CPI Russia, CPI USA, US Unemployment, M2 Russia |
| Cbonds.ru | PMI Russia, Ключевая ставка ЦБ |
| FRED | Fed Funds Rate |
| TradingView | USD/RUB фьючерс Si (4H свечи) |

## Результат обработки

| Тикер | Описание | Частота | Raw строк | Период raw | Daily строк | Filled % | Покрытие |
|-------|----------|---------|-----------|------------|-------------|----------|----------|
| CPI_RU | CPI Russia (%) | Месячный | 408 | 1992-01-01 — 2025-12-01 | 2870 | 95.3% | 99.9% |
| CPI_US | CPI USA (index) | Месячный | 911 | 1950-01-01 — 2025-12-01 | 2870 | 95.3% | 99.9% |
| UNEMP_US | US Unemployment (%) | Месячный | 935 | 1948-01-01 — 2025-12-01 | 2870 | 95.3% | 99.9% |
| M2_RU | M2 Money Supply Russia | Недельный | 397 | 1992-12-28 — 2025-12-29 | 2850 | 95.2% | 99.2% |
| PMI_RU | PMI Russia Manufacturing | Месячный | 173 | 2011-09-30 — 2026-01-31 | 2849 | 95.2% | 99.1% |
| KEY_RATE_CBR | CBR Key Rate (%) | Событийный | 3101 | 2013-09-13 — 2026-02-06 | 2874 | 0.3% | 100.0% |
| FED_RATE | Fed Funds Rate (%) | Дневной | 5881 | 2010-01-01 — 2026-02-06 | 2874 | -45.4% | 100.0% |
| SI_FUT | Si Futures (RUB/USD) | Дневной | 3158 | 2013-07-08 — 2026-02-05 | 2874 | -0.6% | 100.0% |

## Корреляционная матрица

Дневные доходности (pct_change):

| | CPI_RU | CPI_US | UNEMP_US | M2_RU | PMI_RU | KEY_RATE_CBR | FED_RATE | SI_FUT |
|---|---|---|---|---|---|---|---|---|
| CPI_RU | 1.00 | 0.49 | 0.13 | -0.01 | -0.00 | -0.00 | 0.18 | 0.00 |
| CPI_US | 0.49 | 1.00 | -0.09 | -0.01 | -0.00 | -0.00 | 0.06 | -0.01 |
| UNEMP_US | 0.13 | -0.09 | 1.00 | -0.00 | -0.00 | -0.00 | -0.06 | 0.00 |
| M2_RU | -0.01 | -0.01 | -0.00 | 1.00 | -0.01 | -0.00 | -0.02 | -0.01 |
| PMI_RU | -0.00 | -0.00 | -0.00 | -0.01 | 1.00 | -0.00 | -0.03 | 0.01 |
| KEY_RATE_CBR | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | 1.00 | 0.39 | 0.24 |
| FED_RATE | 0.18 | 0.06 | -0.06 | -0.02 | -0.03 | 0.39 | 1.00 | 0.19 |
| SI_FUT | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | 0.24 | 0.19 | 1.00 |

## Финальные файлы

### Raw (исходные данные)

```
moex_discovery/data/external/macro/raw/
├── cpi_russia.parquet
├── cpi_usa.parquet
├── unemployment_usa.parquet
├── m2_russia.parquet
├── pmi_russia.parquet
├── key_rate_cbr.parquet
├── fed_rate.parquet
└── si_fut_daily.parquet
```

### Processed (дневные, выровненные)

```
moex_discovery/data/external/macro/processed/
├── macro_daily.parquet   ← Long format (22931 records)
└── macro_wide.parquet    ← Wide format (2874 × 9)

moex_discovery/data/final_v5/
└── macro_wide.parquet    ← Копия для интеграции
```

### Структура macro_wide.parquet

- **Столбцы:** date, CPI_RU, CPI_US, UNEMP_US, M2_RU, PMI_RU, KEY_RATE_CBR, FED_RATE, SI_FUT
- **Строк:** 2874
- **Период:** 2014-08-26 — 2026-02-03

## Использование

```python
import pandas as pd

# Загрузка
macro = pd.read_parquet('moex_discovery/data/final_v5/macro_wide.parquet')

# Merge с акциями
stocks = pd.read_parquet('moex_discovery/data/final_v5/rv_stocks.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
macro['date'] = pd.to_datetime(macro['date'])
merged = stocks.merge(macro, on='date', how='left')
```

## Примечания

1. **Forward-fill:** Месячные данные (CPI, PMI, M2) заполнены вперёд до следующего известного значения
2. **SI_FUT:** Фьючерс Si — это RUB/USD (обратный к USDRUB), используется как альтернатива Yahoo USDRUB
3. **RV для SI_FUT:** Рассчитана из 4H баров, сохранена в si_fut_daily.parquet
4. **Покрытие ~99%:** Начало периода (август 2014) может не иметь данных для некоторых индикаторов
