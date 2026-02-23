# FINAL DATASET V6 — Итоговый отчёт

**Дата создания:** 2026-02-06 09:23

## Резюме

| Параметр | Значение |
|----------|----------|
| Период | 2014-08-26 — 2026-02-03 |
| Торговых дней | 2,874 |
| Тикеров акций | 17 |
| Внешних факторов | 39 |
| Общий размер | 128.9 MB |

## Главные файлы

### master_dataset.parquet

| Параметр | Значение |
|----------|----------|
| Формат | Long (ticker × date) |
| Строк | 48,808 |
| Колонок | 46 |
| Размер | 2.63 MB |

### master_wide.parquet

| Параметр | Значение |
|----------|----------|
| Формат | Wide (date) |
| Строк | 2,874 |
| Колонок | 75 |
| Размер | 1.38 MB |

## Структура файлов

```
final_v6/                                    [128.9 MB total]
│
├── master_dataset.parquet                   [2690 KB]
├── master_wide.parquet                      [1409 KB]
├── common_dates.csv                         [31 KB]
├── README.md
│
├── source/
│   ├── stocks/rv_stocks.parquet             [1008 KB]
│   ├── external_moex/
│   │   ├── rv_external_daily.parquet        [853 KB]
│   │   └── rv_external_wide.parquet         [510 KB]
│   ├── yahoo/yahoo_wide.parquet             [183 KB]
│   └── macro/macro_wide.parquet             [53 KB]
│
└── candles_10m/                             [122.3 MB]
    ├── stocks/      [17 files]
    └── external/    [11 files]
```

## Статистика данных

### RV акций (годовая)

| Тикер | Среднее RV | Медиана RV |
|-------|------------|------------|
| PHOR | 67.5% | 23.4% |
| MTLR | 67.1% | 35.0% |
| AFLT | 64.8% | 26.3% |
| IRAO | 58.5% | 24.4% |
| NVTK | 55.5% | 26.3% |
| OGKB | 51.8% | 29.5% |
| MGNT | 49.5% | 24.5% |
| TATN | 48.2% | 25.8% |
| LSRG | 47.9% | 25.9% |
| ALRS | 47.8% | 25.5% |
| MTSS | 47.0% | 21.0% |
| RTKM | 46.3% | 21.3% |
| VTBR | 46.3% | 23.6% |
| SBER | 46.1% | 22.1% |
| MOEX | 44.9% | 22.6% |
| LKOH | 41.9% | 21.7% |
| HYDR | 39.1% | 23.3% |

### Корреляции

**RV акций vs RV индексов:**

| Пара | Корреляция |
|------|------------|
| SBER ↔ LKOH | 0.87 |
| SBER ↔ IMOEX | 0.54 |
| LKOH ↔ TATN | 0.75 |
| IMOEX ↔ RVI | 0.16 |

### NaN (пропуски)

| Колонка | NaN % |
|---------|-------|
| pmi_ru | 0.86% |
| m2_ru | 0.83% |
| cpi_ru | 0.14% |
| cpi_us | 0.14% |
| unemp_us | 0.14% |

## Источники данных

| Категория | Количество | Источник |
|-----------|------------|----------|
| Акции MOEX | 17 тикеров | MOEX ISS API (10-мин) |
| Индексы MOEX | 11 индексов | MOEX ISS API (10-мин) |
| Глобальные рынки | 10 инструментов | Yahoo Finance (дневные) |
| Макро-факторы | 8 индикаторов | ЦБ, FRED, TradingView |

## Готовность к моделированию

✅ **Датасет готов к Feature Engineering и моделированию**

### Следующие шаги:

1. **Лаги и скользящие средние:**
   ```python
   df['rv_lag1'] = df.groupby('ticker')['rv_daily'].shift(1)
   df['rv_ma22'] = df.groupby('ticker')['rv_daily'].transform(lambda x: x.rolling(22).mean())
   ```

2. **HAR модель:**
   ```python
   df['rv_day'] = df.groupby('ticker')['rv_daily'].shift(1)
   df['rv_week'] = df.groupby('ticker')['rv_daily'].transform(lambda x: x.rolling(5).mean().shift(1))
   df['rv_month'] = df.groupby('ticker')['rv_daily'].transform(lambda x: x.rolling(22).mean().shift(1))
   ```

3. **Внешние факторы:**
   - VIX, SP500 — глобальные риски
   - Brent, Gold — сырьевые рынки
   - KEY_RATE_CBR, FED_RATE — монетарная политика

---

**Файл:** `moex_discovery/data/final_v6/`
