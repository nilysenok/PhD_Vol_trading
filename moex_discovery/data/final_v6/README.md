# MOEX Discovery — Финальный датасет v6

**Создан:** 2026-02-06 09:22

## Период

**2014-08-26 — 2026-02-03** (2,874 торговых дней)

## Тикеры акций (17)

AFLT, ALRS, HYDR, IRAO, LKOH, LSRG, MGNT, MOEX, MTLR, MTSS, NVTK, OGKB, PHOR, RTKM, SBER, TATN, VTBR

## Структура папки

```
final_v6/
├── master_dataset.parquet     ← Главный файл (long format)
├── master_wide.parquet        ← Альтернативный формат (wide)
├── common_dates.csv           ← Список торговых дат
├── README.md                  ← Этот файл
│
├── source/                    ← Исходные данные
│   ├── stocks/
│   │   └── rv_stocks.parquet          ← RV акций
│   ├── external_moex/
│   │   ├── rv_external_daily.parquet  ← RV индексов (long)
│   │   └── rv_external_wide.parquet   ← RV индексов (wide)
│   ├── yahoo/
│   │   └── yahoo_wide.parquet         ← Глобальные рынки
│   └── macro/
│       └── macro_wide.parquet         ← Макро-факторы
│
└── candles_10m/               ← 10-минутные свечи
    ├── stocks/                ← 17 файлов
    └── external/              ← 11 файлов
```

## master_dataset.parquet

**Главный файл для моделирования.**

- **Формат:** Long (одна строка = один тикер в один день)
- **Строк:** 48,808
- **Колонок:** 46

### Колонки

| # | Колонка | Описание | Тип |
|---|---------|----------|-----|
| 1 | date | Дата торгов | datetime |
| 2 | ticker | Тикер акции | str |
| 3 | rv_daily | Realized Volatility (дневная) | float |
| 4 | rv_annualized | RV в годовом выражении (0.25 = 25%) | float |
| 5 | n_bars | Количество 10-мин баров | int |
| 6 | close | Цена закрытия акции | float |
| 7-17 | rv_* | RV внешних индексов MOEX | float |
| 18-20 | close_IMOEX, close_RVI, close_CNYRUB_TOM | Ключевые close индексов | float |
| 21-30 | brent, vix, sp500, gold, ... | Yahoo Finance (10 инструментов) | float |
| 31-38 | cpi_ru, cpi_us, ... si_fut | Макро-факторы (8 индикаторов) | float |
| 39-46 | close_* | Close остальных индексов MOEX | float |

### Внешние индексы MOEX (11)

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
| CNYRUB_TOM | Юань/рубль (валюта) |

### Yahoo Finance (10)

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

### Макро-факторы (8)

| Колонка | Описание | Частота |
|---------|----------|---------|
| cpi_ru | CPI Russia (%) | Месячный |
| cpi_us | CPI USA (index) | Месячный |
| unemp_us | US Unemployment (%) | Месячный |
| m2_ru | M2 Money Supply Russia | Недельный |
| pmi_ru | PMI Russia Manufacturing | Месячный |
| key_rate_cbr | Ключевая ставка ЦБ (%) | Событийный |
| fed_rate | Fed Funds Rate (%) | Дневной |
| si_fut | USD/RUB фьючерс | Дневной |

## master_wide.parquet

**Альтернативный формат для панельного анализа.**

- **Формат:** Wide (одна строка = один день)
- **Строк:** 2,874
- **Колонок:** 75

Содержит:
- rv_SBER, rv_LKOH, ... (17 RV акций)
- close_SBER, close_LKOH, ... (17 close акций)
- rv_IMOEX, rv_RVI, ... (11 RV индексов)
- brent, vix, sp500, ... (10 Yahoo)
- cpi_ru, key_rate_cbr, ... (8 макро)

## Использование

```python
import pandas as pd

# === Загрузка ===
df = pd.read_parquet('final_v6/master_dataset.parquet')

# === Фильтр по тикеру ===
sber = df[df['ticker'] == 'SBER']

# === Все данные за период ===
df_2024 = df[df['date'].dt.year == 2024]

# === Wide формат ===
df_wide = pd.read_parquet('final_v6/master_wide.parquet')

# === Корреляция RV ===
rv_cols = [c for c in df_wide.columns if c.startswith('rv_')]
corr = df_wide[rv_cols].corr()

# === Feature Engineering ===
# Лаги
df['rv_lag1'] = df.groupby('ticker')['rv_daily'].shift(1)
df['rv_lag5'] = df.groupby('ticker')['rv_daily'].shift(5)

# Скользящие средние
df['rv_ma5'] = df.groupby('ticker')['rv_daily'].transform(lambda x: x.rolling(5).mean())
df['rv_ma22'] = df.groupby('ticker')['rv_daily'].transform(lambda x: x.rolling(22).mean())

# Log-returns
df['log_return'] = df.groupby('ticker')['close'].transform(lambda x: np.log(x / x.shift(1)))
```

## Статистика

### RV (годовая, в %)

| Метрика | Значение |
|---------|----------|
| Среднее | 51.2% |
| Медиана | 24.6% |
| Мин | 2.6% |
| Макс | 34982.3% |

### Ключевые корреляции RV

- SBER ↔ LKOH: 0.87
- SBER ↔ IMOEX: 0.54
- LKOH ↔ TATN: 0.75

## Источники данных

| Источник | Данные | Период |
|----------|--------|--------|
| MOEX ISS API | Акции 10-мин, индексы 10-мин, валюта | 2014-2026 |
| Yahoo Finance | Глобальные рынки (дневные) | 2014-2026 |
| TradingView | CPI, M2, PMI, Si фьючерс | 1992-2026 |
| Cbonds.ru | Ключевая ставка ЦБ | 2013-2026 |
| FRED | Fed Funds Rate | 2010-2026 |

## Примечания

1. **RV (Realized Volatility):** Рассчитана как сумма квадратов логарифмических доходностей внутри дня
2. **Forward-fill:** Месячные данные (CPI, PMI) заполнены вперёд до следующего известного значения
3. **NaN:** Небольшое количество пропусков в начале периода для некоторых макро-индикаторов
4. **SI_FUT:** Курс USD/RUB из фьючерса Si (замена некачественных данных Yahoo)
