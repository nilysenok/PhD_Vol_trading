# Yahoo Finance Download Report

**Дата:** 2026-02-05 20:10

## Параметры

- Период: 2014-08-26 — 2026-02-03
- Инструменты: 12
- Успешно загружено: 12

## Результаты загрузки

| Ticker | Инструмент | Строк | Период | Overlap с MOEX |
|--------|------------|-------|--------|----------------|
| BZ=F | Brent Crude Oil Futures | 2877 | 2014-08-26 — 2026-02-02 | 96.3% (2767) |
| ^VIX | CBOE Volatility Index | 2876 | 2014-08-26 — 2026-02-02 | 96.2% (2766) |
| ^GSPC | S&P 500 Index | 2876 | 2014-08-26 — 2026-02-02 | 96.2% (2766) |
| GC=F | Gold Futures | 2875 | 2014-08-26 — 2026-02-02 | 96.2% (2765) |
| DX-Y.NYB | US Dollar Index | 2877 | 2014-08-26 — 2026-02-02 | 96.3% (2767) |
| EEM | iShares MSCI Emerging Markets  | 2876 | 2014-08-26 — 2026-02-02 | 96.2% (2766) |
| USDRUB=X | USD/RUB Exchange Rate | 2977 | 2014-08-26 — 2026-02-02 | 99.4% (2858) |
| EURRUB=X | EUR/RUB Exchange Rate | 2978 | 2014-08-26 — 2026-02-02 | 99.5% (2859) |
| CL=F | WTI Crude Oil Futures | 2876 | 2014-08-26 — 2026-02-02 | 96.2% (2766) |
| ^IRX | 13-Week Treasury Bill Rate | 2875 | 2014-08-26 — 2026-02-02 | 96.2% (2765) |
| ^TNX | 10-Year Treasury Note Yield | 2875 | 2014-08-26 — 2026-02-02 | 96.2% (2765) |
| ^FVX | 5-Year Treasury Note Yield | 2875 | 2014-08-26 — 2026-02-02 | 96.2% (2765) |

## Итоговые файлы

| Файл | Описание |
|------|----------|
| `yahoo_wide.parquet` | Wide format, 2862 rows × 12 tickers |
| `yahoo_long.parquet` | Long format, 34344 records |

## Структура данных

### yahoo_wide.parquet
- Столбцы: date, BZ_F, VIX, GSPC, GC_F, DX_Y_NYB, EEM...
- Строки: 2862 (aligned to common_dates)

### yahoo_long.parquet
- Столбцы: date, ticker, close
- Строки: 34344

## Покрытие

- **BZ_F**: 2862/2862 (100.0%)
- **VIX**: 2862/2862 (100.0%)
- **GSPC**: 2862/2862 (100.0%)
- **GC_F**: 2862/2862 (100.0%)
- **DX_Y_NYB**: 2862/2862 (100.0%)
- **EEM**: 2862/2862 (100.0%)
- **USDRUB_X**: 2862/2862 (100.0%)
- **EURRUB_X**: 2862/2862 (100.0%)
- **CL_F**: 2862/2862 (100.0%)
- **IRX**: 2862/2862 (100.0%)
- **TNX**: 2862/2862 (100.0%)
- **FVX**: 2862/2862 (100.0%)

## Использование

```python
import pandas as pd

# Wide format (удобно для merge)
yahoo = pd.read_parquet('moex_discovery/data/external/yahoo/final/yahoo_wide.parquet')

# Long format
yahoo_long = pd.read_parquet('moex_discovery/data/external/yahoo/final/yahoo_long.parquet')

# Merge с final_v5
stocks = pd.read_parquet('moex_discovery/data/final_v5/rv_stocks.parquet')
merged = stocks.merge(yahoo, on='date', how='left')
```

## Примечания

- Данные Yahoo Finance — дневные (не 10-мин)
- Используются close prices (adjusted)
- Пропуски заполнены forward-fill
- Валютные курсы (USDRUB, EURRUB) могут иметь пропуски в выходные


---

## ЭТАП 2: Выравнивание до 100% overlap

**Дата:** 2026-02-05 20:11

### Результат

- Common dates: 2874
- Yahoo dates (после выравнивания): 2874
- Заполнено пропусков: 12 дат

### Пропущенные даты (заполнены forward-fill)

| Дата | День недели | Причина |
|------|-------------|---------|
| 2016-02-20 | Saturday | MOEX торгует, США нет |
| 2018-04-28 | Saturday | MOEX торгует, США нет |
| 2018-06-09 | Saturday | MOEX торгует, США нет |
| 2018-12-29 | Saturday | MOEX торгует, США нет |
| 2021-02-20 | Saturday | MOEX торгует, США нет |
| 2024-04-27 | Saturday | MOEX торгует, США нет |
| 2024-11-02 | Saturday | MOEX торгует, США нет |
| 2024-12-28 | Saturday | MOEX торгует, США нет |
| 2025-04-18 | Friday | Good Friday (США) |
| 2025-11-01 | Saturday | MOEX торгует, США нет |
| 2025-12-25 | Thursday | Christmas (США) |
| 2026-02-03 | Tuesday | Праздник США или нет данных |

### Покрытие после выравнивания

| Инструмент | Строк | Покрытие |
|------------|-------|----------|
| BZ_F | 2874 | 100.0% |
| VIX | 2874 | 100.0% |
| GSPC | 2874 | 100.0% |
| GC_F | 2874 | 100.0% |
| DX_Y_NYB | 2874 | 100.0% |
| EEM | 2874 | 100.0% |
| USDRUB_X | 2874 | 100.0% |
| EURRUB_X | 2874 | 100.0% |
| CL_F | 2874 | 100.0% |
| IRX | 2874 | 100.0% |
| TNX | 2874 | 100.0% |
| FVX | 2874 | 100.0% |

### Подтверждение

**✅ 100% overlap достигнут для ВСЕХ 12 Yahoo Finance инструментов**

- 12 инструментов × 2874 дней = 34488 записей (теоретически)
- Фактически: 34488 записей в long формате
