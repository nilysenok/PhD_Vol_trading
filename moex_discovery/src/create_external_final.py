#!/usr/bin/env python3
"""
Create final external data folder + calculate RV
Period: 2014-08-26 — 2026-02-03 (matches final_v4)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data")
INDICES_DIR = BASE_DIR / "external/moex_iss/indices_10m"
CURRENCY_DIR = BASE_DIR / "external/moex_iss/currency_10m"
FINAL_DIR = BASE_DIR / "external/final"
FINAL_CANDLES_DIR = FINAL_DIR / "candles_10m"
REPORT_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_final_report.md")

# Period
START_DATE = "2014-08-26"
END_DATE = "2026-02-03"

# Instruments (without USD000UTSTOM)
INSTRUMENTS = [
    # Indices
    {'secid': 'IMOEX', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXOG', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXFN', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXMM', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXEU', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXTL', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXCN', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'MOEXCH', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'RGBI', 'type': 'index', 'source': INDICES_DIR},
    {'secid': 'RVI', 'type': 'index', 'source': INDICES_DIR},
    # Currency
    {'secid': 'CNYRUB_TOM', 'type': 'currency', 'source': CURRENCY_DIR},
]

def stage1_create_final_folder():
    """Stage 1: Create final folder with trimmed candles"""
    print("\n" + "="*70)
    print("ЭТАП 1: СОЗДАНИЕ ФИНАЛЬНОЙ ПАПКИ")
    print("="*70)

    FINAL_CANDLES_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for inst in INSTRUMENTS:
        secid = inst['secid']
        source_path = inst['source'] / f"{secid}.parquet"

        print(f"\n--- {secid} ---")

        if not source_path.exists():
            print(f"  ERROR: File not found: {source_path}")
            continue

        # Read original
        df = pd.read_parquet(source_path)
        original_count = len(df)

        # Parse date
        df['datetime'] = pd.to_datetime(df['begin'])
        df['date'] = df['datetime'].dt.date

        # Trim to period
        start = pd.to_datetime(START_DATE).date()
        end = pd.to_datetime(END_DATE).date()
        df_trimmed = df[(df['date'] >= start) & (df['date'] <= end)].copy()
        trimmed_count = len(df_trimmed)

        # Drop helper columns for saving
        df_save = df_trimmed.drop(columns=['datetime', 'date'])

        # Save to final
        output_path = FINAL_CANDLES_DIR / f"{secid}.parquet"
        df_save.to_parquet(output_path, index=False)

        first_date = df_trimmed['date'].min()
        last_date = df_trimmed['date'].max()

        print(f"  Было: {original_count:,} → Стало: {trimmed_count:,}")
        print(f"  Период: {first_date} — {last_date}")

        results.append({
            'secid': secid,
            'type': inst['type'],
            'original_count': original_count,
            'trimmed_count': trimmed_count,
            'first_date': first_date,
            'last_date': last_date
        })

    return results

def stage2_quality_control():
    """Stage 2: Quality control for final candles"""
    print("\n" + "="*70)
    print("ЭТАП 2: КОНТРОЛЬ ФИНАЛЬНЫХ 10-МИНУТОК")
    print("="*70)

    results = []

    for inst in INSTRUMENTS:
        secid = inst['secid']
        path = FINAL_CANDLES_DIR / f"{secid}.parquet"

        if not path.exists():
            print(f"\n--- {secid}: FILE NOT FOUND ---")
            continue

        df = pd.read_parquet(path)
        df['datetime'] = pd.to_datetime(df['begin'])
        df['date'] = df['datetime'].dt.date

        total_candles = len(df)
        unique_days = df['date'].nunique()
        first_date = df['date'].min()
        last_date = df['date'].max()

        bars_per_day = df.groupby('date').size()
        avg_bars = bars_per_day.mean()
        min_bars = bars_per_day.min()
        max_bars = bars_per_day.max()

        print(f"\n--- {secid} ---")
        print(f"  Период: {first_date} — {last_date}")
        print(f"  Свечей: {total_candles:,}")
        print(f"  Дней: {unique_days:,}")
        print(f"  Баров/день: avg={avg_bars:.1f}, min={min_bars}, max={max_bars}")

        results.append({
            'secid': secid,
            'type': inst['type'],
            'first_date': first_date,
            'last_date': last_date,
            'total_candles': total_candles,
            'unique_days': unique_days,
            'avg_bars': round(avg_bars, 1),
            'min_bars': min_bars,
            'max_bars': max_bars
        })

    # Check coverage
    print("\n--- ПРОВЕРКА ПОКРЫТИЯ ---")
    start = pd.to_datetime(START_DATE).date()
    end = pd.to_datetime(END_DATE).date()

    all_cover_start = all(r['first_date'] <= start for r in results)
    all_cover_end = all(r['last_date'] >= end for r in results)

    print(f"  Все first_date <= {START_DATE}: {'✓' if all_cover_start else '✗'}")
    print(f"  Все last_date >= {END_DATE}: {'✓' if all_cover_end else '✗'}")

    for r in results:
        if r['first_date'] > start:
            print(f"    ⚠ {r['secid']}: first_date = {r['first_date']}")
        if r['last_date'] < end:
            print(f"    ⚠ {r['secid']}: last_date = {r['last_date']}")

    return results

def stage3_calculate_rv():
    """Stage 3: Calculate daily RV"""
    print("\n" + "="*70)
    print("ЭТАП 3: РАСЧЁТ DAILY RV")
    print("="*70)

    all_rv = []

    for inst in INSTRUMENTS:
        secid = inst['secid']
        path = FINAL_CANDLES_DIR / f"{secid}.parquet"

        if not path.exists():
            print(f"\n--- {secid}: SKIPPED (no file) ---")
            continue

        print(f"\n--- {secid} ---")

        df = pd.read_parquet(path)
        df['datetime'] = pd.to_datetime(df['begin'])
        df['date'] = df['datetime'].dt.date
        df = df.sort_values(['date', 'datetime'])

        # Calculate RV for each day
        rv_records = []
        skipped_days = 0

        for date, group in df.groupby('date'):
            n_bars = len(group)

            if n_bars < 10:
                skipped_days += 1
                continue

            closes = group['close'].values

            # Log returns between consecutive bars
            log_returns = np.log(closes[1:] / closes[:-1])

            # RV daily = sum of squared returns
            rv_daily = np.sum(log_returns ** 2)

            # RV annualized (percentage)
            rv_annualized = np.sqrt(rv_daily * 252) * 100

            # Last close of the day
            last_close = closes[-1]

            rv_records.append({
                'date': date,
                'ticker': secid,
                'rv_daily': rv_daily,
                'rv_annualized': rv_annualized,
                'n_bars': n_bars,
                'close': last_close
            })

        rv_df = pd.DataFrame(rv_records)
        all_rv.append(rv_df)

        if len(rv_df) > 0:
            print(f"  Дней RV: {len(rv_df):,}")
            print(f"  Пропущено (n_bars < 10): {skipped_days}")
            print(f"  RV%: avg={rv_df['rv_annualized'].mean():.1f}, med={rv_df['rv_annualized'].median():.1f}, max={rv_df['rv_annualized'].max():.1f}")
        else:
            print(f"  NO RV calculated")

    # Combine all
    combined_rv = pd.concat(all_rv, ignore_index=True)
    print(f"\n--- ИТОГО ---")
    print(f"  Всего записей RV: {len(combined_rv):,}")

    return combined_rv

def stage4_save_rv(rv_df):
    """Stage 4: Save RV in long and wide formats"""
    print("\n" + "="*70)
    print("ЭТАП 4: СОХРАНЕНИЕ RV")
    print("="*70)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # 4.1 Long format
    long_path = FINAL_DIR / "rv_external_daily.parquet"
    rv_df.to_parquet(long_path, index=False)
    print(f"\n  Long-формат: {long_path}")
    print(f"    Записей: {len(rv_df):,}")
    print(f"    Колонки: {list(rv_df.columns)}")

    # 4.2 Wide format
    # RV columns
    rv_wide = rv_df.pivot(index='date', columns='ticker', values='rv_annualized')
    rv_wide.columns = [f'rv_{c}' for c in rv_wide.columns]

    # Close columns
    close_wide = rv_df.pivot(index='date', columns='ticker', values='close')
    close_wide.columns = [f'close_{c}' for c in close_wide.columns]

    # Combine
    wide_df = pd.concat([rv_wide, close_wide], axis=1).reset_index()

    wide_path = FINAL_DIR / "rv_external_wide.parquet"
    wide_df.to_parquet(wide_path, index=False)
    print(f"\n  Wide-формат: {wide_path}")
    print(f"    Записей: {len(wide_df):,}")
    print(f"    Колонки: {len(wide_df.columns)}")

    return wide_df

def stage5_check_consistency(rv_df):
    """Stage 5: Check consistency with stocks"""
    print("\n" + "="*70)
    print("ЭТАП 5: ПРОВЕРКА СОГЛАСОВАННОСТИ С АКЦИЯМИ")
    print("="*70)

    stocks_path = BASE_DIR / "final_v4/rv_daily_v4.parquet"

    if not stocks_path.exists():
        print(f"  ERROR: Stocks file not found: {stocks_path}")
        return None

    stocks_df = pd.read_parquet(stocks_path)
    stocks_df['date'] = pd.to_datetime(stocks_df['date']).dt.date

    # Unique dates
    stocks_dates = set(stocks_df['date'].unique())

    # Use IMOEX as reference for external
    imoex_rv = rv_df[rv_df['ticker'] == 'IMOEX']
    external_dates = set(imoex_rv['date'].unique())

    # Comparison
    common_dates = stocks_dates & external_dates
    only_stocks = stocks_dates - external_dates
    only_external = external_dates - stocks_dates

    print(f"\n  Дней в акциях (final_v4): {len(stocks_dates):,}")
    print(f"  Дней во внешних (IMOEX): {len(external_dates):,}")
    print(f"  Совпадающих дней: {len(common_dates):,}")
    print(f"  Только в акциях: {len(only_stocks)}")
    print(f"  Только во внешних: {len(only_external)}")

    if only_stocks:
        sorted_only = sorted(only_stocks)[:10]
        print(f"\n  Даты только в акциях (первые 10): {sorted_only}")

    if only_external:
        sorted_only = sorted(only_external)[:10]
        print(f"\n  Даты только во внешних (первые 10): {sorted_only}")

    # Overlap percentage
    overlap_pct = len(common_dates) / len(stocks_dates) * 100 if stocks_dates else 0
    print(f"\n  Покрытие акций внешними: {overlap_pct:.1f}%")

    return {
        'stocks_days': len(stocks_dates),
        'external_days': len(external_dates),
        'common_days': len(common_dates),
        'only_stocks': len(only_stocks),
        'only_external': len(only_external),
        'overlap_pct': overlap_pct,
        'stocks_first': min(stocks_dates),
        'stocks_last': max(stocks_dates),
        'external_first': min(external_dates),
        'external_last': max(external_dates)
    }

def stage6_generate_report(qc_results, rv_df, consistency):
    """Stage 6: Generate final report"""
    print("\n" + "="*70)
    print("ЭТАП 6: ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*70)

    # Calculate RV stats per ticker
    rv_stats = rv_df.groupby('ticker').agg({
        'rv_annualized': ['count', 'mean', 'median', 'max'],
        'date': ['min', 'max']
    }).reset_index()
    rv_stats.columns = ['ticker', 'rv_days', 'rv_mean', 'rv_median', 'rv_max', 'first_date', 'last_date']

    # Find max RV date for each ticker
    max_rv_dates = rv_df.loc[rv_df.groupby('ticker')['rv_annualized'].idxmax()][['ticker', 'date', 'rv_annualized']]
    max_rv_dates = max_rv_dates.rename(columns={'date': 'max_rv_date'})
    rv_stats = rv_stats.merge(max_rv_dates[['ticker', 'max_rv_date']], on='ticker')

    # Skipped days (difference between QC days and RV days)
    qc_dict = {r['secid']: r['unique_days'] for r in qc_results}
    rv_stats['qc_days'] = rv_stats['ticker'].map(qc_dict)
    rv_stats['skipped_days'] = rv_stats['qc_days'] - rv_stats['rv_days']

    report = f"""# External Data Final Report

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Период:** {START_DATE} — {END_DATE}

## Таблица 10-минуток

| # | Инструмент | Тип | Свечей | Дней | Баров/день | Период |
|---|------------|-----|--------|------|------------|--------|
"""

    for i, r in enumerate(qc_results, 1):
        report += f"| {i} | {r['secid']} | {r['type']} | {r['total_candles']:,} | {r['unique_days']:,} | {r['avg_bars']} | {r['first_date']} — {r['last_date']} |\n"

    report += f"""
## Таблица RV

| # | Инструмент | Дней RV | Ср. RV% | Мед. RV% | Макс RV% | Дата макс | Пропущено |
|---|------------|---------|---------|----------|----------|-----------|-----------|
"""

    for i, (_, row) in enumerate(rv_stats.iterrows(), 1):
        report += f"| {i} | {row['ticker']} | {row['rv_days']:,} | {row['rv_mean']:.1f} | {row['rv_median']:.1f} | {row['rv_max']:.1f} | {row['max_rv_date']} | {row['skipped_days']} |\n"

    if consistency:
        report += f"""
## Согласованность с акциями

| Данные | Период | Дней | Совпадение |
|--------|--------|------|------------|
| Акции (final_v4) | {consistency['stocks_first']} — {consistency['stocks_last']} | {consistency['stocks_days']:,} | — |
| Внешние (IMOEX) | {consistency['external_first']} — {consistency['external_last']} | {consistency['external_days']:,} | {consistency['overlap_pct']:.1f}% |

- Совпадающих дней: {consistency['common_days']:,}
- Дней только в акциях: {consistency['only_stocks']}
- Дней только во внешних: {consistency['only_external']}
"""

    report += f"""
## Структура данных

```
moex_discovery/data/external/
├── moex_iss/                         ← исходные (НЕ ТРОГАТЬ)
│   ├── indices_10m/                  ← 10 индексов, полный период
│   │   ├── IMOEX.parquet
│   │   ├── MOEXOG.parquet
│   │   ├── ...
│   │   └── RVI.parquet
│   └── currency_10m/                 ← CNYRUB + USD
│       ├── CNYRUB_TOM.parquet
│       └── USD000UTSTOM.parquet     ← excluded from final
└── final/                            ← ФИНАЛЬНЫЕ ДАННЫЕ
    ├── candles_10m/                  ← 11 инструментов, обрезанные
    │   ├── IMOEX.parquet
    │   ├── MOEXOG.parquet
    │   ├── ...
    │   ├── RVI.parquet
    │   └── CNYRUB_TOM.parquet
    ├── rv_external_daily.parquet     ← RV long-формат
    └── rv_external_wide.parquet      ← RV wide-формат
```

## Примечания

- USD000UTSTOM исключён из финальной выборки (данные обрываются 2024-06-11)
- RV рассчитан только для дней с >= 10 баров
- Пропуск 2022-02-25 — 2022-03-28 связан с приостановкой торгов

## Формат RV файлов

### rv_external_daily.parquet (long)
- `date`: дата
- `ticker`: инструмент
- `rv_daily`: realized variance (sum of squared log returns)
- `rv_annualized`: годовая волатильность в процентах
- `n_bars`: количество баров в день
- `close`: цена закрытия дня

### rv_external_wide.parquet (wide)
- `date`: дата
- `rv_IMOEX`, `rv_MOEXOG`, ... : RV по каждому инструменту
- `close_IMOEX`, `close_MOEXOG`, ... : close по каждому инструменту
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  Отчёт сохранён: {REPORT_PATH}")

def main():
    print("="*70)
    print("CREATE EXTERNAL FINAL + CALCULATE RV")
    print(f"Period: {START_DATE} — {END_DATE}")
    print("="*70)

    # Stage 1
    stage1_create_final_folder()

    # Stage 2
    qc_results = stage2_quality_control()

    # Stage 3
    rv_df = stage3_calculate_rv()

    # Stage 4
    stage4_save_rv(rv_df)

    # Stage 5
    consistency = stage5_check_consistency(rv_df)

    # Stage 6
    stage6_generate_report(qc_results, rv_df, consistency)

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)

if __name__ == "__main__":
    main()
