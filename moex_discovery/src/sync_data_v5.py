#!/usr/bin/env python3
"""
Synchronize stocks and external data to 100% overlap
Create final_v5 with perfectly aligned dates
"""

import os
import shutil
import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
BASE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data")
STOCKS_RV = BASE_DIR / "final_v4/rv_daily_v4.parquet"
EXTERNAL_RV = BASE_DIR / "external/final/rv_external_daily.parquet"
STOCKS_CANDLES = BASE_DIR / "final_v4/candles_10m"
EXTERNAL_CANDLES = BASE_DIR / "external/final/candles_10m"
RAW_CANDLES = BASE_DIR / "raw/candles_10m"

FINAL_V5 = BASE_DIR / "final_v5"
REPORT_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/SYNC_REPORT.md")

BASE_URL = "https://iss.moex.com/iss"

# External instruments
EXTERNAL_INSTRUMENTS = [
    {'secid': 'IMOEX', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXOG', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXFN', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXMM', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXEU', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXTL', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCN', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCH', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'RGBI', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'RVI', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'},
    {'secid': 'CNYRUB_TOM', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'},
]

# Stock tickers
STOCK_TICKERS = ['SBER', 'LKOH', 'TATN', 'NVTK', 'VTBR', 'ALRS', 'AFLT', 'HYDR',
                 'MGNT', 'MOEX', 'RTKM', 'MTSS', 'MTLR', 'IRAO', 'OGKB', 'PHOR', 'LSRG']

def get_json(url, params=None, retries=3, timeout=15):
    """Get JSON from MOEX ISS API"""
    if params is None:
        params = {}
    params['iss.json'] = 'extended'
    params['iss.meta'] = 'off'

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

def fetch_candles(engine, market, board, secid, date_str):
    """Fetch candles for a specific date"""
    url = f"{BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"
    params = {'interval': 10, 'from': date_str, 'till': date_str}
    data = get_json(url, params)

    if not data or len(data) < 2:
        return []

    return data[1].get('candles', [])

def calculate_rv(closes):
    """Calculate RV from close prices"""
    if len(closes) < 2:
        return None, None

    log_returns = np.log(closes[1:] / closes[:-1])
    rv_daily = np.sum(log_returns ** 2)
    rv_annualized = np.sqrt(rv_daily * 252) * 100

    return rv_daily, rv_annualized

# =====================================
# STAGE 1: FIND ALL DISCREPANCIES
# =====================================

def stage1_find_discrepancies():
    """Find all date discrepancies between stocks and external"""
    print("\n" + "="*70)
    print("ЭТАП 1: НАЙТИ ВСЕ РАСХОЖДЕНИЯ")
    print("="*70)

    # Load stocks
    stocks_df = pd.read_parquet(STOCKS_RV)
    stocks_df['date'] = pd.to_datetime(stocks_df['date']).dt.date
    stocks_dates = set(stocks_df['date'].unique())
    print(f"\nАкции: {len(stocks_dates)} уникальных дней")

    # Load external (use IMOEX as reference)
    external_df = pd.read_parquet(EXTERNAL_RV)
    external_df['date'] = pd.to_datetime(external_df['date']).dt.date
    imoex_df = external_df[external_df['ticker'] == 'IMOEX']
    external_dates = set(imoex_df['date'].unique())
    print(f"Внешние (IMOEX): {len(external_dates)} уникальных дней")

    # Find discrepancies
    set_a = stocks_dates - external_dates  # In stocks, not in external
    set_b = external_dates - stocks_dates  # In external, not in stocks

    print(f"\nSET_A (в акциях, нет во внешних): {len(set_a)} дней")
    print(f"SET_B (во внешних, нет в акциях): {len(set_b)} дней")

    if set_a:
        print(f"\nПолный список SET_A:")
        for d in sorted(set_a):
            print(f"  {d}")

    if set_b:
        print(f"\nПолный список SET_B:")
        for d in sorted(set_b):
            print(f"  {d}")

    return stocks_df, external_df, stocks_dates, external_dates, set_a, set_b

# =====================================
# STAGE 2: TRY TO FETCH EXTERNAL FOR SET_A
# =====================================

def stage2_fetch_external(set_a, external_df):
    """Try to fetch external data for dates only in stocks"""
    print("\n" + "="*70)
    print("ЭТАП 2: ДОЗАГРУЗКА ВНЕШНИХ (SET_A)")
    print("="*70)

    if not set_a:
        print("  SET_A пуст, пропускаем")
        return [], set_a

    new_records = []
    unfixable = []

    for i, date in enumerate(sorted(set_a)):
        date_str = str(date)
        print(f"\n[{i+1}/{len(set_a)}] {date_str}")

        # Try IMOEX first
        time.sleep(0.2)
        candles = fetch_candles('stock', 'index', 'SNDX', 'IMOEX', date_str)

        if candles and len(candles) >= 2:
            closes = np.array([c['close'] for c in candles])
            rv_daily, rv_ann = calculate_rv(closes)

            if rv_daily is not None:
                new_records.append({
                    'date': date,
                    'ticker': 'IMOEX',
                    'rv_daily': rv_daily,
                    'rv_annualized': rv_ann,
                    'n_bars': len(closes),
                    'close': closes[-1],
                    'low_bars': len(closes) < 10
                })
                print(f"  IMOEX: OK ({len(closes)} баров)")

                # Fetch other instruments
                for inst in EXTERNAL_INSTRUMENTS[1:]:  # Skip IMOEX
                    time.sleep(0.15)
                    inst_candles = fetch_candles(inst['engine'], inst['market'], inst['board'], inst['secid'], date_str)

                    if inst_candles and len(inst_candles) >= 2:
                        inst_closes = np.array([c['close'] for c in inst_candles])
                        inst_rv_daily, inst_rv_ann = calculate_rv(inst_closes)

                        if inst_rv_daily is not None:
                            new_records.append({
                                'date': date,
                                'ticker': inst['secid'],
                                'rv_daily': inst_rv_daily,
                                'rv_annualized': inst_rv_ann,
                                'n_bars': len(inst_closes),
                                'close': inst_closes[-1],
                                'low_bars': len(inst_closes) < 10
                            })
        else:
            print(f"  IMOEX: НЕТ ДАННЫХ")
            unfixable.append(date)

    print(f"\n--- Итог этапа 2 ---")
    print(f"  Дозагружено записей: {len(new_records)}")
    print(f"  Unfixable: {len(unfixable)}")

    return new_records, set(unfixable)

# =====================================
# STAGE 3: TRY TO FETCH STOCKS FOR SET_B
# =====================================

def stage3_fetch_stocks(set_b, stocks_df):
    """Try to fetch stock data for dates only in external"""
    print("\n" + "="*70)
    print("ЭТАП 3: ДОЗАГРУЗКА АКЦИЙ (SET_B)")
    print("="*70)

    if not set_b:
        print("  SET_B пуст, пропускаем")
        return [], set_b

    new_records = []
    unfixable = []

    # Load raw candles for main tickers
    raw_candles = {}
    for ticker in ['SBER', 'LKOH', 'VTBR']:
        raw_path = RAW_CANDLES / f"{ticker}.parquet"
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            # Handle different column names
            if 'begin' in df.columns:
                df['datetime'] = pd.to_datetime(df['begin'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                continue
            df['date'] = df['datetime'].dt.date
            raw_candles[ticker] = df

    for i, date in enumerate(sorted(set_b)):
        date_str = str(date)
        print(f"\n[{i+1}/{len(set_b)}] {date_str}")

        # Check raw candles
        has_data = False
        for ticker, df in raw_candles.items():
            day_df = df[df['date'] == date]
            if len(day_df) >= 5:
                has_data = True
                print(f"  {ticker}: {len(day_df)} баров в raw")
                break

        if has_data:
            # Fetch all 17 tickers
            for ticker in STOCK_TICKERS:
                time.sleep(0.15)
                candles = fetch_candles('stock', 'shares', 'TQBR', ticker, date_str)

                if candles and len(candles) >= 2:
                    closes = np.array([c['close'] for c in candles])
                    rv_daily, rv_ann = calculate_rv(closes)

                    if rv_daily is not None:
                        new_records.append({
                            'date': date,
                            'ticker': ticker,
                            'rv_daily': rv_daily,
                            'rv_annualized': rv_ann,
                            'n_bars': len(closes),
                            'close': closes[-1]
                        })
        else:
            print(f"  НЕТ ДАННЫХ в raw")
            unfixable.append(date)

    print(f"\n--- Итог этапа 3 ---")
    print(f"  Дозагружено записей: {len(new_records)}")
    print(f"  Unfixable: {len(unfixable)}")

    return new_records, set(unfixable)

# =====================================
# STAGE 4: TRIM TO COMMON INTERSECTION
# =====================================

def stage4_trim(stocks_df, external_df, new_external, new_stocks, unfixable_external, unfixable_stocks):
    """Trim both datasets to common date intersection"""
    print("\n" + "="*70)
    print("ЭТАП 4: ОБРЕЗКА ДО ОБЩЕГО ПЕРЕСЕЧЕНИЯ")
    print("="*70)

    # Add new records
    if new_external:
        new_ext_df = pd.DataFrame(new_external)
        new_ext_df['date'] = pd.to_datetime(new_ext_df['date']).dt.date
        external_df = pd.concat([external_df, new_ext_df], ignore_index=True)
        external_df = external_df.drop_duplicates(subset=['date', 'ticker'], keep='last')

    if new_stocks:
        new_stk_df = pd.DataFrame(new_stocks)
        new_stk_df['date'] = pd.to_datetime(new_stk_df['date']).dt.date
        stocks_df = pd.concat([stocks_df, new_stk_df], ignore_index=True)
        stocks_df = stocks_df.drop_duplicates(subset=['date', 'ticker'], keep='last')

    # Recalculate dates
    stocks_dates = set(stocks_df['date'].unique())
    imoex_df = external_df[external_df['ticker'] == 'IMOEX']
    external_dates = set(imoex_df['date'].unique())

    # Common dates
    common_dates = stocks_dates & external_dates
    print(f"\nДней в акциях: {len(stocks_dates)}")
    print(f"Дней во внешних (IMOEX): {len(external_dates)}")
    print(f"Общее пересечение: {len(common_dates)}")

    # Lost dates
    lost_stocks = stocks_dates - common_dates
    lost_external = external_dates - common_dates
    print(f"\nПотеряно из акций: {len(lost_stocks)}")
    print(f"Потеряно из внешних: {len(lost_external)}")

    if lost_stocks:
        print(f"  Даты потерянные из акций: {sorted(lost_stocks)[:10]}...")
    if lost_external:
        print(f"  Даты потерянные из внешних: {sorted(lost_external)[:10]}...")

    # Trim
    stocks_trimmed = stocks_df[stocks_df['date'].isin(common_dates)].copy()
    external_trimmed = external_df[external_df['date'].isin(common_dates)].copy()

    print(f"\nАкции после обрезки: {len(stocks_trimmed)} записей, {stocks_trimmed['date'].nunique()} дней")
    print(f"Внешние после обрезки: {len(external_trimmed)} записей, {external_trimmed['date'].nunique()} дней")

    return stocks_trimmed, external_trimmed, common_dates, lost_stocks, lost_external

# =====================================
# STAGE 5: SAVE SYNCHRONIZED VERSIONS
# =====================================

def stage5_save(stocks_df, external_df, common_dates):
    """Save synchronized versions to final_v5"""
    print("\n" + "="*70)
    print("ЭТАП 5: СОХРАНЕНИЕ СИНХРОНИЗИРОВАННЫХ ВЕРСИЙ")
    print("="*70)

    # Create directory
    FINAL_V5.mkdir(parents=True, exist_ok=True)

    # 5.1 Save stocks RV
    stocks_path = FINAL_V5 / "rv_stocks.parquet"
    stocks_df.to_parquet(stocks_path, index=False)
    print(f"\n  Сохранён: {stocks_path}")
    print(f"    {len(stocks_df)} записей, {stocks_df['ticker'].nunique()} тикеров")

    # 5.2 Save external long
    external_long_path = FINAL_V5 / "rv_external_daily.parquet"
    external_df.to_parquet(external_long_path, index=False)
    print(f"\n  Сохранён: {external_long_path}")
    print(f"    {len(external_df)} записей, {external_df['ticker'].nunique()} инструментов")

    # 5.3 Save external wide
    rv_wide = external_df.pivot(index='date', columns='ticker', values='rv_annualized')
    rv_wide.columns = [f'rv_{c}' for c in rv_wide.columns]

    close_wide = external_df.pivot(index='date', columns='ticker', values='close')
    close_wide.columns = [f'close_{c}' for c in close_wide.columns]

    wide_df = pd.concat([rv_wide, close_wide], axis=1).reset_index()
    wide_path = FINAL_V5 / "rv_external_wide.parquet"
    wide_df.to_parquet(wide_path, index=False)
    print(f"\n  Сохранён: {wide_path}")
    print(f"    {len(wide_df)} дней, {len(wide_df.columns)} колонок")

    # 5.4 Copy candles
    stocks_candles_dst = FINAL_V5 / "stocks_candles_10m"
    external_candles_dst = FINAL_V5 / "external_candles_10m"

    if STOCKS_CANDLES.exists():
        if stocks_candles_dst.exists():
            shutil.rmtree(stocks_candles_dst)
        shutil.copytree(STOCKS_CANDLES, stocks_candles_dst)
        print(f"\n  Скопированы свечи акций: {stocks_candles_dst}")

    if EXTERNAL_CANDLES.exists():
        if external_candles_dst.exists():
            shutil.rmtree(external_candles_dst)
        shutil.copytree(EXTERNAL_CANDLES, external_candles_dst)
        print(f"  Скопированы свечи внешних: {external_candles_dst}")

    # Save common dates list
    dates_df = pd.DataFrame({'date': sorted(common_dates)})
    dates_path = FINAL_V5 / "common_dates.csv"
    dates_df.to_csv(dates_path, index=False)
    print(f"\n  Сохранён список дат: {dates_path}")

    return wide_df

# =====================================
# STAGE 6: FINAL CHECK 100%
# =====================================

def stage6_final_check(stocks_df, external_df):
    """Final check for 100% overlap"""
    print("\n" + "="*70)
    print("ЭТАП 6: ФИНАЛЬНАЯ ПРОВЕРКА 100%")
    print("="*70)

    stocks_dates = set(stocks_df['date'].unique())

    results = []

    # Check each external instrument
    print(f"\n{'Инструмент':<15} {'Дней RV':>10} {'Совпадение':>12}")
    print("-" * 40)

    for inst in EXTERNAL_INSTRUMENTS:
        secid = inst['secid']
        inst_df = external_df[external_df['ticker'] == secid]
        inst_dates = set(inst_df['date'].unique())

        overlap = stocks_dates & inst_dates
        overlap_pct = len(overlap) / len(stocks_dates) * 100 if stocks_dates else 0

        results.append({
            'secid': secid,
            'days': len(inst_dates),
            'overlap': len(overlap),
            'overlap_pct': overlap_pct
        })

        print(f"{secid:<15} {len(inst_dates):>10} {overlap_pct:>11.1f}%")

    # Summary
    imoex_result = next((r for r in results if r['secid'] == 'IMOEX'), None)
    is_100 = imoex_result and imoex_result['overlap_pct'] == 100.0

    print(f"\n--- ИТОГ ---")
    print(f"  Дней в акциях: {len(stocks_dates)}")
    print(f"  IMOEX overlap: {imoex_result['overlap_pct']:.1f}%" if imoex_result else "N/A")
    print(f"  100% совпадение: {'✅ ДА' if is_100 else '❌ НЕТ'}")

    return results, is_100

# =====================================
# STAGE 7: REPORT
# =====================================

def stage7_report(stocks_df, external_df, set_a, set_b, lost_stocks, lost_external, check_results, is_100):
    """Generate sync report"""
    print("\n" + "="*70)
    print("ЭТАП 7: ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*70)

    stocks_dates = stocks_df['date'].unique()
    first_date = min(stocks_dates)
    last_date = max(stocks_dates)

    report = f"""# Sync Report: Final V5

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Исходные расхождения

- SET_A (в акциях, нет во внешних): **{len(set_a)}** дней
- SET_B (во внешних, нет в акциях): **{len(set_b)}** дней

## Обрезка до пересечения

- Потеряно из акций: **{len(lost_stocks)}** дней
- Потеряно из внешних: **{len(lost_external)}** дней

## Итоговый датасет

| Параметр | Значение |
|----------|----------|
| Период | {first_date} — {last_date} |
| Дней | {len(stocks_dates)} |
| Тикеров акций | {stocks_df['ticker'].nunique()} |
| Внешних инструментов | {external_df['ticker'].nunique()} |

## Проверка совпадения

| Инструмент | Дней RV | Overlap с акциями |
|------------|---------|-------------------|
"""

    for r in check_results:
        report += f"| {r['secid']} | {r['days']} | {r['overlap_pct']:.1f}% |\n"

    report += f"""
## Статус

**{'✅ 100% СИНХРОНИЗАЦИЯ ДОСТИГНУТА' if is_100 else '⚠️ НЕ 100%'}**

{'IMOEX и акции теперь имеют идентичный набор дат.' if is_100 else 'Остались расхождения.'}

## Структура final_v5/

```
moex_discovery/data/final_v5/
├── rv_stocks.parquet           ← RV 17 акций, {len(stocks_dates)} дней
├── rv_external_daily.parquet   ← RV 11 внешних (long), {len(stocks_dates)} дней
├── rv_external_wide.parquet    ← RV 11 внешних (wide)
├── common_dates.csv            ← Список общих дат
├── stocks_candles_10m/         ← 10-мин свечи 17 акций (полные)
│   ├── SBER.parquet
│   └── ...
└── external_candles_10m/       ← 10-мин свечи 11 внешних (полные)
    ├── IMOEX.parquet
    └── ...
```

## Примечания

- **RVI** имеет меньше дней ({next((r['days'] for r in check_results if r['secid'] == 'RVI'), 'N/A')}) — индекс волатильности не торгуется каждый день
- **CNYRUB_TOM** имеет меньше дней ({next((r['days'] for r in check_results if r['secid'] == 'CNYRUB_TOM'), 'N/A')}) — валютная секция имеет свой календарь
- **RGBI** имеет меньше дней ({next((r['days'] for r in check_results if r['secid'] == 'RGBI'), 'N/A')}) — облигационный индекс

Это нормально — эти инструменты можно использовать с пропусками (NaN в wide-формате).

## Использование

```python
import pandas as pd

# Акции
stocks = pd.read_parquet('moex_discovery/data/final_v5/rv_stocks.parquet')

# Внешние (long)
external = pd.read_parquet('moex_discovery/data/final_v5/rv_external_daily.parquet')

# Внешние (wide) - удобно для merge
external_wide = pd.read_parquet('moex_discovery/data/final_v5/rv_external_wide.parquet')

# Merge акции + внешние
merged = stocks.merge(external_wide, on='date', how='left')
```
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n  Отчёт сохранён: {REPORT_PATH}")

# =====================================
# MAIN
# =====================================

def main():
    print("="*70)
    print("СИНХРОНИЗАЦИЯ АКЦИЙ И ВНЕШНИХ ДАННЫХ ДО 100%")
    print("="*70)

    # Stage 1
    stocks_df, external_df, stocks_dates, external_dates, set_a, set_b = stage1_find_discrepancies()

    # Stage 2
    new_external, unfixable_external = stage2_fetch_external(set_a, external_df)

    # Stage 3
    new_stocks, unfixable_stocks = stage3_fetch_stocks(set_b, stocks_df)

    # Stage 4
    stocks_trimmed, external_trimmed, common_dates, lost_stocks, lost_external = stage4_trim(
        stocks_df, external_df, new_external, new_stocks, unfixable_external, unfixable_stocks
    )

    # Stage 5
    stage5_save(stocks_trimmed, external_trimmed, common_dates)

    # Stage 6
    check_results, is_100 = stage6_final_check(stocks_trimmed, external_trimmed)

    # Stage 7
    stage7_report(stocks_trimmed, external_trimmed, set_a, set_b, lost_stocks, lost_external, check_results, is_100)

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)

if __name__ == "__main__":
    main()
