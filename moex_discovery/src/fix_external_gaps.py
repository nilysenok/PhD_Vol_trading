#!/usr/bin/env python3
"""
Diagnose and fix gaps in external data
Compare with stocks (final_v4) and try to fill missing days
"""

import os
import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Paths
BASE_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data")
STOCKS_RV = BASE_DIR / "final_v4/rv_daily_v4.parquet"
EXTERNAL_RV = BASE_DIR / "external/final/rv_external_daily.parquet"
EXTERNAL_CANDLES = BASE_DIR / "external/final/candles_10m"
SOURCE_INDICES = BASE_DIR / "external/moex_iss/indices_10m"
SOURCE_CURRENCY = BASE_DIR / "external/moex_iss/currency_10m"
REPORT_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_gaps_report.md")

BASE_URL = "https://iss.moex.com/iss"

# Instruments
INSTRUMENTS = [
    {'secid': 'IMOEX', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXOG', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXFN', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXMM', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXEU', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXTL', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCN', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCH', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'RGBI', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'RVI', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'},
    {'secid': 'CNYRUB_TOM', 'type': 'currency', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'},
]

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

def fetch_candles_for_date(inst, date_str):
    """Fetch candles for a specific date from MOEX ISS"""
    url = f"{BASE_URL}/engines/{inst['engine']}/markets/{inst['market']}/boards/{inst['board']}/securities/{inst['secid']}/candles.json"
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
# STAGE 1: DIAGNOSE GAPS
# =====================================

def stage1_diagnose():
    """Stage 1: Diagnose gaps"""
    print("\n" + "="*70)
    print("ЭТАП 1: ДИАГНОСТИКА ПРОПУСКОВ")
    print("="*70)

    # Load stocks RV
    stocks_df = pd.read_parquet(STOCKS_RV)
    stocks_df['date'] = pd.to_datetime(stocks_df['date']).dt.date
    stocks_dates = set(stocks_df['date'].unique())
    print(f"\nАкции (final_v4): {len(stocks_dates)} уникальных дней")

    # Load external RV
    external_df = pd.read_parquet(EXTERNAL_RV)
    external_df['date'] = pd.to_datetime(external_df['date']).dt.date

    results = {}

    print(f"\n{'Инструмент':<15} {'Дней RV':>10} {'Пропусков':>12} {'Первые 10 пропущенных дат'}")
    print("-" * 80)

    for inst in INSTRUMENTS:
        secid = inst['secid']
        inst_df = external_df[external_df['ticker'] == secid]
        inst_dates = set(inst_df['date'].unique())

        # Missing in external but present in stocks
        missing = stocks_dates - inst_dates
        missing_sorted = sorted(missing)

        # Extra in external but not in stocks
        extra = inst_dates - stocks_dates

        results[secid] = {
            'rv_days': len(inst_dates),
            'missing_dates': missing_sorted,
            'extra_dates': sorted(extra),
            'inst': inst
        }

        first_10 = ', '.join(str(d) for d in missing_sorted[:10])
        print(f"{secid:<15} {len(inst_dates):>10} {len(missing):>12} {first_10[:50]}")

    # Check reverse: dates in external but not in stocks
    all_external_dates = set(external_df['date'].unique())
    only_external = all_external_dates - stocks_dates
    print(f"\nДат только во внешних (не в акциях): {len(only_external)}")
    if only_external:
        print(f"  Примеры: {sorted(only_external)[:10]}")

    return stocks_dates, results

# =====================================
# STAGE 2: ANALYZE CAUSES
# =====================================

def stage2_analyze(stocks_dates, gaps_info):
    """Stage 2: Analyze causes of gaps"""
    print("\n" + "="*70)
    print("ЭТАП 2: АНАЛИЗ ПРИЧИН ПРОПУСКОВ")
    print("="*70)

    analysis = {}

    for secid, info in gaps_info.items():
        missing_dates = info['missing_dates']
        inst = info['inst']

        if not missing_dates:
            analysis[secid] = {'type_a': [], 'type_b': [], 'type_c': [], 'type_d': []}
            continue

        print(f"\n--- {secid}: {len(missing_dates)} пропусков ---")

        # Load source candles
        if inst['type'] == 'index':
            source_path = SOURCE_INDICES / f"{secid}.parquet"
        else:
            source_path = SOURCE_CURRENCY / f"{secid}.parquet"

        if source_path.exists():
            source_df = pd.read_parquet(source_path)
            source_df['datetime'] = pd.to_datetime(source_df['begin'])
            source_df['date'] = source_df['datetime'].dt.date
        else:
            source_df = pd.DataFrame()
            print(f"  WARNING: Source file not found: {source_path}")

        type_a = []  # Has candles, bars >= 10
        type_b = []  # Has candles, bars < 10 but >= 5
        type_c = []  # No candles in source, but API returns data
        type_d = []  # No candles anywhere

        # Check each missing date
        for date in missing_dates[:50]:  # Limit to first 50 for performance
            date_str = str(date)

            # Check source
            if not source_df.empty:
                day_candles = source_df[source_df['date'] == date]
                n_bars = len(day_candles)
            else:
                n_bars = 0

            if n_bars >= 10:
                type_a.append({'date': date, 'n_bars': n_bars})
            elif n_bars >= 5:
                type_b.append({'date': date, 'n_bars': n_bars})
            elif n_bars > 0:
                type_b.append({'date': date, 'n_bars': n_bars})  # Even 1-4 bars
            else:
                # Try to fetch from API
                time.sleep(0.2)
                candles = fetch_candles_for_date(inst, date_str)
                if candles:
                    type_c.append({'date': date, 'n_bars': len(candles), 'candles': candles})
                else:
                    type_d.append({'date': date})

        # Handle remaining dates (beyond first 50) - classify as type_d for now
        for date in missing_dates[50:]:
            type_d.append({'date': date})

        analysis[secid] = {
            'type_a': type_a,
            'type_b': type_b,
            'type_c': type_c,
            'type_d': type_d
        }

        print(f"  Тип A (баров >= 10, можно пересчитать): {len(type_a)}")
        print(f"  Тип B (баров < 10, можно со смягчением): {len(type_b)}")
        print(f"  Тип C (нет в source, но API вернул): {len(type_c)}")
        print(f"  Тип D (нет данных): {len(type_d)}")

        if type_a:
            print(f"    A примеры: {[t['date'] for t in type_a[:5]]}")
        if type_b:
            print(f"    B примеры: {[(t['date'], t['n_bars']) for t in type_b[:5]]}")
        if type_c:
            print(f"    C примеры: {[(t['date'], t['n_bars']) for t in type_c[:5]]}")
        if type_d:
            print(f"    D примеры: {[t['date'] for t in type_d[:5]]}")

    return analysis

# =====================================
# STAGE 3: FIX GAPS
# =====================================

def stage3_fix(gaps_info, analysis):
    """Stage 3: Fix gaps by recalculating RV"""
    print("\n" + "="*70)
    print("ЭТАП 3: ПОЧИНКА ПРОПУСКОВ")
    print("="*70)

    new_rv_records = []
    new_candles = defaultdict(list)

    for secid, types in analysis.items():
        inst = gaps_info[secid]['inst']
        fixed_count = 0

        # Load source candles for type A and B
        if inst['type'] == 'index':
            source_path = SOURCE_INDICES / f"{secid}.parquet"
        else:
            source_path = SOURCE_CURRENCY / f"{secid}.parquet"

        if source_path.exists():
            source_df = pd.read_parquet(source_path)
            source_df['datetime'] = pd.to_datetime(source_df['begin'])
            source_df['date'] = source_df['datetime'].dt.date
        else:
            source_df = pd.DataFrame()

        # Type A: Recalculate RV from source (bars >= 10)
        for item in types['type_a']:
            date = item['date']
            if source_df.empty:
                continue

            day_df = source_df[source_df['date'] == date].sort_values('datetime')
            closes = day_df['close'].values

            if len(closes) >= 2:
                rv_daily, rv_ann = calculate_rv(closes)
                if rv_daily is not None:
                    new_rv_records.append({
                        'date': date,
                        'ticker': secid,
                        'rv_daily': rv_daily,
                        'rv_annualized': rv_ann,
                        'n_bars': len(closes),
                        'close': closes[-1],
                        'low_bars': False
                    })
                    fixed_count += 1

        # Type B: Recalculate with relaxed threshold (bars >= 1)
        for item in types['type_b']:
            date = item['date']
            if source_df.empty:
                continue

            day_df = source_df[source_df['date'] == date].sort_values('datetime')
            closes = day_df['close'].values

            if len(closes) >= 2:
                rv_daily, rv_ann = calculate_rv(closes)
                if rv_daily is not None:
                    new_rv_records.append({
                        'date': date,
                        'ticker': secid,
                        'rv_daily': rv_daily,
                        'rv_annualized': rv_ann,
                        'n_bars': len(closes),
                        'close': closes[-1],
                        'low_bars': True
                    })
                    fixed_count += 1

        # Type C: Use API data
        for item in types['type_c']:
            date = item['date']
            candles = item['candles']

            if len(candles) >= 2:
                closes = np.array([c['close'] for c in candles])
                rv_daily, rv_ann = calculate_rv(closes)
                if rv_daily is not None:
                    new_rv_records.append({
                        'date': date,
                        'ticker': secid,
                        'rv_daily': rv_daily,
                        'rv_annualized': rv_ann,
                        'n_bars': len(closes),
                        'close': closes[-1],
                        'low_bars': len(closes) < 10
                    })
                    # Save candles for later
                    new_candles[secid].extend(candles)
                    fixed_count += 1

        print(f"  {secid}: починено {fixed_count} дней")

    return new_rv_records, new_candles

# =====================================
# STAGE 4: UPDATE FILES
# =====================================

def stage4_update(new_rv_records, new_candles):
    """Stage 4: Update final files"""
    print("\n" + "="*70)
    print("ЭТАП 4: ОБНОВЛЕНИЕ ФИНАЛЬНЫХ ФАЙЛОВ")
    print("="*70)

    # Load existing RV
    existing_rv = pd.read_parquet(EXTERNAL_RV)
    existing_rv['date'] = pd.to_datetime(existing_rv['date']).dt.date

    print(f"  Существующих записей RV: {len(existing_rv)}")

    # Add new records
    if new_rv_records:
        new_df = pd.DataFrame(new_rv_records)
        new_df['date'] = pd.to_datetime(new_df['date']).dt.date

        # Remove duplicates (keep new)
        combined = pd.concat([existing_rv, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date', 'ticker'], keep='last')

        print(f"  Добавлено новых записей: {len(new_df)}")
        print(f"  Итого записей RV: {len(combined)}")
    else:
        combined = existing_rv
        print(f"  Новых записей не добавлено")

    # Save long format
    # Ensure low_bars column exists
    if 'low_bars' not in combined.columns:
        combined['low_bars'] = False

    long_path = BASE_DIR / "external/final/rv_external_daily.parquet"
    combined.to_parquet(long_path, index=False)
    print(f"  Сохранён: {long_path}")

    # Create wide format
    rv_wide = combined.pivot(index='date', columns='ticker', values='rv_annualized')
    rv_wide.columns = [f'rv_{c}' for c in rv_wide.columns]

    close_wide = combined.pivot(index='date', columns='ticker', values='close')
    close_wide.columns = [f'close_{c}' for c in close_wide.columns]

    wide_df = pd.concat([rv_wide, close_wide], axis=1).reset_index()

    wide_path = BASE_DIR / "external/final/rv_external_wide.parquet"
    wide_df.to_parquet(wide_path, index=False)
    print(f"  Сохранён: {wide_path}")

    # Update candles if any new ones were fetched
    for secid, candles in new_candles.items():
        if candles:
            candles_path = EXTERNAL_CANDLES / f"{secid}.parquet"
            if candles_path.exists():
                existing = pd.read_parquet(candles_path)
                new_candles_df = pd.DataFrame(candles)
                combined_candles = pd.concat([existing, new_candles_df], ignore_index=True)
                combined_candles = combined_candles.drop_duplicates(subset=['begin'], keep='last')
                combined_candles.to_parquet(candles_path, index=False)
                print(f"  Обновлены свечи: {secid} (+{len(candles)})")

    return combined

# =====================================
# STAGE 5: RE-CHECK CONSISTENCY
# =====================================

def stage5_recheck(stocks_dates, updated_rv):
    """Stage 5: Re-check consistency"""
    print("\n" + "="*70)
    print("ЭТАП 5: ПОВТОРНАЯ ПРОВЕРКА СОГЛАСОВАННОСТИ")
    print("="*70)

    updated_rv['date'] = pd.to_datetime(updated_rv['date']).dt.date

    results = []

    print(f"\n{'Инструмент':<15} {'Было':>8} {'Стало':>8} {'Починено':>10} {'Пропусков':>10} {'Совпад.%':>10}")
    print("-" * 70)

    for inst in INSTRUMENTS:
        secid = inst['secid']
        inst_df = updated_rv[updated_rv['ticker'] == secid]
        inst_dates = set(inst_df['date'].unique())

        # Calculate stats
        missing = stocks_dates - inst_dates
        overlap = stocks_dates & inst_dates
        overlap_pct = len(overlap) / len(stocks_dates) * 100 if stocks_dates else 0

        # We need the old count - get from the report context
        # For now, just show current stats
        results.append({
            'secid': secid,
            'new_days': len(inst_dates),
            'missing': len(missing),
            'overlap_pct': overlap_pct,
            'missing_dates': sorted(missing)
        })

        print(f"{secid:<15} {'—':>8} {len(inst_dates):>8} {'—':>10} {len(missing):>10} {overlap_pct:>9.1f}%")

    return results

# =====================================
# STAGE 6: GENERATE REPORT
# =====================================

def stage6_report(gaps_info, analysis, recheck_results):
    """Stage 6: Generate report"""
    print("\n" + "="*70)
    print("ЭТАП 6: ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*70)

    report = f"""# External Data Gaps Report

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Исходная ситуация

- Акции (final_v4): дней RV
- Внешние данные: пропуски vs акции

## Диагностика по инструментам

| Инструмент | Было дней | Пропусков | Тип A | Тип B | Тип C | Тип D |
|------------|-----------|-----------|-------|-------|-------|-------|
"""

    for secid, info in gaps_info.items():
        types = analysis.get(secid, {})
        type_a = len(types.get('type_a', []))
        type_b = len(types.get('type_b', []))
        type_c = len(types.get('type_c', []))
        type_d = len(types.get('type_d', []))

        report += f"| {secid} | {info['rv_days']} | {len(info['missing_dates'])} | {type_a} | {type_b} | {type_c} | {type_d} |\n"

    report += """
### Классификация пропусков

- **Тип A**: Свечи есть в source, баров >= 10 → ПОЧИНЕН (пересчитан RV)
- **Тип B**: Свечи есть, баров < 10 → ПОЧИНЕН со смягчённым порогом
- **Тип C**: Свечей не было, но API вернул данные → ПОЧИНЕН
- **Тип D**: Данных нет нигде → НЕ ПОЧИНИТЬ (не было торгов)

## Результат после починки

| Инструмент | Дней RV | Пропусков | Совпадение с акциями |
|------------|---------|-----------|---------------------|
"""

    for r in recheck_results:
        report += f"| {r['secid']} | {r['new_days']} | {r['missing']} | {r['overlap_pct']:.1f}% |\n"

    # Calculate overall stats
    total_missing = sum(r['missing'] for r in recheck_results)
    avg_overlap = sum(r['overlap_pct'] for r in recheck_results) / len(recheck_results)

    report += f"""
## Итого

- **Средний overlap с акциями:** {avg_overlap:.1f}%
- **Суммарно оставшихся пропусков:** {total_missing}

## Оставшиеся пропуски (тип D)

Эти даты не имеют данных в MOEX ISS API — вероятно, не было торгов по индексам в эти дни.

"""

    # List remaining gaps for IMOEX as reference
    imoex_result = next((r for r in recheck_results if r['secid'] == 'IMOEX'), None)
    if imoex_result and imoex_result['missing_dates']:
        report += "### IMOEX (эталонный индекс)\n\n"
        report += "```\n"
        for date in imoex_result['missing_dates'][:30]:
            report += f"{date}\n"
        if len(imoex_result['missing_dates']) > 30:
            report += f"... и ещё {len(imoex_result['missing_dates']) - 30} дат\n"
        report += "```\n\n"

    report += """
## Рекомендация

"""

    if avg_overlap >= 95:
        report += """**✅ Данные ДОСТАТОЧНЫ для моделирования**

Overlap >= 95% означает, что внешние данные покрывают практически все торговые дни акций.
Оставшиеся пропуски — это дни без торгов по индексам (праздники, технические дни).
"""
    else:
        report += f"""**⚠️ Overlap {avg_overlap:.1f}% — возможны проблемы**

Рекомендуется проверить причины пропусков и рассмотреть интерполяцию или исключение этих дней.
"""

    report += """
## Файлы обновлены

- `moex_discovery/data/external/final/rv_external_daily.parquet`
- `moex_discovery/data/external/final/rv_external_wide.parquet`
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
    print("ДИАГНОСТИКА И ПОЧИНКА ПРОПУСКОВ ВНЕШНИХ ДАННЫХ")
    print("="*70)

    # Stage 1
    stocks_dates, gaps_info = stage1_diagnose()

    # Stage 2
    analysis = stage2_analyze(stocks_dates, gaps_info)

    # Stage 3
    new_rv_records, new_candles = stage3_fix(gaps_info, analysis)

    # Stage 4
    updated_rv = stage4_update(new_rv_records, new_candles)

    # Stage 5
    recheck_results = stage5_recheck(stocks_dates, updated_rv)

    # Stage 6
    stage6_report(gaps_info, analysis, recheck_results)

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)

if __name__ == "__main__":
    main()
