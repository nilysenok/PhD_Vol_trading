#!/usr/bin/env python3
"""
Download 10-min candles for indices and currency from MOEX ISS
Full available period - no date truncation
"""

import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://iss.moex.com/iss"

# Output directories
INDICES_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/external/moex_iss/indices_10m")
CURRENCY_DIR = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/data/external/moex_iss/currency_10m")
REPORT_PATH = Path("/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_download_01.md")

# Instruments to download
INSTRUMENTS = [
    # Indices on SNDX board
    {'secid': 'IMOEX', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXOG', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXFN', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXMM', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXEU', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXTL', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCN', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'MOEXCH', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    {'secid': 'RGBI', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'SNDX'},
    # RVI on RTSI board
    {'secid': 'RVI', 'type': 'index', 'engine': 'stock', 'market': 'index', 'board': 'RTSI'},
    # Currency on CETS board
    {'secid': 'CNYRUB_TOM', 'type': 'currency', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'},
    {'secid': 'USD000UTSTOM', 'type': 'currency', 'engine': 'currency', 'market': 'selt', 'board': 'CETS'},
]

def get_json(url, params=None, retries=3, timeout=15):
    """Get JSON from MOEX ISS API with retries"""
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
                print(f"    ERROR after {retries} attempts: {e}")
                return None
    return None

def get_candleborders(instrument, interval=10):
    """Get available date range for candles"""
    url = f"{BASE_URL}/engines/{instrument['engine']}/markets/{instrument['market']}/boards/{instrument['board']}/securities/{instrument['secid']}/candleborders.json"
    data = get_json(url)

    if not data or len(data) < 2:
        return None, None

    borders = data[1].get('borders', [])
    for row in borders:
        if isinstance(row, dict) and row.get('interval') == interval:
            return row.get('begin'), row.get('end')

    return None, None

def download_candles(instrument, start_date, end_date):
    """Download all candles with pagination"""
    all_candles = []
    current_date = start_date[:10]  # YYYY-MM-DD
    end_date_str = end_date[:10]

    request_count = 0

    while current_date <= end_date_str:
        url = f"{BASE_URL}/engines/{instrument['engine']}/markets/{instrument['market']}/boards/{instrument['board']}/securities/{instrument['secid']}/candles.json"
        params = {
            'interval': 10,
            'from': current_date,
            'till': end_date_str
        }

        data = get_json(url, params)
        request_count += 1

        if not data or len(data) < 2:
            break

        candles = data[1].get('candles', [])
        if not candles:
            break

        all_candles.extend(candles)

        # Get last candle date and move to next day
        last_candle = candles[-1]
        if isinstance(last_candle, dict):
            last_begin = last_candle.get('begin', '')
            if last_begin:
                last_date = datetime.strptime(last_begin[:10], '%Y-%m-%d')
                next_date = last_date + timedelta(days=1)
                current_date = next_date.strftime('%Y-%m-%d')
            else:
                break
        else:
            break

        # Rate limiting
        time.sleep(0.2)

        # Progress every 50 requests
        if request_count % 50 == 0:
            print(f"    Requests: {request_count}, Candles: {len(all_candles)}")

    return all_candles, request_count

def analyze_candles(df):
    """Analyze candle data quality"""
    if df.empty:
        return {}

    df['date'] = pd.to_datetime(df['begin']).dt.date

    # Basic stats
    first_date = df['date'].min()
    last_date = df['date'].max()
    total_candles = len(df)
    unique_days = df['date'].nunique()

    # Bars per day stats
    bars_per_day = df.groupby('date').size()
    avg_bars = bars_per_day.mean()
    min_bars = bars_per_day.min()
    max_bars = bars_per_day.max()

    # Find gaps > 5 business days
    dates = pd.to_datetime(df['date'].unique())
    dates = dates.sort_values()
    gaps = []
    for i in range(1, len(dates)):
        diff = (dates[i] - dates[i-1]).days
        if diff > 7:  # More than a week (accounting for weekends)
            gaps.append({
                'from': dates[i-1].strftime('%Y-%m-%d'),
                'to': dates[i].strftime('%Y-%m-%d'),
                'days': diff
            })

    return {
        'first_date': first_date,
        'last_date': last_date,
        'total_candles': total_candles,
        'unique_days': unique_days,
        'avg_bars_per_day': round(avg_bars, 1),
        'min_bars_per_day': min_bars,
        'max_bars_per_day': max_bars,
        'gaps': gaps
    }

def download_instrument(instrument):
    """Download and save one instrument"""
    secid = instrument['secid']
    inst_type = instrument['type']

    print(f"\n{'='*60}")
    print(f"DOWNLOADING: {secid} ({inst_type})")
    print('='*60)

    # Get available period
    begin, end = get_candleborders(instrument)
    if not begin or not end:
        print(f"  ERROR: Could not get candleborders for {secid}")
        return None

    print(f"  Available period: {begin[:10]} — {end[:10]}")

    # Download candles
    print(f"  Downloading...")
    candles, request_count = download_candles(instrument, begin, end)
    print(f"  Requests made: {request_count}")
    print(f"  Candles received: {len(candles)}")

    if not candles:
        print(f"  ERROR: No candles received for {secid}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Select and rename columns
    columns_to_keep = ['begin', 'end', 'open', 'close', 'high', 'low', 'value', 'volume']
    df = df[[c for c in columns_to_keep if c in df.columns]]

    # Analyze quality
    stats = analyze_candles(df)

    # Print stats
    print(f"\n  QUALITY REPORT:")
    print(f"    First date:    {stats['first_date']}")
    print(f"    Last date:     {stats['last_date']}")
    print(f"    Total candles: {stats['total_candles']:,}")
    print(f"    Unique days:   {stats['unique_days']:,}")
    print(f"    Bars/day:      avg={stats['avg_bars_per_day']}, min={stats['min_bars_per_day']}, max={stats['max_bars_per_day']}")

    if stats['gaps']:
        print(f"    Gaps > 7 days: {len(stats['gaps'])}")
        for gap in stats['gaps'][:5]:  # Show first 5
            print(f"      {gap['from']} — {gap['to']} ({gap['days']} days)")
    else:
        print(f"    Gaps > 7 days: None")

    # Save to parquet
    if inst_type == 'index':
        output_dir = INDICES_DIR
    else:
        output_dir = CURRENCY_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{secid}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    return {
        'secid': secid,
        'type': inst_type,
        'board': instrument['board'],
        'stats': stats
    }

def generate_report(results):
    """Generate markdown report"""
    report = f"""# External Data Download Report: MOEX ISS Indices & Currency

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Source:** MOEX ISS API
**Interval:** 10 minutes

## Summary

| # | Instrument | Type | Board | Candles | Days | Bars/day | First Date | Last Date |
|---|------------|------|-------|---------|------|----------|------------|-----------|
"""

    total_candles = 0
    total_days = 0

    for i, r in enumerate(results, 1):
        if r is None:
            continue
        s = r['stats']
        total_candles += s['total_candles']
        total_days += s['unique_days']
        report += f"| {i} | {r['secid']} | {r['type']} | {r['board']} | {s['total_candles']:,} | {s['unique_days']:,} | {s['avg_bars_per_day']} | {s['first_date']} | {s['last_date']} |\n"

    report += f"""
## Totals

- **Instruments downloaded:** {len([r for r in results if r])}
- **Total candles:** {total_candles:,}
- **Total unique days:** {total_days:,}

## Data Gaps > 7 Days

"""

    for r in results:
        if r is None:
            continue
        gaps = r['stats'].get('gaps', [])
        if gaps:
            report += f"### {r['secid']}\n\n"
            for gap in gaps:
                report += f"- {gap['from']} — {gap['to']} ({gap['days']} days)\n"
            report += "\n"

    no_gaps = [r['secid'] for r in results if r and not r['stats'].get('gaps')]
    if no_gaps:
        report += f"**No significant gaps:** {', '.join(no_gaps)}\n"

    report += f"""
## File Locations

- **Indices:** `moex_discovery/data/external/moex_iss/indices_10m/`
- **Currency:** `moex_discovery/data/external/moex_iss/currency_10m/`

## Notes

- All data downloaded for the FULL available period (no date truncation)
- Pagination handled automatically (500 candles per request)
- Rate limiting: 0.2 sec between requests
"""

    return report

def main():
    print("="*70)
    print("MOEX ISS EXTERNAL DATA DOWNLOAD")
    print("Indices and Currency - 10 minute candles")
    print("="*70)

    # Create directories
    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    CURRENCY_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for instrument in INSTRUMENTS:
        result = download_instrument(instrument)
        results.append(result)

    # Generate report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)

    report = generate_report(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved: {REPORT_PATH}")

    # Final summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)

    successful = [r for r in results if r]
    print(f"\nSuccessful: {len(successful)}/{len(INSTRUMENTS)}")

    for r in successful:
        print(f"  {r['secid']}: {r['stats']['total_candles']:,} candles, {r['stats']['first_date']} — {r['stats']['last_date']}")

if __name__ == "__main__":
    main()
