"""
Download Yahoo Finance data for global market instruments.

Instruments:
- BZ=F     : Brent Crude Oil Futures
- ^VIX     : CBOE Volatility Index
- ^GSPC    : S&P 500 Index
- GC=F     : Gold Futures
- DX-Y.NYB : US Dollar Index
- EEM      : iShares MSCI Emerging Markets ETF
- USDRUB=X : USD/RUB Exchange Rate
- EURRUB=X : EUR/RUB Exchange Rate
- CL=F     : WTI Crude Oil Futures
- ^IRX     : 13-Week Treasury Bill Rate
- ^TNX     : 10-Year Treasury Note Yield
- ^FVX     : 5-Year Treasury Note Yield
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path('/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery')
RAW_DIR = BASE_DIR / 'data/external/yahoo/raw'
FINAL_DIR = BASE_DIR / 'data/external/yahoo/final'
FINAL_V5_DIR = BASE_DIR / 'data/final_v5'
REPORT_PATH = BASE_DIR / 'reports/yahoo_download_report.md'

RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Instruments to download
INSTRUMENTS = {
    'BZ=F': 'Brent Crude Oil Futures',
    '^VIX': 'CBOE Volatility Index',
    '^GSPC': 'S&P 500 Index',
    'GC=F': 'Gold Futures',
    'DX-Y.NYB': 'US Dollar Index',
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'USDRUB=X': 'USD/RUB Exchange Rate',
    'EURRUB=X': 'EUR/RUB Exchange Rate',
    'CL=F': 'WTI Crude Oil Futures',
    '^IRX': '13-Week Treasury Bill Rate',
    '^TNX': '10-Year Treasury Note Yield',
    '^FVX': '5-Year Treasury Note Yield',
}

# Target period (same as final_v5)
START_DATE = '2014-08-26'
END_DATE = '2026-02-03'


def download_instrument(ticker: str, name: str):
    """Download daily data for one instrument."""
    print(f"  Downloading {ticker} ({name})...")

    try:
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            print(f"    WARNING: No data returned for {ticker}")
            return None

        # Reset index to make date a column
        data = data.reset_index()
        data['ticker'] = ticker
        data['name'] = name

        # Rename columns to lowercase
        data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]

        print(f"    OK: {len(data)} rows, {data['date'].min().date()} — {data['date'].max().date()}")
        return data

    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("=" * 60)
    print("Yahoo Finance Data Download")
    print("=" * 60)
    print(f"Target period: {START_DATE} — {END_DATE}")
    print(f"Instruments: {len(INSTRUMENTS)}")
    print()

    # Load common dates from final_v5
    common_dates_path = FINAL_V5_DIR / 'common_dates.csv'
    if common_dates_path.exists():
        common_dates = pd.read_csv(common_dates_path)
        common_dates['date'] = pd.to_datetime(common_dates['date'])
        common_dates_set = set(common_dates['date'].dt.date)
        print(f"Loaded {len(common_dates_set)} common dates from final_v5")
    else:
        common_dates_set = None
        print("WARNING: common_dates.csv not found")
    print()

    # Download all instruments
    print("Downloading instruments...")
    results = {}
    all_data = []

    for ticker, name in INSTRUMENTS.items():
        df = download_instrument(ticker, name)
        if df is not None:
            # Save raw
            safe_ticker = ticker.replace('^', '').replace('=', '_').replace('-', '_').replace('.', '_')
            raw_path = RAW_DIR / f'{safe_ticker}.parquet'
            df.to_parquet(raw_path, index=False)

            results[ticker] = {
                'name': name,
                'rows': len(df),
                'start': df['date'].min().date(),
                'end': df['date'].max().date(),
                'safe_ticker': safe_ticker
            }
            all_data.append(df)
        else:
            results[ticker] = None

    print()
    print("=" * 60)
    print("Processing and aligning data...")
    print("=" * 60)

    # Combine all data
    if not all_data:
        print("ERROR: No data downloaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date']).dt.date

    # Create wide format (close prices only)
    print("\nCreating wide format (close prices)...")

    wide_data = []
    stats = []

    for ticker, info in results.items():
        if info is None:
            continue

        safe_ticker = info['safe_ticker']
        df = pd.read_parquet(RAW_DIR / f'{safe_ticker}.parquet')
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Get close prices
        close_df = df[['date', 'close']].copy()
        close_df = close_df.rename(columns={'close': safe_ticker})
        close_df = close_df.drop_duplicates(subset='date', keep='last')

        wide_data.append(close_df.set_index('date'))

        # Calculate overlap with common_dates
        if common_dates_set:
            ticker_dates = set(df['date'])
            overlap = len(ticker_dates & common_dates_set)
            overlap_pct = overlap / len(common_dates_set) * 100
        else:
            overlap = 0
            overlap_pct = 0

        stats.append({
            'ticker': ticker,
            'safe_ticker': safe_ticker,
            'name': info['name'],
            'rows': info['rows'],
            'start': info['start'],
            'end': info['end'],
            'overlap': overlap,
            'overlap_pct': overlap_pct
        })

    # Merge all into wide format
    wide_df = wide_data[0]
    for df in wide_data[1:]:
        wide_df = wide_df.join(df, how='outer')

    wide_df = wide_df.reset_index()
    wide_df['date'] = pd.to_datetime(wide_df['date'])
    wide_df = wide_df.sort_values('date')

    print(f"Wide format: {len(wide_df)} rows × {len(wide_df.columns)} columns")

    # Filter to common_dates if available
    if common_dates_set:
        common_dates_dt = [pd.Timestamp(d) for d in common_dates_set]
        wide_aligned = wide_df[wide_df['date'].isin(common_dates_dt)].copy()
        print(f"Aligned to common_dates: {len(wide_aligned)} rows")

        # Forward-fill missing values within aligned data
        for col in wide_aligned.columns:
            if col != 'date':
                missing_before = wide_aligned[col].isna().sum()
                wide_aligned[col] = wide_aligned[col].ffill()
                missing_after = wide_aligned[col].isna().sum()
                if missing_before > missing_after:
                    print(f"  {col}: filled {missing_before - missing_after} missing values")
    else:
        wide_aligned = wide_df

    # Save final data
    print("\nSaving final data...")

    # Wide format
    wide_path = FINAL_DIR / 'yahoo_wide.parquet'
    wide_aligned.to_parquet(wide_path, index=False)
    print(f"  Saved: {wide_path}")

    # Long format
    long_df = wide_aligned.melt(id_vars=['date'], var_name='ticker', value_name='close')
    long_df = long_df.dropna(subset=['close'])
    long_path = FINAL_DIR / 'yahoo_long.parquet'
    long_df.to_parquet(long_path, index=False)
    print(f"  Saved: {long_path}")

    # Generate report
    print("\nGenerating report...")

    stats_df = pd.DataFrame(stats)

    report = f"""# Yahoo Finance Download Report

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Параметры

- Период: {START_DATE} — {END_DATE}
- Инструменты: {len(INSTRUMENTS)}
- Успешно загружено: {len(stats)}

## Результаты загрузки

| Ticker | Инструмент | Строк | Период | Overlap с MOEX |
|--------|------------|-------|--------|----------------|
"""

    for _, row in stats_df.iterrows():
        report += f"| {row['ticker']} | {row['name'][:30]} | {row['rows']} | {row['start']} — {row['end']} | {row['overlap_pct']:.1f}% ({row['overlap']}) |\n"

    report += f"""
## Итоговые файлы

| Файл | Описание |
|------|----------|
| `yahoo_wide.parquet` | Wide format, {len(wide_aligned)} rows × {len(wide_aligned.columns)-1} tickers |
| `yahoo_long.parquet` | Long format, {len(long_df)} records |

## Структура данных

### yahoo_wide.parquet
- Столбцы: date, {', '.join([s['safe_ticker'] for s in stats][:6])}...
- Строки: {len(wide_aligned)} (aligned to common_dates)

### yahoo_long.parquet
- Столбцы: date, ticker, close
- Строки: {len(long_df)}

## Покрытие

"""

    # Calculate coverage stats
    for col in wide_aligned.columns:
        if col != 'date':
            non_null = wide_aligned[col].notna().sum()
            pct = non_null / len(wide_aligned) * 100
            report += f"- **{col}**: {non_null}/{len(wide_aligned)} ({pct:.1f}%)\n"

    report += """
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
"""

    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"  Report: {REPORT_PATH}")

    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Total instruments: {len(stats)}")
    print(f"Wide format rows: {len(wide_aligned)}")
    print(f"Long format records: {len(long_df)}")


if __name__ == '__main__':
    main()
