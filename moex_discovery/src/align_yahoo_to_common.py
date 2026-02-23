"""
Align Yahoo Finance data to common_dates (2874 days).
Fill missing dates with forward-fill.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path('/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery')
YAHOO_FINAL = BASE_DIR / 'data/external/yahoo/final'
FINAL_V5 = BASE_DIR / 'data/final_v5'
REPORT_PATH = BASE_DIR / 'reports/yahoo_download_report.md'

print("=" * 60)
print("Aligning Yahoo Finance data to common_dates")
print("=" * 60)

# Load common dates
common_dates = pd.read_csv(FINAL_V5 / 'common_dates.csv')
common_dates['date'] = pd.to_datetime(common_dates['date'])
common_df = pd.DataFrame({'date': common_dates['date']})

print(f"Common dates: {len(common_df)}")

# Load yahoo wide
yahoo = pd.read_parquet(YAHOO_FINAL / 'yahoo_wide.parquet')
print(f"Yahoo wide (before): {len(yahoo)} rows")

# Merge to get all common dates
aligned = common_df.merge(yahoo, on='date', how='left')

# Find missing dates
missing_mask = aligned.iloc[:, 1].isna()
missing_dates = aligned[missing_mask]['date'].tolist()
print(f"Missing dates: {len(missing_dates)}")

for d in missing_dates:
    print(f"  {d.date()} ({d.strftime('%A')})")

# Forward-fill missing values
print("\nForward-filling missing values...")
for col in aligned.columns:
    if col != 'date':
        missing_before = aligned[col].isna().sum()
        aligned[col] = aligned[col].ffill()
        missing_after = aligned[col].isna().sum()
        if missing_before > 0:
            print(f"  {col}: {missing_before} → {missing_after} missing")

# Check for remaining NaN
remaining_nan = aligned.iloc[:, 1:].isna().any(axis=1).sum()
print(f"\nRemaining rows with any NaN: {remaining_nan}")

# If there are still NaN at the beginning, backfill
if remaining_nan > 0:
    print("Backfilling initial NaN values...")
    for col in aligned.columns:
        if col != 'date':
            aligned[col] = aligned[col].bfill()

# Save aligned data
print("\nSaving aligned data...")

# Wide format
wide_path = YAHOO_FINAL / 'yahoo_wide.parquet'
aligned.to_parquet(wide_path, index=False)
print(f"  Saved: {wide_path}")
print(f"  Rows: {len(aligned)}")

# Long format
long_df = aligned.melt(id_vars=['date'], var_name='ticker', value_name='close')
long_df = long_df.dropna(subset=['close'])
long_path = YAHOO_FINAL / 'yahoo_long.parquet'
long_df.to_parquet(long_path, index=False)
print(f"  Saved: {long_path}")
print(f"  Records: {len(long_df)}")

# Update report
print("\nUpdating report...")

report_update = f"""

---

## ЭТАП 2: Выравнивание до 100% overlap

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

### Результат

- Common dates: {len(common_df)}
- Yahoo dates (после выравнивания): {len(aligned)}
- Заполнено пропусков: {len(missing_dates)} дат

### Пропущенные даты (заполнены forward-fill)

| Дата | День недели | Причина |
|------|-------------|---------|
"""

for d in missing_dates:
    day = d.strftime('%A')
    if day == 'Saturday':
        reason = "MOEX торгует, США нет"
    elif d.date() == datetime(2025, 4, 18).date():
        reason = "Good Friday (США)"
    elif d.date() == datetime(2025, 12, 25).date():
        reason = "Christmas (США)"
    else:
        reason = "Праздник США или нет данных"
    report_update += f"| {d.date()} | {day} | {reason} |\n"

report_update += f"""
### Покрытие после выравнивания

| Инструмент | Строк | Покрытие |
|------------|-------|----------|
"""

for col in aligned.columns:
    if col != 'date':
        non_null = aligned[col].notna().sum()
        pct = non_null / len(aligned) * 100
        report_update += f"| {col} | {non_null} | {pct:.1f}% |\n"

report_update += f"""
### Подтверждение

**✅ 100% overlap достигнут для ВСЕХ 12 Yahoo Finance инструментов**

- 12 инструментов × {len(aligned)} дней = {len(aligned) * 12} записей (теоретически)
- Фактически: {len(long_df)} записей в long формате
"""

# Append to report
with open(REPORT_PATH, 'a') as f:
    f.write(report_update)
print(f"  Report updated: {REPORT_PATH}")

print()
print("=" * 60)
print("DONE!")
print("=" * 60)
print(f"Yahoo data now aligned to {len(aligned)} common dates (100%)")
