"""Проверка п. 4: бары бэктеста (ohlcv_daily_full) против данных главы 3 (features_pit_v3), таблица 8.1."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

M = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
OH = M / 'moex_volatility/results/final/strategies/data/ohlcv_daily_full.parquet'
FP = M / 'data/regen/features_pit_v3.parquet'
SG = M / 'data/regen/sigma_pred_h1_final3_full.parquet'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


o = pd.read_parquet(OH)
o['date'] = pd.to_datetime(o['date'])
f = pd.read_parquet(FP, columns=['date', 'ticker', 'close', 'volume', 'n_bars', 'source'])
f['date'] = pd.to_datetime(f['date'])
s = pd.read_parquet(SG)
print('SHA ohlcv', sha(OH), '\nSHA features_pit_v3', sha(FP))
print('ohlcv строк', len(o), o.date.min().date(), o.date.max().date(), 'тикеров', o.ticker.nunique())
print('ohlcv NaN по столбцам', o.isna().sum().to_dict(), 'дубли (ticker,date)', int(o.duplicated(['ticker', 'date']).sum()))
print('нулевой/отрицательный close', int((o.close <= 0).sum()), 'high<low', int((o.high < o.low).sum()),
      'нулевой объём', int((o.volume <= 0).sum()))
m = o.merge(f, on=['ticker', 'date'], how='outer', indicator=True, suffixes=('', '_ch3'))
print('ohlcv vs features_pit_v3 (ticker,date):', m._merge.value_counts().to_dict())
b = m[m._merge == 'both']
dc = np.abs(b.close - b.close_ch3)
print('close: равно точно', int((b.close == b.close_ch3).sum()), 'из', len(b), '; max|Δ|', float(dc.max()),
      '; отн. max', float((dc / b.close).max()))
print('volume: равно', int((b.volume == b.volume_ch3).sum()), 'из', len(b))
if (dc > 0).any():
    print(b[dc > 0][['ticker', 'date', 'close', 'close_ch3']].head(10).to_string())
print('source в features:', f.source.value_counts().to_dict())

# Таблица 8.1 (2017–2026)
o17 = o[o.date >= '2017-01-01']
rows = []
for y, g in o17.groupby(o17.date.dt.year):
    per = g.groupby('ticker').size()
    sy = s[s.date.dt.year == y]
    rows.append(dict(year=y, dates=g.date.nunique(), pairs=len(g), min_t=int(per.min()), max_t=int(per.max()),
                     weekend=int(g.drop_duplicates('date').date.dt.dayofweek.ge(5).sum()),
                     sigma_dates=sy.date.nunique(), sigma_rows=len(sy)))
print(pd.DataFrame(rows).to_string(index=False))

# Пары без σ и σ без бара (2017–2026)
ms = o17.merge(s, on=['ticker', 'date'], how='outer', indicator=True)
print('бары×σ:', ms._merge.value_counts().to_dict())
print('бар без σ:\n', ms[ms._merge == 'left_only'][['ticker', 'date']].to_string())

# Разрыв 2022
d22 = np.sort(o[(o.date >= '2022-02-15') & (o.date <= '2022-04-10')].date.unique())
gaps = [(pd.Timestamp(a).date(), pd.Timestamp(b).date(), (pd.Timestamp(b) - pd.Timestamp(a)).days)
        for a, b in zip(d22[:-1], d22[1:]) if (pd.Timestamp(b) - pd.Timestamp(a)).days > 4]
print('разрывы > 4 дней (фев–апр 2022):', gaps)
w = o[(o.date >= '2022-02-28') & (o.date <= '2022-03-23')]
print('баров 28.02–23.03.2022:', len(w))
first_after = o[o.date >= '2022-03-24'].groupby('ticker').date.min()
print('первый бар после разрыва по тикерам:', {k: str(v.date()) for k, v in first_after.items()})
last_before = o[o.date <= '2022-02-27'].groupby('ticker').date.max()
print('последний бар до разрыва:', sorted(set(str(v.date()) for v in last_before)))
# Все разрывы > 5 календарных дней 2017–2026 по тикерам
for t, g in o17.groupby('ticker'):
    dd = g.date.sort_values().diff().dt.days
    big = g.date.sort_values()[dd > 5]
    if len(big):
        print('  разрыв > 5 дн.', t, [(str((x - pd.Timedelta(days=int(k))).date()), str(x.date())) for x, k in zip(big, dd[dd > 5])])
