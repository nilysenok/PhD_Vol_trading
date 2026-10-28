#!/usr/bin/env python3
"""Отбор тикеров 20 -> 17 (раздел 2.1.1): метрики качества 10-минутных данных по 20 кандидатам.

Запуск из корня репо: python3 scripts/analysis/otbor_metriki.py
Выход: data/out/otbor_metriki_20.csv. Основная сессия 10:00-18:40, 52 бара в полной сессии,
календарь = дни с барами не менее чем у 10 из 20 тикеров, начиная с 09.06.2014.
Доля пропущенных интервалов: 1 - (бары, обрезанные сверху 52 на день) / (52 * дней календаря).
Дни RV и доля карантина - из moex_discovery/reports (TICKER_SELECTION.md, d002_summary.csv).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MD = ROOT / 'moex_discovery'
T20 = ('SBER LKOH TATN NVTK VTBR ALRS AFLT HYDR MGNT MOEX RTKM MTSS MTLR IRAO OGKB PHOR '
       'MSNG TRMK LSRG TGKA').split()
EXCLUDED = {'TGKA', 'MSNG', 'TRMK'}
RV_DAYS = {'SBER': 2990, 'LKOH': 2990, 'TATN': 2989, 'NVTK': 2986, 'VTBR': 2986, 'ALRS': 2990,
           'AFLT': 2991, 'HYDR': 2991, 'MGNT': 2991, 'MOEX': 2991, 'RTKM': 2990, 'MTSS': 2980,
           'MTLR': 2991, 'IRAO': 2964, 'OGKB': 2838, 'PHOR': 2656, 'MSNG': 2558, 'TRMK': 2527,
           'LSRG': 2344, 'TGKA': 2315}  # TICKER_SELECTION.md, таблица финальной выборки


def bars_per_day(t):
    x = pd.read_parquet(MD / f'data/clean/candles_10m/{t}.parquet')
    ts = pd.to_datetime(x['timestamp'])
    tm = ts.dt.time
    ts = ts[(tm >= pd.Timestamp('10:00').time()) & (tm <= pd.Timestamp('18:40').time())]
    if ts.empty:
        raise SystemExit(f'ОШИБКА: нет баров основной сессии у {t}')
    return ts.dt.normalize().value_counts()


def main():
    per = {t: bars_per_day(t) for t in T20}
    cnt = pd.concat([s.rename(t) > 0 for t, s in per.items()], axis=1).fillna(False).sum(axis=1)
    cal = cnt[(cnt >= 10) & (cnt.index >= pd.Timestamp('2014-06-09'))].index.sort_values()
    rows = []
    for t in T20:
        b = per[t].reindex(cal).fillna(0)
        rows.append({
            'тикер': t, 'в_17': t not in EXCLUDED,
            'дней_без_баров_%': (b == 0).mean() * 100,
            'неполных_сессий_%': (b < 52).mean() * 100,
            'пропущено_интервалов_%': (1 - b.clip(upper=52).sum() / (52 * len(cal))) * 100,
            'дней_RV': RV_DAYS[t],
        })
    d = pd.DataFrame(rows).set_index('тикер')
    q = pd.read_csv(MD / 'reports/d002_summary.csv').set_index('ticker')['pct_quarantine']
    d['карантин_%'] = q.reindex(d.index)
    d = d.sort_values('пропущено_интервалов_%', ascending=False)
    out = ROOT / 'data/out'
    out.mkdir(parents=True, exist_ok=True)
    d.round(2).to_csv(out / 'otbor_metriki_20.csv')
    print(f'дней календаря: {len(cal)}')
    print(d.round(2).to_string())


if __name__ == '__main__':
    main()
