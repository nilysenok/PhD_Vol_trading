#!/usr/bin/env python3
"""Доля ячеек признаков, изменившихся в features_pit_v3 против features_pit_v2, по рядам (29 внешних + собственные + «дни с
обновления»); изменение — неравенство значений (NaN против числа — изменение). Разбивка по причинам: строки до первого
наблюдения (до 26.08.2014 — в v3 NaN), строки вне общего календаря после его начала (выходные сессии), прочие (внутренние пропуски).
Выход: docs/otchety/m0/v3_vs_v2.csv. Запуск: python3 scripts/analysis/v3_vs_v2.py"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
OUT = Path(__file__).resolve().parents[2] / 'docs/otchety/m0/v3_vs_v2.csv'
INDICES = ["CNYRUB_TOM", "MOEXMM", "IMOEX", "MOEXTL", "MOEXEU", "RVI", "MOEXCH", "MOEXFN", "MOEXCN", "RGBI", "MOEXOG"]
YAHOO = ["brent", "vix", "sp500", "gold", "dxy", "eem", "wti", "tbill_3m", "treasury_10y", "treasury_5y"]
MACRO = ["cpi_ru", "cpi_us", "unemp_us", "m2_ru", "pmi_ru", "key_rate_cbr", "fed_rate", "si_fut"]


def series(c, own):
    if c in own:
        return 'собственные (23)'
    if c.endswith('_days_since_upd'):
        return 'дни с обновления (8)'
    for t in INDICES:
        if re.search(rf'(^|_){t}(_|$)', c):
            return f'индекс {t}'
    for y in YAHOO:
        if c == f'yahoo_{y}' or c.startswith(f'yahoo_{y}_') or c == y or c.startswith(y + '_'):
            return f'Yahoo {y}'
    for m in MACRO:
        if c == m or c.startswith(m + '_'):
            return f'макро {m}'
    return 'прочее'


def main():
    fc = json.load(open(R / 'features_pit_v3.config.json'))['feature_cols']
    assert fc == json.load(open(R / 'features_pit_v2.config.json'))['feature_cols']
    v2, v3 = pd.read_parquet(R / 'features_pit_v2.parquet'), pd.read_parquet(R / 'features_pit_v3.parquet')
    assert (v2[['date', 'ticker']].values == v3[['date', 'ticker']].values).all()
    own = fc[:23]
    d = pd.to_datetime(v2['date'])
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(
        '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/dataset_final/02_external/yahoo.parquet')['date']))
    pre = (d < cal.min()).values
    off = (~d.isin(cal) & ~pre).values
    rest = ~pre & ~off
    a, b = v2[fc].to_numpy(float), v3[fc].to_numpy(float)
    ch = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    grp = pd.Series([series(c, own) for c in fc])
    rows = []
    for g, idx in grp.groupby(grp).groups.items():
        cc = ch[:, list(idx)]
        rows.append({'ряд': g, 'признаков': len(idx), 'изменено_ячеек_%': round(100 * cc.mean(), 3),
                     'из них до первого наблюдения (до 26.08.2014), п.п.': round(100 * cc[pre].sum() / cc.size, 3),
                     'вне календаря (выходные сессии), п.п.': round(100 * cc[off].sum() / cc.size, 3),
                     'внутренние пропуски, п.п.': round(100 * cc[rest].sum() / cc.size, 3),
                     'NaN в v3_%': round(100 * np.isnan(b[:, list(idx)]).mean(), 3)})
    t = pd.DataFrame(rows).sort_values('ряд')
    tot = {'ряд': 'ВСЕГО', 'признаков': len(fc), 'изменено_ячеек_%': round(100 * ch.mean(), 3),
           'из них до первого наблюдения (до 26.08.2014), п.п.': round(100 * ch[pre].sum() / ch.size, 3),
           'вне календаря (выходные сессии), п.п.': round(100 * ch[off].sum() / ch.size, 3),
           'внутренние пропуски, п.п.': round(100 * ch[rest].sum() / ch.size, 3), 'NaN в v3_%': round(100 * np.isnan(b).mean(), 3)}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f'строк {len(v2)}; до 26.08.2014: {pre.sum()}; вне календаря: {off.sum()} ({sorted(set(d[off].dt.date.astype(str)))[:6]}...)')
    print('строк, где изменилась хотя бы одна ячейка:', int(ch.any(1).sum()))
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
