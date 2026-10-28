#!/usr/bin/env python3
"""etap5 а): масштаб сохранённых прогнозов — медиана прогноза / медиана RV по тикер × год × h × модель.

Запуск из корня репо: python3 scripts/analysis/etap5_masshtab.py
Выход: data/out/etap5_masshtab_all.csv (все ячейки), data/out/etap5_anomalii.csv (вне [0,5; 2]).
Сканирует все parquet в папках прогнозов, где есть rv_actual и колонки rv_pred / pred_*.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRS = [ROOT / 'moex_volatility/results/final/data/predictions_walkforward',
        ROOT / 'moex_volatility/data/predictions']
LO, HI = 0.5, 2.0
OUT = ROOT / 'data/out'


def horizon(name):
    m = re.search(r'_h(\d+)', name)
    return int(m.group(1)) if m else None


def scan(path):
    d = pd.read_parquet(path)
    if 'rv_actual' not in d.columns or 'ticker' not in d.columns:
        return []
    if 'year' not in d.columns:
        d['year'] = pd.to_datetime(d['date']).dt.year
    cols = [c for c in d.columns if c == 'rv_pred' or c.startswith('pred_')]
    rows = []
    for c in cols:
        g = d.groupby(['ticker', 'year']).agg(med_pred=(c, 'median'), med_rv=('rv_actual', 'median'),
                                              n=(c, 'size'))
        g = g.reset_index()
        g['ratio'] = g.med_pred / g.med_rv
        g['file'] = str(path.relative_to(ROOT))
        g['h'] = horizon(path.name)
        g['model'] = c if c != 'rv_pred' else path.name.split('_h')[0]
        rows.append(g)
    return rows


def main():
    parts = []
    for base in DIRS:
        for p in sorted(base.rglob('*.parquet')):
            try:
                parts += scan(p)
            except Exception as e:  # битый/чужой формат — в лог, не молча
                print(f'пропуск {p.relative_to(ROOT)}: {e}', file=sys.stderr)
    if not parts:
        sys.exit('ОШИБКА: не найдено ни одного файла прогнозов')
    allc = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    allc.to_csv(OUT / 'etap5_masshtab_all.csv', index=False)
    bad = allc[(allc.ratio < LO) | (allc.ratio > HI) | allc.ratio.isna()]
    bad.to_csv(OUT / 'etap5_anomalii.csv', index=False)
    print(f'ячеек {len(allc)}, файлов {allc.file.nunique()}, аномалий {len(bad)}')
    if len(bad):
        s = bad.groupby(['file', 'model', 'h']).agg(yachejki=('ratio', 'size'),
                                                     gody=('year', lambda x: ','.join(map(str, sorted(set(x))))),
                                                     ratio_min=('ratio', 'min'), ratio_max=('ratio', 'max'))
        print(s.to_string())


if __name__ == '__main__':
    main()
