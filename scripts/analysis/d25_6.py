#!/usr/bin/env python3
"""Д25.6 (26.10, ночь): наличие 1-минутных свечей ISS (interval=1) для 17 тикеров и оценка объёма выгрузки.

Только GET на iss.moex.com, последовательно, пауза PAUSE между запросами.
  python3 scripts/analysis/d25_6.py borders   -> candleborders interval 1 и 10 по 17 тикерам
  python3 scripts/analysis/d25_6.py sample    -> выборочные дни: число 1-мин баров, страниц, время ответа
Выход: data/regen/D25/iss_1min_borders.csv, iss_1min_sample.csv (абсолютный путь общего data/).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
OUT = DATA / 'regen' / 'D25'
TICKERS = ['SBER', 'LKOH', 'TATN', 'NVTK', 'MTLR', 'HYDR', 'IRAO', 'MTSS', 'MGNT', 'LSRG',
           'RTKM', 'VTBR', 'PHOR', 'ALRS', 'MOEX', 'AFLT', 'OGKB']
BASE = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}'
PAUSE = 0.5
# выборка: ранний обучающий год, середина, конец обучения шага 6, ликвидные и неликвидные бумаги
SAMPLE = [('SBER', '2014-06-10'), ('SBER', '2018-03-15'), ('LSRG', '2015-03-02'), ('OGKB', '2016-06-01'),
          ('MOEX', '2020-09-15'), ('AFLT', '2014-11-12'), ('PHOR', '2017-05-17'), ('MTLR', '2019-12-03')]


def get(url):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read().decode('utf-8'))
    time.sleep(PAUSE)
    return d, time.time() - t0


def borders():
    rows = []
    for t in TICKERS:
        d, _ = get(BASE.format(t=t) + '/candleborders.json?iss.meta=off')
        b = d['borders']
        for r in b['data']:
            rec = dict(zip(b['columns'], r))
            if rec['interval'] in (1, 10):
                rows.append({'ticker': t, 'interval': rec['interval'], 'begin': rec['begin'], 'end': rec['end']})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'iss_1min_borders.csv', index=False)
    print(df.to_string(index=False))


def sample():
    rows = []
    for t, day in SAMPLE:
        bars, pages, sec, start = [], 0, 0.0, 0
        while True:
            url = BASE.format(t=t) + f'/candles.json?interval=1&from={day}&till={day}&start={start}&iss.meta=off'
            d, dt = get(url)
            c = d['candles']
            bars += [dict(zip(c['columns'], r)) for r in c['data']]
            pages += 1
            sec += dt
            if len(c['data']) < 500:
                break
            start += 500
        ins = [b for b in bars if '09:50:00' <= b['begin'][11:19] < '18:50:00']
        rows.append({'ticker': t, 'day': day, 'bars_1m': len(bars), 'bars_1m_in_window': len(ins),
                     'first': bars[0]['begin'] if bars else None, 'last': bars[-1]['begin'] if bars else None,
                     'pages': pages, 'sec_with_pause': round(sec, 2)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'iss_1min_sample.csv', index=False)
    # оценка объёма: 17 тикеров x обучающие дни 2014-06..2020-12 (шаг 6 обучается до 2020 включительно)
    days = 17 * 1650
    bpd = df['bars_1m'].mean()
    pages = days * max(1.0, bpd / 500.0)
    per_page = df['sec_with_pause'].sum() / df['pages'].sum()
    print(f'оценка: {days} тикер-дней, ~{bpd:.0f} баров/день, ~{pages:.0f} страниц по 500 '
          f'(запрос с from..till на месяц), {per_page:.2f} с/страницу с паузой -> '
          f'{pages * per_page / 3600:.1f} ч последовательно, {pages * per_page / 3600 / 4:.1f} ч в 4 потока')


if __name__ == '__main__':
    {'borders': borders, 'sample': sample}[sys.argv[1]]()
