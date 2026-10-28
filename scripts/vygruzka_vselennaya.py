#!/usr/bin/env python3
"""Выгрузка ISS для расширенной вселенной (рычаг (а), Ж-069 п. 3): 1-мин, 10-мин свечи и дневной VALUE, период только < 2022.

Образец — scripts/analysis/d25_6.py. Только GET на iss.moex.com, последовательно, пауза после каждого запроса.
  python3 scripts/vygruzka_vselennaya.py --tickers SBER,LKOH --start 2015-01-01 --end 2021-12-31 --dry-run
Выход (новый каталог data/regen/V40/, существующие каталоги — D25 и moex_discovery — не трогаются):
  bars1m/{T}_{Y}.csv + .done           — формат data/regen/D25/bars1m (begin, open, high, low, close, volume, value)
  candles_10m/{T}.parquet + .done      — формат moex_discovery/data/raw/candles_10m (timestamp, open, high, low, close, volume)
  daily_value/{T}.csv + .done          — date, value (оборот за день, interval=24)
  sha_journal.csv                      — SHA-256 каждого записанного файла (только дополняется)
Выгрузку запускает дирижёр; тесты ходят в фикстуру, не в сеть.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'
ROOT = DATA / 'regen' / 'V40'
LAST_OK = pd.Timestamp('2021-12-31')
BASE = 'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{t}'
PAGE = 500
PAUSE = 0.2
COLS = ['begin', 'open', 'high', 'low', 'close', 'volume', 'value']
TYPES = {'1m': 1, '10m': 10, 'daily': 24}


def check_end(end):
    """Отказ до сети: end обязателен и не позже 2021-12-31."""
    e = pd.Timestamp(end)
    if e > LAST_OK:
        raise ValueError(f'end={end} позже {LAST_OK.date()}: данные >= 2022 запрещены')
    return e


def check_out(out, root=ROOT):
    """Писать можно только внутрь каталога V40 (не D25, не moex_discovery)."""
    out, root = Path(out).resolve(), Path(root).resolve()
    if root.name != 'V40' or (out != root and root not in out.parents):
        raise ValueError(f'запись вне {root}: {out}')
    return out


def make_get(tries=6):
    def get(url):
        last = None
        for k in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.loads(r.read().decode('utf-8'))
                time.sleep(PAUSE)
                return d
            except Exception as e:  # обрыв/429/5xx: пауза с ростом, затем повтор
                last = e
                time.sleep(min(60, 2 ** k * 2))
        raise RuntimeError(f'GET не удался после {tries} попыток: {url}: {last}')
    return get


def paginate(get, url):
    """Страницы по PAGE подряд до страницы короче PAGE; возвращает список dict."""
    rows, start = [], 0
    while True:
        c = get(f'{url}&start={start}')['candles']
        rows += [dict(zip(c['columns'], r)) for r in c['data']]
        if len(c['data']) < PAGE:
            return rows
        start += PAGE


def fetch(get, t, interval, fr, till):
    return paginate(get, BASE.format(t=t) + f'/candles.json?interval={interval}&from={fr}&till={till}&iss.meta=off')


def months(fr, till):
    """Помесячные отрезки [fr; till] (годовой запрос ISS у d25_6 обрывался в середине года)."""
    a = pd.Timestamp(fr)
    while a <= till:
        b = min(a + pd.offsets.MonthEnd(0), till)
        yield a.strftime('%Y-%m-%d'), b.strftime('%Y-%m-%d')
        a = b + pd.Timedelta(days=1)


def year_spans(start, end):
    for y in range(start.year, end.year + 1):
        yield y, max(start, pd.Timestamp(y, 1, 1)), min(end, pd.Timestamp(y, 12, 31))


def sha_log(out, path, rows):
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    j = out / 'sha_journal.csv'
    with open(j, 'a') as f:
        if f.tell() == 0:
            f.write('file,sha256,rows,time\n')
        f.write(f'{Path(path).relative_to(out)},{h},{rows},{time.strftime("%F %T")}\n')


def finish(out, path, rows, key):
    sha_log(out, path, rows)
    path.with_suffix('.done').write_text(json.dumps({'rows': rows, 'key': key, 'time': time.strftime('%F %T')}))


def done_ok(path, key):
    d = path.with_suffix('.done')
    return d.exists() and path.exists() and json.loads(d.read_text()).get('key') == key


def job_1m(get, out, t, y, fr, till):
    path, key = out / 'bars1m' / f'{t}_{y}.csv', f'{fr.date()}..{till.date()}'
    if done_ok(path, key):
        return 0
    rows = [r for a, b in months(fr, till) for r in fetch(get, t, 1, a, b)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    finish(out, path, len(rows), key)
    return len(rows)


def job_10m(get, out, t, start, end):
    path, key = out / 'candles_10m' / f'{t}.parquet', f'{start.date()}..{end.date()}'
    if done_ok(path, key):
        return 0
    rows = [r for _, f, tl in year_spans(start, end) for a, b in months(f, tl) for r in fetch(get, t, 10, a, b)]
    df = pd.DataFrame(rows, columns=COLS).rename(columns={'begin': 'timestamp'}).drop(columns='value')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    finish(out, path, len(df), key)
    return len(df)


def job_daily(get, out, t, start, end):
    path, key = out / 'daily_value' / f'{t}.csv', f'{start.date()}..{end.date()}'
    if done_ok(path, key):
        return 0
    rows = [r for _, f, tl in year_spans(start, end) for r in fetch(get, t, 24, f.date(), tl.date())]
    df = pd.DataFrame({'date': [r['begin'][:10] for r in rows], 'value': [r['value'] for r in rows]})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    finish(out, path, len(df), key)
    return len(df)


def run(tickers, start, end, types, out=ROOT, get=None, dry_run=False, root=ROOT):
    """Все проверки (end, каталог, тикеры) — до первого GET."""
    end = check_end(end)
    start = pd.Timestamp(start)
    if start > end:
        raise ValueError('start позже end')
    out = check_out(out, root)
    for t in tickers:
        if not re.fullmatch(r'[A-Z0-9]+', t):
            raise ValueError(f'тикер {t!r}')
    if dry_run:
        for t in tickers:
            for k in types:
                print(f'{t}\t{start.date()}..{end.date()}\t{k}')
        print(f'итого: {len(tickers)} тикеров x {len(types)} типов = {len(tickers) * len(types)} заданий; сеть не использована')
        return 0
    get = get or make_get()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in tickers:
        if '1m' in types:
            n += sum(job_1m(get, out, t, y, f, tl) for y, f, tl in year_spans(start, end))
        if '10m' in types:
            n += job_10m(get, out, t, start, end)
        if 'daily' in types:
            n += job_daily(get, out, t, start, end)
        print(f'{time.strftime("%T")} {t}: готово', flush=True)
    return n


def main(argv=None, get=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tickers', required=True, help='через запятую, например SBER,LKOH')
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', required=True, help='не позже 2021-12-31')
    ap.add_argument('--types', default='1m,10m,daily')
    ap.add_argument('--out', default=str(ROOT))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    types = a.types.split(',')
    if any(k not in TYPES for k in types):
        ap.error(f'типы: {sorted(TYPES)}')
    return run(a.tickers.split(','), a.start, a.end, types, Path(a.out), get, a.dry_run)


if __name__ == '__main__':
    try:
        main()
    except ValueError as e:
        sys.exit(f'ОТКАЗ: {e}')
