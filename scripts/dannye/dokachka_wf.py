#!/usr/bin/env python3
"""Докачка хвоста 02.2026 для счёта wf (лист wf-L3): тем же форматом, что исходные загрузчики, ничего после KONEC.

Источники формата: moex_discovery/src/download_candles.py (акции TQBR, 10 мин, колонки timestamp,open,high,low,close,volume),
download_external_01.py (индексы/валюта, 10 мин, begin,end,open,close,high,low,value,volume,date),
scripts/analysis/d25_6.py (1-минутки CSV begin,open,high,low,close,volume,value + {T}_{Y}.done).
Только GET к iss.moex.com; пауза PAUSE на запрос (потоков не больше THREADS). Пишет в --out, исходники только читает.
  python3 scripts/dannye/dokachka_wf.py {akcii|indeksy|minutki} --root <основная копия> --out <папка>
Самопроверка без сети: python3 scripts/dannye/dokachka_wf.py
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

KONEC = pd.Timestamp('2026-02-28')   # включительно по дате begin; позже ничего не пишем
PAUSE = float(os.environ.get('DOKACHKA_PAUZA', '1.0'))                         # с после каждого запроса; запускать не более 2 процессов -> не более 2 запросов/с суммарно
BASE = 'https://iss.moex.com/iss/engines/'
TQBR = BASE + 'stock/markets/shares/boards/TQBR/securities/'
IDX = BASE + 'stock/markets/index/boards/{b}/securities/'
CUR = BASE + 'currency/markets/selt/boards/CETS/securities/'
INDEKSY = {'IMOEX': 'SNDX', 'MOEXOG': 'SNDX', 'MOEXFN': 'SNDX', 'MOEXMM': 'SNDX', 'MOEXEU': 'SNDX', 'MOEXTL': 'SNDX',
           'MOEXCN': 'SNDX', 'MOEXCH': 'SNDX', 'RGBI': 'SNDX', 'RVI': 'RTSI'}


def get_json(url, tries=6):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read().decode('utf-8'))
            time.sleep(PAUSE)
            return d
        except Exception as e:  # обрыв/429/5xx: пауза с ростом, повтор
            last = e
            time.sleep(min(60, 2 ** k * 2))
    raise RuntimeError(f'GET не удался: {url}: {last}')


def stranicy(base, secid, interval, frm, till, get=get_json):
    """Все свечи страницами по 500; страница < 500 — конец. Возвращает DataFrame как в ответе ISS."""
    rows, cols, start = [], None, 0
    while True:
        d = get(f'{base}{secid}/candles.json?interval={interval}&from={frm}&till={till}&start={start}&iss.meta=off')['candles']
        cols = d['columns']
        rows += d['data']
        if len(d['data']) < 500:
            return pd.DataFrame(rows, columns=cols)
        start += 500


def obrezat(df, col, konec=KONEC):
    """Строки с датой col <= konec (включительно). Мутация «не обрезает» ловится test_dokachka_wf."""
    return df[pd.to_datetime(df[col]).dt.normalize() <= konec].reset_index(drop=True)


def privesti(df, old):
    """Типы колонок как у old. Если ISS отдал null в целочисленной колонке (CNYRUB_TOM: value/volume с 2025-12 пусты) —
    тип Int64 с пропусками: значение не придумывается (Д4), в файле тот же arrow-тип int64."""
    t = {c: ('Int64' if old[c].dtype.kind in 'iu' and df[c].isna().any() else old[c].dtype)
         for c in df.columns if c != 'date' and c in old.columns}
    return df.astype(t)


def dopisat(old, new, col, konec=KONEC):
    """old + хвост, дата <= konec. Всё до последнего дня old — как было, никогда не заменяется.
    Последний день old сверяется с ISS (колонки времени и цен; value/volume не сравниваются: у CNY ISS их больше не отдаёт):
    если неполон или расходится (обрыв прошлой выгрузки) — заменяется данными ISS целиком, иначе не трогается; дальше — хвост.
    Возвращает (итог, заменено): число строк последнего дня old, замененных данными ISS."""
    new = obrezat(new[old.columns.intersection(new.columns)], col, konec)
    k_old, k_new = pd.to_datetime(old[col]), pd.to_datetime(new[col])
    day = k_old.max().normalize()
    last_old, last_new = old[k_old.dt.normalize() == day], new[k_new.dt.normalize() == day]
    cc = [c for c in ('timestamp', 'begin', 'end', 'open', 'close', 'high', 'low') if c in old.columns]
    cmp = last_old[cc].merge(privesti(last_new, old)[cc], how='outer', indicator=True)
    zamena = len(last_new) > 0 and bool((cmp['_merge'] != 'both').any())
    base = old[k_old.dt.normalize() < day] if zamena else old
    tail = new[k_new.dt.normalize() >= day] if zamena else new[k_new > k_old.max()]
    return pd.concat([base, privesti(tail, old)[old.columns]], ignore_index=True), (len(last_old) if zamena else 0)


def akcii(root, out, tikery=None):
    for p in sorted((root / 'moex_discovery/data/raw/candles_10m').glob('*.parquet')):
        if tikery and p.stem not in tikery:
            continue
        old = pd.read_parquet(p)
        frm = old['timestamp'].max().strftime('%Y-%m-%d')
        raw = stranicy(TQBR, p.stem, 10, frm, KONEC.strftime('%Y-%m-%d'))
        new = raw.rename(columns={'begin': 'timestamp'})[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        new['timestamp'] = pd.to_datetime(new['timestamp'])
        res, bad = dopisat(old, new, 'timestamp')
        o = out / 'moex_discovery/data/raw/candles_10m' / p.name
        o.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(o, index=False)
        print(p.stem, len(old), len(res), 'заменено строк последнего дня', bad, flush=True)


def indeksy(root, out):
    """moex_iss/{indices,currency}_10m (с колонкой date) + final_v6/external и external/final (те же строки без date, с 2014-08-26)."""
    d = root / 'moex_discovery/data'
    spisok = [('external/moex_iss/indices_10m', n, IDX.format(b=b)) for n, b in INDEKSY.items()]
    spisok.append(('external/moex_iss/currency_10m', 'CNYRUB_TOM', CUR))
    for sub, n, base in spisok:
        old = pd.read_parquet(d / sub / f'{n}.parquet')
        frm = pd.to_datetime(old['begin']).max().strftime('%Y-%m-%d')
        new = stranicy(base, n, 10, frm, KONEC.strftime('%Y-%m-%d'))
        new = new[['begin', 'end', 'open', 'close', 'high', 'low', 'value', 'volume']]
        new['date'] = pd.to_datetime(new['begin']).dt.date
        res, bad = dopisat(old, new, 'begin')
        o = out / 'moex_discovery/data' / sub / f'{n}.parquet'
        o.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(o, index=False)
        print(n, len(old), len(res), 'заменено строк последнего дня', bad, flush=True)
        # копии final_v6/external берут свежий ответ ISS целиком (частичный последний бар старого файла — на день докачки — не тянут)
        cut = obrezat(new, 'begin')
        fresh = pd.concat([old[pd.to_datetime(old['begin']) < pd.to_datetime(cut['begin']).min()], privesti(cut, old)], ignore_index=True)
        kopiya(d, out / 'moex_discovery/data', n, fresh)


def kopiya(d, out, n, fresh):
    """final_v6/candles_10m/external/{n} и external/final/candles_10m/{n} (то же без колонки date): старый файл + строки fresh с датой > его последней."""
    for sub in ('final_v6/candles_10m/external', 'external/final/candles_10m'):
        p = d / sub / f'{n}.parquet'
        if not p.exists():
            continue
        old = pd.read_parquet(p)
        last = pd.to_datetime(old['begin']).max().normalize()
        add = fresh[pd.to_datetime(fresh['begin']).dt.normalize() > last][old.columns]
        o = out / sub / f'{n}.parquet'
        o.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([old, privesti(add, old)], ignore_index=True).to_parquet(o, index=False)


COLS1 = ['begin', 'open', 'high', 'low', 'close', 'volume', 'value']


def zapis_csv(res, f, szhat):
    """CSV как pandas.to_csv(index=False); szhat -> {f}.gz (gzip без метки времени, диск был полон). Возвращает sha256 несжатых байт."""
    b = res.to_csv(index=False).encode()
    if szhat:
        f.with_name(f.name + '.gz').write_bytes(gzip.compress(b, 9, mtime=0))
    else:
        f.write_bytes(b)
    return hashlib.sha256(b).hexdigest()


def minutki(root, out, god, tikery, szhat=False):
    """{T}_{god}.csv (+ .done) как d25_6._chunk: помесячно, без дедупликации; для 2026 — по KONEC."""
    o = out / 'data/regen/D25/bars1m'
    o.mkdir(parents=True, exist_ok=True)
    for t in tikery:
        f, done = o / f'{t}_{god}.csv', o / f'{t}_{god}.done'
        if done.exists():
            continue
        rows, pages = [], 0
        for mo in range(1, 13):
            fr = pd.Timestamp(year=god, month=mo, day=1)
            if fr > KONEC:
                break
            till = min(fr + pd.offsets.MonthEnd(0), KONEC)
            df = stranicy(TQBR, t, 1, fr.strftime('%Y-%m-%d'), till.strftime('%Y-%m-%d'))
            rows.append(obrezat(df[COLS1], 'begin') if len(df) else df[COLS1])
            pages += max(1, -(-len(df) // 500))
        res = pd.concat(rows, ignore_index=True)
        sha = zapis_csv(res, f, szhat)
        done.write_text(json.dumps({'rows': len(res), 'pages': pages, 'time': time.strftime('%F %T'), 'sha256_csv': sha}))
        print(t, god, len(res), flush=True)


def main():
    if len(sys.argv) == 1:  # самопроверка без сети; полные тесты — tests/test_dokachka_wf.py
        assert len(obrezat(pd.DataFrame({'begin': ['2026-02-27 10:00:00', '2026-03-01 10:00:00']}), 'begin')) == 1
        return print('самопроверка ok')
    ap = argparse.ArgumentParser()
    ap.add_argument('rezhim', choices=['akcii', 'indeksy', 'minutki'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--god', type=int)
    ap.add_argument('--tikery', default='')
    ap.add_argument('--szhat', action='store_true', help='1-минутки писать как .csv.gz')
    a = ap.parse_args()
    tk = [t for t in a.tikery.split(',') if t]
    if a.rezhim == 'akcii':
        akcii(a.root, a.out, tk)
    elif a.rezhim == 'indeksy':
        indeksy(a.root, a.out)
    else:
        minutki(a.root, a.out, a.god, tk, a.szhat)


if __name__ == '__main__':
    main()
