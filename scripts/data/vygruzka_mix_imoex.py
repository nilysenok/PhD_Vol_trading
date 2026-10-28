#!/usr/bin/env python3
"""Выгрузка ISS: дневной IMOEX и 20 контрактов MIX, 2015-2019 (Ж-094 п. 3). Только GET, последовательно.

Каждый ответ обрезается по TRADEDATE <= 2019-12-31 ДО записи; число выброшенных строк идёт в sha_journal.csv.
Лимит 80 запросов (превышение = ошибка), пауза не меньше 0,5 с. Пример: --out <каталог> [--dry-run]
Журнал: file,sha256,rows,dropped_ge2020,url. Самопроверка без аргументов: python3 vygruzka_mix_imoex.py
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

LAST_OK = '2019-12-31'
LIMIT, PAUSE, PAGE = 80, 0.5, 100
HIST = 'https://iss.moex.com/iss/history/engines/{e}/markets/{m}/securities/{s}.json'
QUERY = '?iss.only=history&iss.meta=off&from=2015-01-01&till=' + LAST_OK + '&start={k}'
# суффикс года обязателен у 2015-2018 и запрещён у 2019: иначе ISS отдаёт контракты 2025+
MIX = [f'MX{m}{y}_{2010 + y}' for y in (5, 6, 7, 8) for m in 'HMUZ'] + [f'MX{m}9' for m in 'HMUZ']
SOURCES = {'IMOEX': ('stock', 'index')} | {s: ('futures', 'forts') for s in MIX}


def build_url(secid, k=0):
    """URL истории; код вне списка (в т.ч. MXH9 с суффиксом или MXH5 без) отвергается до сети."""
    if secid not in SOURCES:
        raise ValueError(f'код {secid!r} вне списка запросов')
    e, m = SOURCES[secid]
    return HIST.format(e=e, m=m, s=secid) + QUERY.format(k=k)


def clip(cols, rows):
    """Оставляет строки с TRADEDATE <= LAST_OK; возвращает (оставшиеся, число выброшенных)."""
    i = cols.index('TRADEDATE')
    kept = [r for r in rows if r[i] <= LAST_OK]
    return kept, len(rows) - len(kept)


class Getter:
    """Сетевой GET с лимитом запросов и паузой; считает все обращения, включая повторы."""

    def __init__(self, limit=LIMIT, pause=PAUSE, opener=None):
        self.n, self.limit, self.pause, self.opener = 0, limit, pause, opener

    def __call__(self, url):
        if self.n >= self.limit:
            raise RuntimeError(f'лимит {self.limit} запросов исчерпан')
        self.n += 1
        if self.opener:
            d = self.opener(url)
        else:
            with urllib.request.urlopen(url, timeout=60) as r:
                d = json.loads(r.read().decode('utf-8'))
        time.sleep(self.pause)
        return d


def fetch(get, secid):
    """Все страницы одного кода; обрезка каждой страницы до накопления. -> (cols, rows, dropped, url0)."""
    cols, rows, dropped, k = None, [], 0, 0
    while True:
        h = get(build_url(secid, k))['history']
        cols = h['columns']
        kept, n = clip(cols, h['data'])
        rows += kept
        dropped += n
        if len(h['data']) < PAGE:
            return cols, rows, dropped, build_url(secid, 0)
        k += PAGE


def write(out, secid, cols, rows):
    i = cols.index('TRADEDATE')
    if rows and max(r[i] for r in rows) > LAST_OK:
        raise RuntimeError(f'{secid}: дата > {LAST_OK} перед записью')
    path = out / f'{secid}.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    return path


def run(out, get=None, secids=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    get = get or Getter()
    journal = []
    for s in secids or list(SOURCES):
        cols, rows, dropped, url = fetch(get, s)
        p = write(out, s, cols, rows)
        journal.append([p.name, hashlib.sha256(p.read_bytes()).hexdigest(), len(rows), dropped, url])
        print(f'{s}: {len(rows)} строк, dropped_ge2020={dropped}', flush=True)
    with open(out / 'sha_journal.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'sha256', 'rows', 'dropped_ge2020', 'url'])
        w.writerows(journal)
    return journal, get.n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if a.dry_run:
        print('\n'.join(build_url(s) for s in SOURCES))
        return 0
    return run(a.out)[1]


def _selftest():
    assert len(MIX) == 20 and 'MXH5_2015' in MIX and 'MXZ8_2018' in MIX and 'MXH9' in MIX
    for bad in ('MXH5', 'MXH9_2019', 'MXH5_2016'):
        try:
            build_url(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    cols = ['TRADEDATE']
    assert clip(cols, [['2019-12-31'], ['2020-01-03']]) == ([['2019-12-31']], 1)
    print('самопроверка ок')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
