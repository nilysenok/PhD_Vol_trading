"""Переформатирование таблиц единственного запуска теста стратегий главы 4 (scripts/ch4_test_run.sh, 26.10 10:32–10:38,
коммит 25b76aa) в формат владельца: разделитель ';', десятичная запятая, UTF-8 с BOM. БЕЗ ПЕРЕСЧЁТА: работает со строками
ячеек, числа не парсятся и не округляются; в числовой ячейке меняется только десятичная точка на запятую
(-0.4202822138076673 -> -0,4202822138076673). Нечисловые ячейки ('inf', 'True', '2022–2026', текст выбора, пустые = NaN)
не трогаются. Образец — scripts/analysis/tables_test_format.py (тест главы 3).

Переформатируются только 17 таблиц docs/tables/strat_test/tabl_4_*.csv, не соответствующих формату (15 — ',' и точка,
2 — ';' и запятая, но без BOM). 12 таблиц s9_*_final3_test.csv уже в формате (';', запятая, BOM) и не трогаются.

Исходник <имя>.csv переименовывается в <имя>_raw.csv (байты не меняются), на прежнем имени пишется переформатированная
таблица. Проверка после записи: (1) _raw побайтно равен файлу коммита 25b76aa (git show); (2) таблицы совпадают по числу
строк и столбцов; (3) каждая ячейка: нормализованная строка (',' -> '.' в числовой ячейке) совпадает с исходной
побайтно, нечисловая ячейка совпадает как есть; (4) в новой таблице нет числовых ячеек с точкой.

python3 scripts/analysis/tables_strat_test_format.py            # переименовать, переформатировать, проверить
python3 scripts/analysis/tables_strat_test_format.py --check    # только проверка уже переформатированных
"""
import argparse
import csv
import hashlib
import io
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'docs/tables/strat_test'
COMMIT = '25b76aa'
# 17 таблиц запуска 26.10 10:32 не в формате владельца (проверено по байтам файлов коммита 25b76aa)
NAMES = ['tabl_4_9_h3', 'tabl_4_a_razlozhenie', 'tabl_4_a_utechka', 'tabl_4_a_vklad_strategiy',
         'tabl_4_final_c_gibrid_vs_harj', 'tabl_4_final_ce_kmin', 'tabl_4_final_delta', 'tabl_4_final_h3',
         'tabl_4_final_marzha_2022_2026', 'tabl_4_final_msp', 'tabl_4_final_sharpe_vs_stavka_2022_2026',
         'tabl_4_final_vybor', 'tabl_4_msp_a', 'tabl_4_msp_a_mask1', 'tabl_4_msp_a_po_godam',
         'tabl_4_msp_a_po_godam_mask1', 'tabl_4_otsev_a']
NUM_DOT = re.compile(r'^[+-]?\d+\.\d*([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$')
NUM_COMMA = re.compile(r'^[+-]?\d+,\d*([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$')
BOM = '﻿'


def read_cells(b):
    """Байты CSV -> список строк-ячеек. Разделитель: ';' если ';' в заголовке, иначе ','."""
    t = b.decode('utf-8')
    bom = t.startswith(BOM)
    t = t[1:] if bom else t
    sep = ';' if ';' in t.splitlines()[0] else ','
    return list(csv.reader(io.StringIO(t, newline=''), delimiter=sep)), sep, bom


def conv(cell, sep):
    # в таблицах с ';' десятичный знак уже запятая — ячейки не меняются, добавляется только BOM
    return cell.replace('.', ',') if sep == ',' and NUM_DOT.match(cell) else cell


def back(cell):
    return cell.replace(',', '.') if NUM_COMMA.match(cell) else cell


def to_bytes(rows):
    s = io.StringIO(newline='')
    w = csv.writer(s, delimiter=';', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    w.writerows(rows)
    return (BOM + s.getvalue()).encode('utf-8')


def git_bytes(rel):
    return subprocess.run(['git', '-C', str(ROOT), 'show', f'{COMMIT}:{rel}'], capture_output=True, check=True).stdout


def check(n):
    new, raw = TABLES / f'{n}.csv', TABLES / f'{n}_raw.csv'
    rb, nb = raw.read_bytes(), new.read_bytes()
    assert rb == git_bytes(f'docs/tables/strat_test/{n}.csv'), f'{raw.name}: не равен файлу коммита {COMMIT}'
    R, rsep, _ = read_cells(rb)
    N, sep, bom = read_cells(nb)
    assert sep == ';' and bom, f'{new.name}: формат не «;» + BOM'
    assert len(R) == len(N) and all(len(a) == len(b) for a, b in zip(R, N)), f'{new.name}: размер не совпал'
    k = 0
    for i, (a, b) in enumerate(zip(R, N)):
        for j, (x, y) in enumerate(zip(a, b)):
            if rsep == ',':
                assert back(y) == x, f'{new.name}: строка {i} столбец {j}: {x!r} -> {y!r}'
            else:
                assert y == x, f'{new.name}: строка {i} столбец {j}: {x!r} -> {y!r}'
            assert not NUM_DOT.match(y), f'{new.name}: осталась точка {y!r}'
            k += 1
    return len(N) - 1, len(N[0]), k, hashlib.sha256(rb).hexdigest()[:16], hashlib.sha256(nb).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    for n in NAMES:
        src, raw = TABLES / f'{n}.csv', TABLES / f'{n}_raw.csv'
        if not a.check:
            if raw.exists():
                sys.exit(f'СТОП: {raw.name} уже есть — повторный запуск не нужен (--check)')
            b = src.read_bytes()
            src.rename(raw)
            rows, sep, _ = read_cells(b)
            src.write_bytes(to_bytes([[conv(c, sep) for c in r] for r in rows]))
        r, c, k, hr, hn = check(n)
        print(f'OK {n}.csv: {r} строк × {c} столбцов, {k} ячеек совпали; raw sha {hr}…, new sha {hn}…')


if __name__ == '__main__':
    main()
