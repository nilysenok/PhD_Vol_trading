"""Переформатирование таблиц единственного запуска теста главы 3 (final_test.py --scope test, 26.10 08:57, коммит 002a74c)
в формат владельца: разделитель ';', десятичная запятая, UTF-8 с BOM. БЕЗ ПЕРЕСЧЁТА: работает со строками ячеек, числа не парсятся
и не округляются; в числовой ячейке меняется только десятичная точка на запятую (9.688e-05 -> 9,688e-05). Нечисловые ячейки
('<0,001', 'True', 'Среднее 2017-2026', '2017-2019', пустые = NaN) не трогаются.

Исходник docs/tables/table_3_<имя>_test.csv переименовывается в table_3_<имя>_test_raw.csv (байты не меняются), на прежнем имени
пишется переформатированная таблица. Проверка после записи: (1) _raw побайтно равен файлу коммита 002a74c (git show);
(2) таблицы совпадают по числу строк и столбцов; (3) каждая ячейка: нормализованная строка (',' -> '.' в числовой ячейке)
совпадает с исходной побайтно, нечисловая ячейка совпадает как есть.

python3 scripts/analysis/tables_test_format.py            # переименовать, переформатировать, проверить
python3 scripts/analysis/tables_test_format.py --check    # только проверка уже переформатированных
"""
import argparse, csv, hashlib, io, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'docs/tables'
COMMIT = '002a74c'
# ровно 21 таблица, записанная запуском теста 26.10 08:57 (docs/otchety/final_test_run.log)
NAMES = ['3_2', '3_3', '3_4', '3_6', '3_7', '3_4_rmse', '3_6_rmse', '3_7_rmse', '3_4_mae', '3_6_mae', '3_7_mae', '3_8', '3_9',
         '3_final_po_godam', '3_raskladka', '3_mz', '3_tickers', '3_subperiods', '3_terciles', '3_encompassing', '3_mcs']
NUM_DOT = re.compile(r'^[+-]?\d+\.\d*([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$')
NUM_COMMA = re.compile(r'^[+-]?\d+,\d*([eE][+-]?\d+)?$|^[+-]?\d+[eE][+-]?\d+$')
BOM = '﻿'


def read_cells(b):
    """Байты CSV -> список строк-ячеек. Разделитель: ';' если файл с BOM и ';' в заголовке, иначе ','."""
    t = b.decode('utf-8')
    bom = t.startswith(BOM)
    t = t[1:] if bom else t
    sep = ';' if ';' in t.splitlines()[0] else ','
    return list(csv.reader(io.StringIO(t, newline=''), delimiter=sep)), sep, bom


def conv(cell):
    return cell.replace('.', ',') if NUM_DOT.match(cell) else cell


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
    new, raw = TABLES / f'table_{n}_test.csv', TABLES / f'table_{n}_test_raw.csv'
    rb, nb = raw.read_bytes(), new.read_bytes()
    assert rb == git_bytes(f'docs/tables/table_{n}_test.csv'), f'{raw.name}: не равен файлу коммита {COMMIT}'
    R, _, _ = read_cells(rb)
    N, sep, bom = read_cells(nb)
    assert sep == ';' and bom, f'{new.name}: формат не «;» + BOM'
    assert len(R) == len(N) and all(len(a) == len(b) for a, b in zip(R, N)), f'{new.name}: размер не совпал'
    k = 0
    for i, (a, b) in enumerate(zip(R, N)):
        for j, (x, y) in enumerate(zip(a, b)):
            assert back(y) == back(x), f'{new.name}: строка {i} столбец {j}: {x!r} -> {y!r}'
            assert not NUM_DOT.match(y), f'{new.name}: осталась точка {y!r}'
            k += 1
    return len(N) - 1, len(N[0]), k, hashlib.sha256(rb).hexdigest()[:16], hashlib.sha256(nb).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    for n in NAMES:
        src, raw = TABLES / f'table_{n}_test.csv', TABLES / f'table_{n}_test_raw.csv'
        if not a.check:
            if raw.exists():
                sys.exit(f'СТОП: {raw.name} уже есть — повторный запуск не нужен (--check)')
            b = src.read_bytes()
            src.rename(raw)
            rows, _, _ = read_cells(b)
            src.write_bytes(to_bytes([[conv(c) for c in r] for r in rows]))
        r, c, k, hr, hn = check(n)
        print(f'OK table_{n}_test.csv: {r} строк × {c} столбцов, {k} ячеек совпали; raw sha {hr}…, new sha {hn}…')


if __name__ == '__main__':
    main()
