"""Общие функции таблиц для текста (tables_text.py): пути, формат CSV (';', десятичная запятая, UTF-8 с BOM), строка-источник.
Только разработочный набор: val 2017-2021 (шаги 2-6, OOS). Подтверждающий набор и тест не читаются."""
import csv, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = Path(os.environ.get('PHD_DATA', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')) / 'regen'
OUT = ROOT / 'docs' / 'tables'
OTCH = ROOT / 'docs' / 'otchety'
YEARS = (2017, 2018, 2019, 2020, 2021)


def fmt(x, nd=4):
    """число -> строка с десятичной запятой; None/NaN -> пусто; строки и целые как есть"""
    if x is None or (isinstance(x, float) and x != x): return ''
    if isinstance(x, bool): return 'да' if x else 'нет'
    if isinstance(x, int): return str(x)
    if isinstance(x, float): return f'{x:.{nd}f}'.replace('.', ',')
    return str(x)


def pct(a, b, nd=1):
    """(a / b - 1) в процентах: отрицательное = a лучше b"""
    if a is None or b is None or b != b or a != a or b == 0: return None
    s = fmt(100 * (a / b - 1), nd); return s[1:] if s.startswith('-') and s.strip('-0,') == '' else s  # без «-0,0»


def write_table(name, header, rows, source, note=None):
    """CSV: заголовок, строки, затем строка-примечание и строка-источник; фиксированное имя файла"""
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, delimiter=';', lineterminator='\n')
        w.writerow(header)
        for r in rows: w.writerow([fmt(v) for v in r])
        if note: w.writerow([f'Примечание: {note}'.replace(';', ' ·')])
        w.writerow([f'Источник: {source}'.replace(';', ' ·')])
    return len(rows)
