# -*- coding: utf-8 -*-
"""Основная и справочная строки прогона (Ж-084 п. 2, толкование [Ж-084] п. 2). Данных не читает.

Признак ставится по параметрам точки, не по исходам сделок. Справочная строка — та, где сделка может выйти по сигналу
на close x: t_vyhod = close и выход по сигналу включён правилами стратегии (VYHOD_PO_SIGNALU). Основные: open2 (open дня
x+1, exec_e2._scan: pend due = j+1, hhmm None -> d.o[0]) и close, где выхода по сигналу нет. 1500 / 1700 — справочные
до слова владельца. Справочные строки в ledger и реестре остаются; из выбора (регион, WF, PBO, A′, N DSR) исключаются.
"""
from __future__ import annotations

import numpy as np

SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
# Params.exit_signal = True по умолчанию (exec_e_types.py:28), params_of (etap1_eval.py:63-67) и _params (etap1_eval_v2.py:77-93)
# его не меняют, проверка в exec_e_run.step (exec_e_run.py:147): выход по сигналу включён у всех шести стратегий.
VYHOD_PO_SIGNALU = {s: True for s in SIDS}
OSNOVNYE_T = ("open2",)
SPRAVOCHNYE_T = ("1500", "1700")


def _pt(x: dict) -> dict:
    """Точка из ключа ledger (плоский, либо с вложенным pt) или из строки ledger/реестра."""
    if "klyuch" in x:
        x = x["klyuch"]
    return x.get("pt", x)


def _tok(v) -> str:
    """1500 / '1500' / '1500.0' (tok ledger для числа) -> '1500'; прочее — строкой."""
    s = str(v)
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def osnovnaya(klyuch: dict, sid: str | None = None, tablica: dict | None = None) -> bool:
    """True — основная строка. Только параметры точки: t_vyhod и таблица «выход по сигналу» по стратегии.
    Оси t_vyhod нет — берётся умолчание оси close (контракт §1.1)."""
    pt = _pt(klyuch)
    t = _tok(pt.get("t_vyhod", "close"))
    if t in OSNOVNYE_T:
        return True
    if t in SPRAVOCHNYE_T:
        return False
    if t != "close":
        raise ValueError(f"t_vyhod {t!r}: допустимо close, 1500, 1700, open2")
    tb = VYHOD_PO_SIGNALU if tablica is None else tablica
    sid = sid or klyuch.get("strategiya")
    if sid not in tb:
        raise ValueError(f"для t_vyhod = close нужна стратегия из таблицы {sorted(tb)}, получено {sid!r}")
    return not tb[sid]


def osnovnaya_stroki(row: dict, tablica: dict | None = None):
    """Признак строки ledger / реестра: bool; None — оси t_vyhod в ключе нет (строка не прогона v2, не классифицируется)."""
    if "t_vyhod" not in _pt(row):
        return None
    return osnovnaya(row, row.get("strategiya"), tablica)


def maska(rows: list, tablica: dict | None = None) -> np.ndarray:
    """bool на каждую строку: не классифицируемая (нет оси) остаётся в выборе, справочная — нет."""
    return np.array([osnovnaya_stroki(r, tablica) is not False for r in rows], bool)


def rows_osnovnye(ch, tablica: dict | None = None, status="ok") -> list:
    """Основные строки Chitatel: ch.matrica(rows_osnovnye(ch)) — матрица PBO и региона без справочных столбцов."""
    rs = ch.rows(status)
    return [r for r, m in zip(rs, maska(rs, tablica)) if m]


def chisla_N(rows: list, tablica: dict | None = None) -> dict:
    """N для DSR по основным строкам и отдельно число справочных (уникальные klyuch_sha); в N DSR идёт только osnovnye."""
    osn, sp = set(), set()
    for r, m in zip(rows, maska(rows, tablica)):
        (osn if m else sp).add(r["klyuch_sha"])
    return dict(N_osnovnye=len(osn), N_spravochnye=len(sp))
