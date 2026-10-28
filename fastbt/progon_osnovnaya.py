# -*- coding: utf-8 -*-
"""Основная и справочная строки прогона (Ж-086 п. 1, 3; Ж-085 п. 1–3, 5 отменены). Данных не читает.

Признак — только по параметрам точки, не по исходам сделок. Основная: t_vyhod ∈ {open2, 1500, 1700} (выход по сигналу и τ —
по тому же t_vyhod, τ от него не отделяется) и вход не по close d. Справочная: t_vyhod = close, либо vhod = rynok, пока
rynok исполняется по close d. Вход rynok по open d+1 сделает отдельный лист после прохода 1: тогда RYNOK_ISPOLNENIE = "open_d1"
(одна точка подмены), и rynok станет основным. Строки якоря A = s9 (istochnik из ISTOCHNIKI_YAKORYA) — эталон кода, в отбор не идут.
Справочные строки и якорь остаются в ledger и реестре; из выбора (регион, WF, PBO, A′, N DSR) исключаются.
"""
from __future__ import annotations

import numpy as np

OSNOVNYE_T = ("open2", "1500", "1700")
RYNOK_ISPOLNENIE = "close_d"  # close_d | open_d1: по какой цене исполняется vhod = rynok сейчас
ISTOCHNIKI_YAKORYA = ("yakor",)  # ❓ значение istochnik строк якоря в ledger не зафиксировано; умолчание


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


def osnovnaya(klyuch: dict) -> bool:
    """True — основная строка. Только параметры точки: t_vyhod и vhod. Оси t_vyhod нет — умолчание оси close (контракт §1.1)."""
    pt = _pt(klyuch)
    t = _tok(pt.get("t_vyhod", "close"))
    if t != "close" and t not in OSNOVNYE_T:
        raise ValueError(f"t_vyhod {t!r}: допустимо close, 1500, 1700, open2")
    if t == "close":
        return False
    return not (str(pt.get("vhod")) == "rynok" and RYNOK_ISPOLNENIE != "open_d1")


def osnovnaya_stroki(row: dict):
    """Признак строки ledger / реестра: bool; якорь — False; None — оси t_vyhod в ключе нет (строка не прогона v2)."""
    if row.get("istochnik") in ISTOCHNIKI_YAKORYA:
        return False
    if "t_vyhod" not in _pt(row):
        return None
    return osnovnaya(row)


def maska(rows: list) -> np.ndarray:
    """bool на каждую строку: не классифицируемая (нет оси) остаётся в выборе, справочная и якорь — нет."""
    return np.array([osnovnaya_stroki(r) is not False for r in rows], bool)


def rows_osnovnye(ch, status="ok") -> list:
    """Основные строки Chitatel: ch.matrica(rows_osnovnye(ch)) — матрица PBO и региона без справочных столбцов."""
    rs = ch.rows(status)
    return [r for r, m in zip(rs, maska(rs)) if m]


def chisla_N(rows: list) -> dict:
    """N для DSR по основным строкам и отдельно число справочных (уникальные klyuch_sha); в N DSR идёт только osnovnye."""
    osn, sp = set(), set()
    for r, m in zip(rows, maska(rows)):
        (osn if m else sp).add(r["klyuch_sha"])
    return dict(N_osnovnye=len(osn), N_spravochnye=len(sp))
