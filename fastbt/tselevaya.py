# -*- coding: utf-8 -*-
"""tselevaya.py — целевая этапа 1 по Ж-090 п. 1: mSR за весь период + условие «mSR каждого года > 0». Данных не читает.

Одно определение mSR (Д7): Фавр–Галеано, pbo_pravilo.msr_stolbcy (= metriki_e.msr, тест их равенства в pbo_pravilo).
Сюда вынесено из etap1_poisk_vorota.py (там лимит 200 строк); оттуда реэкспортируется.
"""
from __future__ import annotations

import warnings

import numpy as np

from fastbt.pbo_pravilo import MIN_DNEY, msr_stolbcy

GODA_TRAIN4 = (2016, 2017, 2018, 2019)
STAVKA_SHORT = 0.05      # Ж-089: плата за шорт, годовых


def _goda(daty) -> np.ndarray:
    return np.asarray(daty).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970


def msr_i_gody(X, daty, goda=None, min_dney: int = MIN_DNEY):
    """X — дни × столбцы. Возврат: (msr_vsego[N], msr_po_godam[N × Y], ustojchiva[N], goda).
    goda=None — годы, у которых в daty ≥ min_dney дней; goda задан — год с меньшим числом дней даёт nan (не устойчива).
    ustojchiva = mSR каждого года > 0 (nan > 0 ложно)."""
    X, g = np.asarray(X, float), _goda(daty)
    if X.shape[0] != len(g):
        raise ValueError("X и daty разной длины")
    ys = tuple(goda) if goda is not None else tuple(int(y) for y in np.unique(g) if (g == y).sum() >= min_dney)
    py = np.column_stack([msr_stolbcy(X[g == y]) if (g == y).sum() >= max(min_dney, 2) else np.full(X.shape[1], np.nan)
                          for y in ys]) if ys else np.empty((X.shape[1], 0))
    ust = (py > 0).all(axis=1) if ys else np.zeros(X.shape[1], bool)
    return msr_stolbcy(X), py, ust, ys


def tselevaya_train4(excess_d, daty, zaem_baza=None, stavka: float = STAVKA_SHORT):
    """-> (msr_vsego, msr_po_godam {год: mSR}, ustojchiva). excess_d — дневной excess портфеля 2016–2019 (комиссия 0,04 % уже
    в нём). Плата за шорт: excess − stavka · zaem_baza (zaem_baza — ряд Л-18 той же длины: стоимость короткой позиции в долях
    капитала × календарных дней/365 за день). zaem_baza=None — плата НЕ учтена: UserWarning, не подставляем молча."""
    r = np.asarray(excess_d, float)
    if zaem_baza is None:
        warnings.warn("tselevaya_train4: плата за шорт не учтена (zaem_baza=None)", UserWarning, stacklevel=2)
    else:
        z = np.asarray(zaem_baza, float)
        if z.shape != r.shape:
            raise ValueError("zaem_baza и excess_d разной длины")
        r = r - stavka * z
    m, py, ust, ys = msr_i_gody(r[:, None], daty, GODA_TRAIN4)
    return float(m[0]), {y: float(py[0, i]) for i, y in enumerate(ys)}, bool(ust[0])
