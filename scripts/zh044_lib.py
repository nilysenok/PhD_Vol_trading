"""Ж-044, общие функции: матрица T×N для PBO и справочные PSR/DSR. Только train 2018–2019 (0,04 %); validation не читается.

Столбцы — все уникальные пробы стратегии с боевым эталоном s9 из ledger `sweep_e1v2` (+ доводка), Ж-042 и Ж-043 (реестр отбора A);
значения — дневная избыточная доходность оценщика (`Sweep.evaluate`, сверен с s9 по Ж-040). Пересчёт тех же точек, N не растёт.
Запуск: FASTBT_MAIN=<корень с данными> /usr/bin/python3 scripts/zh044_vorota.py S2
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path[:0] = [".", "scripts"]
import zh043_lib as L                                   # noqa: E402
from fastbt import reestr_otbora as rs                  # noqa: E402
from fastbt import sweep_drivers as sd                  # noqa: E402
from fastbt.ekviv_filtr import prichina_net            # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402

SRC = ("sweep_e1v2", "sweep_e1v2_dostroyka", "zh042_konveyer", "zh043_konveyer")
NBLK = 8


def ledger_keys(sid: str) -> list:
    """Уникальные ключи проб стратегии из ledger (порядок появления). `sweep_e1` (ошибка оценщика) и чужие источники не входят."""
    seen = {}
    for r in rs.chitat(rs.VYHOD):
        if r["strategiya"] == sid and r["istochnik"] in SRC:
            seen.setdefault(r["klyuch"], r["istochnik"])
    return list(seen)


def key_point(exp: dict, key: str) -> dict:
    """Ключ реестра `a=1;b=x` -> точка сетки (пропущенные параметры = None, как в `klyuch_str`)."""
    kv = dict(s.split("=", 1) for s in key.split(";"))
    return {k: L.coerce(exp, k, kv.get(k)) for k in exp["base"]}


def series(sw, p: dict) -> tuple:
    """(доходность 2018, доходность 2019) дневная избыточная при 0,04 %."""
    a, b = sw.evaluate(p, "setup", 0.04), sw.evaluate(p, "control", 0.04)
    return np.asarray(a["excess_d"], float), np.asarray(b["excess_d"], float), list(a["daty"]) + list(b["daty"])


def build_matrix(sid: str, sw, exp: dict) -> dict:
    """Матрица T×N по пробам с боевым эталоном. Возвращает M, ключи, точки, отброшенные (нет эталона / нет в сетке / nan)."""
    cols, keys, pts, drop, dates = [], [], [], {"net_etalona": 0, "ne_v_setke": 0, "nan": 0, "klyuch_ne_kruglyy": 0}, None
    for k in ledger_keys(sid):
        try:
            p = key_point(exp, k)
        except (ValueError, KeyError):
            drop["ne_v_setke"] += 1
            continue
        if rs.klyuch_str(rs.kanon_ledger(pkey(p))) != k:
            drop["klyuch_ne_kruglyy"] += 1
            continue
        if prichina_net(sid, p) is not None:
            drop["net_etalona"] += 1
            continue
        x18, x19, d = series(sw, p)
        x = np.r_[x18, x19]
        if not np.isfinite(x).all():
            drop["nan"] += 1
            continue
        if dates is None:
            dates = d
        assert d == dates, "даты проб расходятся"
        cols.append(x)
        keys.append(k)
        pts.append(p)
    return dict(M=np.column_stack(cols), keys=keys, pts=pts, drop=drop, dates=dates, n18=len(x18))


def pbo(M: np.ndarray) -> dict:
    """PBO по CSCV, 8 блоков (`sweep_drivers.pbo_cscv`, модуль владельца). T и N — размеры матрицы."""
    r = sd.pbo_cscv(M.tolist(), NBLK)
    return dict(pbo=r["pbo"], n_splits=r["n_splits"], T=M.shape[0], N=M.shape[1])


def sr_d(x: np.ndarray) -> float:
    return float(x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0


def moments(x: np.ndarray) -> tuple:
    z = (x - x.mean()) / x.std(ddof=1)
    return float((z ** 3).mean()), float((z ** 4).mean())


def psr_dsr(x: np.ndarray, M: np.ndarray, n_all: int) -> dict:
    """Справочно: PSR (к нулю) и DSR на 2018–2019. T = число дневных наблюдений (len(x)), не сделок. var SR — по столбцам M, N = n_all."""
    T, (sk, ku) = len(x), moments(x)
    s = sr_d(x)
    tr = [sr_d(M[:, j]) for j in range(M.shape[1])]
    star = sd.expected_max_sharpe(n_all, float(np.var(tr, ddof=1)))
    return dict(T=T, psr=sd.psr(s, 0.0, T, sk, ku), dsr=sd.psr(s, star, T, sk, ku), N_dsr=n_all)


def fin(x) -> bool:
    return x is not None and math.isfinite(x)
