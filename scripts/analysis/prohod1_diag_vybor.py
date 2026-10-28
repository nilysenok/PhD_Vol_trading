# -*- coding: utf-8 -*-
"""Шаг 1 диагностики прохода 1 (Ж-091 п. 2): лучшие 20 точек каждой Sk по целевой Ж-090 (справочно, train4, в отбор не идёт).

Только чтение ledger прохода 1. Отбор: основная строка (progon_osnovnaya) -> tselevaya_train4 с платой за шорт 5 %
(zaem_baza из ledger) -> сначала устойчивые (mSR каждого года > 0) по mSR за весь период. Если устойчивых < 20 — все устойчивые;
↺ отступление: добор до 20 неустойчивыми по mSR помечен ustojchiva=False и считается отдельной группой (иначе S1/S5/S6 пусты).
A' — после разбора (центров региона нет).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import ledger_v2 as lm, progon_osnovnaya as po, progon_sobol as ps, zaem_v2  # noqa: E402
from fastbt.etap1_poisk_vorota import tselevaya_train4  # noqa: E402

TOP = 20


def vybor(root, sid: str, top: int = TOP) -> tuple:
    """-> (таблица top точек, {osnovnyh, ustojchivyh}). Колонки: sid, sobol_i, ustojchiva, msr_vsego, msr_<год>, pt (dict)."""
    ch = lm.Chitatel(root, sid, sverka_kod=True)
    daty = np.load(f"{root}/{sid}/daty.npy")
    rows = [r for r, m in zip(ch.rows(), po.maska(ch.rows())) if m and r.get("istochnik") == "sobol1"]
    M, out = ch.matrica(rows), []
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)         # плата за шорт не учтена = ошибка, не молчание
        for j, r in enumerate(rows):
            m, py, ust = tselevaya_train4(M[:, j], daty, zaem_v2.zaem_ryad(root, sid, r))
            if np.isfinite(m) and (ust or all(np.isfinite(list(py.values())))):       # добор — только с mSR во всех 4 годах
                out.append(dict(sid=sid, sobol_i=r["sobol_i"], ustojchiva=bool(ust), msr_vsego=m,
                                **{f"msr_{y}": v for y, v in py.items()}, klyuch=r["klyuch"]))
    df = pd.DataFrame(out).sort_values(["ustojchiva", "msr_vsego", "sobol_i"], ascending=[False, False, True])
    df = df.head(top).reset_index(drop=True)
    pts = dict(ps.tochki(ps.load_kub(sid), 0, 5000))
    df["pt"] = [pts[i] for i in df.sobol_i]
    for pt, kl in zip(df.pt, df.klyuch):
        if lm.klyuch(sid, pt, ps.PERIOD, ps.KOMISSIYA, None) != kl:
            raise ValueError(f"{sid}: точка не восстановлена из куба")
    return df.drop(columns="klyuch"), dict(osnovnyh=len(rows), ustojchivyh=int(sum(o["ustojchiva"] for o in out)))


if __name__ == "__main__":
    d = vybor(sys.argv[1], sys.argv[2])
    print(d[0].drop(columns="pt").to_string(), d[1])
