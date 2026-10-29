"""RunStop(Run2): ось vhod_tip = stop_zayavka (Ж-091 п. 1.2–1.3) — вход и выход стоп-заявками S3/S4.

urovni = {up, dn, xl, xs} по дням (fastbt/stop_zayavki.py): значения дня d считаются по данным ≤ d и действуют на d+1.
Вечером d без позиции и заявки ставится OCO-пара buy-stop up[d] / sell-stop dn[d] на день d+1 (лимит k_e не применяется);
исполнение — exec_e_rules.stop_vhod (касание, цена — худшая из open и уровня, активна с первого бара d+1).
Выход: стоп-заявка на xl[d_пред] / xs[d_пред] — пересчитывается каждый вечер, действует весь день d+1; вместе со стопом σ — строже из двух,
в одном баре стоп первым; причина "kanal" (внутридневной выход, комиссия в строке начала интервала). Сигналом sig вход и выход не управляются.
"""
from __future__ import annotations

import numpy as np

from fastbt.exec_e2 import Run2
from fastbt.exec_e_rules import stop_vhod


def ke_levels(up, dn, k_e, atr):
    """Л6: сдвиг стоп-заявок входа на k_e·ATR — buy-stop вверх, sell-stop вниз (k_e < 0 — оба ближе к цене). nan ATR — обе nan."""
    a = k_e * atr
    return up + a, dn - a


class RunStop(Run2):
    def __init__(s, *a, urovni, **kw):
        super().__init__(*a, **kw)
        s.u = urovni

    def _decide(s, i, j, close, nb):
        if s.tr is not None or s.order is not None or j + 1 >= len(s.ix) or not np.isfinite(s.sigma[i]):
            return
        up, dn = s.u["up"][i], s.u["dn"][i]
        if s.p.k_e != 0:
            up, dn = ke_levels(up, dn, s.p.k_e, s.atr[i])    # ATR на close d; нет ATR — заявки нет
        if np.isfinite(up) or np.isfinite(dn):
            s.n_orders += 1
            s.order = dict(due=j + 1, d_sig=i, up=up if np.isfinite(up) else np.inf, dn=dn if np.isfinite(dn) else -np.inf)

    def _try_order(s, i, j, d):
        o, s.order = s.order, None
        r = stop_vhod(d.o, d.h, d.l, o["up"], o["dn"])
        if r is None:
            return
        sd, b, px = r
        s.n_fills += 1
        s._enter(i, j, b, px, b, s.sigma[o["d_sig"]], o["d_sig"], sd)
        S, _ = Run2._levels(s, j)                   # в баре входа — только защитный стоп σ; стоп-заявка выхода активна со следующего бара
        if not s._stop_vhoda(i, j, d, b, px, S):     # стоп активен с цены входа (БЛОКЕР #596)
            s._scan(i, j, d, b + 1)

    def _levels(s, j):
        S, T = super()._levels(j)
        if j == 0:
            return S, T
        X = s.u["xl" if s.tr.side > 0 else "xs"][s.ix[j - 1]]      # заявка выхода, выставленная вечером прошлого торгового дня
        if np.isfinite(X):
            X = s._stop_shift(X)
            if S is None or (X > S if s.tr.side > 0 else X < S):
                S, s.sr = X, "kanal"
        return S, T
