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
from fastbt.exec_e_zayavki import INTRA


def ke_levels(up, dn, k_e, atr):
    """Л6: сдвиг стоп-заявок входа на k_e·ATR — buy-stop вверх, sell-stop вниз (k_e < 0 — оба ближе к цене). nan ATR — обе nan."""
    a = k_e * atr
    return up + a, dn - a


def stop_novye(p) -> list:
    """V6-11: оси v5, перенесённые на стоп-заявку тренда; быстрое ядро их на ней не знает — точка идёт эталоном (fast_exec.ne_znaet)."""
    if p.vhod_tip != "stop_zayavka":
        return []
    return [n for n in ("ed_s", "ed_t", "chand", "b0", "b0_c", "sp", "step_s", "pir", "reentry_den") if getattr(p, n) != getattr(type(p)(), n)]


class RunStop(Run2):
    def __init__(s, *a, urovni, **kw):
        super().__init__(*a, **kw)
        s.u = urovni

    def _pair(s, i):
        """OCO-пара вечера i: (buy-stop, sell-stop), nan → ±inf; обе nan — None. k_e·ATR на close i; нет ATR — заявки нет."""
        up, dn = s.u["up"][i], s.u["dn"][i]
        if s.p.k_e != 0:
            up, dn = ke_levels(up, dn, s.p.k_e, s.atr[i])
        if np.isfinite(up) or np.isfinite(dn):
            return up if np.isfinite(up) else np.inf, dn if np.isfinite(dn) else -np.inf

    def _decide(s, i, j, close, nb):
        if s.tr is not None or s.order is not None or j + 1 >= len(s.ix) or not np.isfinite(s.sigma[i]):
            return
        pr = s._pair(i)
        if pr is not None:
            s.n_orders += 1
            s.order = dict(due=j + 1, d_sig=i, up=pr[0], dn=pr[1])

    def _place(s, *a, **kw):
        raise RuntimeError("RunStop не ставит лимитных заявок (повторный вход — _reenter)")

    def _try_order(s, i, j, d):
        o, s.order = s.order, None
        r = stop_vhod(d.o, d.h, d.l, o["up"], o["dn"])
        if r is None:
            return
        s.n_fills += 1
        s._open(i, j, d, r, o["d_sig"])

    def _open(s, i, j, d, r, d_sig):
        sd, b, px = r
        s._enter(i, j, b, px, b, s.sigma[d_sig], d_sig, sd)
        S, _ = Run2._levels(s, j)                   # в баре входа — только защитный стоп σ; стоп-заявка выхода активна со следующего бара
        if not s._stop_vhoda(i, j, d, b, px, S):     # стоп активен с цены входа (БЛОКЕР #596)
            s._scan(i, j, d, b + 1)
        s._reenter(i, j, d)

    def _scan(s, i, j, d, b0):
        super()._scan(i, j, d, b0)
        s._reenter(i, j, d)

    def _reenter(s, i, j, d):
        """V6-11c (Ж-111 п. 3): после внутридневного выхода дня i пара вечера j−1 (тот же уровень, что поставил бы _decide) стоит на барах
        строго после бара выхода; не более одного повторного входа за день. Гэп через уровень — по open. done_re ставится до проверки
        уровней: тогда Run.step не вызывает свой (лимитный) reenter."""
        x = s.last_x
        if (not s.p.reentry_den or s.tr is not None or s.order is not None or j == 0 or x is None or x.exit_day != i
                or x.reason not in INTRA or s.done_re == i or x.exit_bar + 1 >= len(d.c)):
            return
        s.done_re = i
        ip = s.ix[j - 1]
        pr = s._pair(ip) if np.isfinite(s.sigma[ip]) else None
        if pr is None:
            return
        b0 = x.exit_bar + 1
        r = stop_vhod(d.o[b0:], d.h[b0:], d.l[b0:], *pr)
        if r is not None:
            s.n_orders, s.n_fills = s.n_orders + 1, s.n_fills + 1
            s._open(i, j, d, (r[0], r[1] + b0, r[2]), ip)

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
