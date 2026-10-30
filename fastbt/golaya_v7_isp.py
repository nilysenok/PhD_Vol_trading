"""Л1.3 (Ж-146): дневной исполнитель голой идеи одного тикера, вес сделки 1. Деньги.

Сигнал дня d (±1/0) известен на закрытии d (или в T дня d). Вход: `d1` — open d+1 (Д2: не раньше d+1; вход в день
сигнала — ValueError), `T` — цена T дня d (`p_T[d]`, её строит вызывающий: первый бар с началом ≥ T). Выход: close дня
d+τ; флаг `exit_l`/`exit_s` дня k (k ≥ день входа для `d1`, > для `T`): `d1` — open k+1, `T` — `p_T[k]`; раньше по времени
выигрывает. Без τ и флага, и если окно кончилось, — по последнему close с обеими комиссиями (Н5; mutations № 5, 6).
Сделки не накрываются. Комиссия и плата за шорт — те же константы, что в движке и `kross_seksiya.kniga` (Д7).
Ночи позиции e…xd−1: плата за шорт ZAEM·gap/365 (вход и выход в один день — 0, как zaem_v2), ставка на занятый капитал.
Дыра в цене входа — сделка пропущена и посчитана (`propusk`); дыра в пути/выходе — ValueError (Д4: без ffill).
"""
from __future__ import annotations

import numpy as np

from fastbt.kross_seksiya import DNEI, ZAEM
from fastbt.tf_os_v5 import KOMISSIYA

VHODY = ("d1", "T")


def proverka_d2(sig_i: int, vhod_i: int) -> None:
    if vhod_i <= sig_i:
        raise ValueError("Д2: вход в день сигнала или раньше")


def _tsena(x, k: int, chto: str) -> float:
    v = float(x[k])
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"пропуск цены {chto} в строке {k} (Д4: не заполняется)")
    return v


def _vyhod(i, e, s, T, vhod, tau, flag, o, p_T):
    """(день, ранг, цена-или-None): ранг 0 — выход по флагу (раньше close того же дня), 1 — по close."""
    luchshiy = (T - 1, 1, None)
    if tau is not None and i + tau < T:
        luchshiy = (i + tau, 1, None)
    if flag is not None:
        for k in range(e if vhod == "d1" else e + 1, T):
            xd = k + 1 if vhod == "d1" else k
            if flag[k] and xd < T:
                if (xd, 0) <= luchshiy[:2]:
                    luchshiy = (xd, 0, _tsena(o if vhod == "d1" else p_T, xd, "выхода"))
                break
    return luchshiy


def _zapis(out, s, e, xd, pe, px, o, c, D, gap, rate, kom, zaem):
    """Дневные ряды сделки; pnl дня k — от цены конца k−1 (для e — от цены входа) до цены конца k, на цену входа."""
    pred = pe
    for k in range(e, xd + 1):
        konec = px if k == xd else _tsena(c, k, "close")
        out["pnl"][k] += s * (konec - pred + (D[k] if k > e else 0.0)) / pe
        out["poz"][k] = s
        pred = konec
    for k in range(e, xd):                                   # ночи
        out["stavka"][k] += rate[k] * gap[k] / DNEI
        if s < 0:
            out["zaem"][k] += zaem * gap[k] / DNEI
    out["kom"][e] += kom
    out["kom"][xd] += kom * px / pe
    out["sdelki"].append((s, e, xd, pe, px))


def ispolnit(sig, o, c, p_T=None, vhod="d1", tau=None, exit_l=None, exit_s=None, *,
             dates, rate, D=None, kom: float = KOMISSIYA, zaem: float = ZAEM) -> dict:
    """dict(pnl, kom, zaem, stavka, excess, poz, sdelki, propusk); ряды float [T], poz — знак в дни с pnl."""
    sig, o, c = np.asarray(sig), np.asarray(o, float), np.asarray(c, float)
    T = len(sig)
    D = np.zeros(T) if D is None else np.asarray(D, float)
    rate = np.asarray(rate, float)
    if vhod == "d0":
        proverka_d2(0, 0)                                    # вход в день сигнала — отказ Д2
    if vhod not in VHODY:
        raise ValueError(f"vhod: {VHODY}")
    if vhod == "T" and (p_T is None or len(p_T) != T):
        raise ValueError("vhod=T требует p_T той же длины")
    if not (len(o) == len(c) == len(D) == len(rate) == len(dates) == T):
        raise ValueError("длины sig, o, c, dates, rate, D должны совпадать")
    if np.any(~np.isin(sig, (-1, 0, 1))):
        raise ValueError("sig: только -1, 0, 1")
    if tau is not None and tau < (1 if vhod == "d1" else 0):
        raise ValueError("τ: выход раньше входа")
    gap = np.append(np.diff(np.asarray(dates, "datetime64[D]")).astype(float), 1.0)[:T]
    out = {k: np.zeros(T) for k in ("pnl", "kom", "zaem", "stavka", "poz")}
    out.update(sdelki=[], propusk=0)
    i = 0
    while i < T:
        s = int(sig[i])
        e = i + 1 if vhod == "d1" else i
        if s == 0 or e >= T:
            i += 1
            continue
        if vhod == "d1":
            proverka_d2(i, e)
        pe = float((o if vhod == "d1" else p_T)[e if vhod == "d1" else i])
        if not np.isfinite(pe) or pe <= 0:
            out["propusk"] += 1
            i += 1
            continue
        xd, rang, px = _vyhod(i, e, s, T, vhod, tau, exit_l if s > 0 else exit_s, o, p_T)
        px = _tsena(c, xd, "close") if px is None else px
        _zapis(out, s, e, xd, pe, px, o, c, D, gap, rate, kom, zaem)
        i = xd + 1 if (vhod == "T" and rang == 1) else xd
    out["excess"] = out["pnl"] - out["kom"] - out["zaem"] - out["stavka"]
    return out
