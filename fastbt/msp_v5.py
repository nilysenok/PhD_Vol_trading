"""Ф-1: драйвер МСП над ансамблями v5 (λ = 0 / 1). Склейка готовых `reyting` (Ф3) и `kniga_msp` (Ф4); новой логики нет.

Вход — сигналы членов ансамбля, которые пересчитывает вызывающий (в ledger 2.1 стороны и силы нет):
  sig[Sk] = {"storona": [d×тикер] ∈ {−1,0,+1}, "sila": [d×тикер], "g": [d×тикер]} — всё по данным к 18:50 дня d;
  sigma [d×тикер] — σ̂_d; sektor {тикер: сектор}; dohod = {"R_brutto": [d×тикер],
  "T": [d×тикер]} — брутто-доходность ставки ПРИНЯТОЙ на день d на единицу позиции (знаковая, без комиссии) и
  оборот дня на единицу (≥ 0, Σ|pos| входов и выходов, `tf_os_v5.dohod`); исполнение d+1, Д2, вызывающий выравнивает;
  imoex [d] — то же для хеджа MIX. ex_kniga = u·R_brutto − ставка·Σ|u|·T − ставка·|Δ хеджа|.
День d: скоры (фазы по ≤ d−1, `fazy._LAG = 1`) -> `kniga_msp.kniga_dnya` по истории строк < d -> только потом доходность d
дописывается в историю. λ = 0 — фазы выключены (вес 1), совпадает с запуском без фаз побайтно.
Данные ≥ 2020 не читаются (ValueError).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fastbt import kniga_msp, reyting

END = pd.Timestamp("2020-01-01")


@dataclass
class Rezultat:
    u: pd.DataFrame        # веса ставок [d × (Sk, тикер)] в долях NAV, после нетто, плеча и выключателя
    hedzh: pd.Series       # доля NAV в MIX
    ex_kniga: pd.Series    # excess книги по дням (ставки + хедж)
    skory: dict            # {Sk: score [d × тикер]}


def skory(sig: dict, faza_r, faza_s, lam: float, porog: float = 0.0) -> dict:
    """{Sk: score-панель} по `reyting.reyting`. faza_r [d], faza_s [d × тикер]; NaN → вес 1."""
    return {sk: reyting.reyting(sk, v["storona"], v["sila"], v["g"], faza_r, faza_s, lam, porog)
            for sk, v in sig.items()}


def _fazy(sig: dict, dannye, daty, faza_r, faza_s):
    cols = next(iter(sig.values()))["storona"].columns
    if faza_r is not None and faza_s is not None:
        return faza_r, faza_s
    if dannye is None:      # фаз нет вовсе: вес 1
        return pd.Series(np.nan, index=daty, dtype=object), pd.DataFrame(np.nan, index=daty, columns=cols, dtype=object)
    return reyting.fazy_dnya(list(cols), daty, dannye)


def _kvartal(idx) -> np.ndarray:
    return (idx.year * 4 + (idx.month - 1) // 3).to_numpy()


def progon(sig: dict, sigma: pd.DataFrame, sektor: dict, dohod: pd.DataFrame, imoex: pd.Series, lam: float,
           dannye: dict | None = None, faza_r=None, faza_s=None, k: int = 20, rho: float = 0.01,
           plecho: float = 2.0, tsel: float = 0.10, porog: float = 0.0) -> Rezultat:
    """Дневной цикл книги МСП. Без `dannye` и без faza_* фазы не используются (вес 1)."""
    daty = next(iter(sig.values()))["storona"].index
    if (daty >= END).any():
        raise ValueError("данные ≥ 2020 не читаются")
    sks, tik = list(sig), list(sigma.columns)
    fr, fs = _fazy(sig, dannye, daty, faza_r, faza_s)
    sc = skory(sig, fr, fs, lam, porog)
    kand = pd.MultiIndex.from_product([sks, tik], names=["sk", "tiker"])
    S = np.concatenate([sc[s].reindex(daty)[tik].to_numpy(float) for s in sks], axis=1)       # [T, S·N]
    sg = np.tile(sigma.reindex(daty)[tik].to_numpy(float), (1, len(sks)))
    if not (isinstance(dohod, dict) and {"R_brutto", "T"} <= set(dohod)):
        raise ValueError("dohod: нужен dict {'R_brutto','T'} (брутто на единицу позиции со знаком и оборот дня ≥ 0); "
                         "один ряд с комиссией внутри даёт шорту возврат комиссии (Ж-149)")
    pan = lambda x: np.nan_to_num(np.tile(x.reindex(daty)[tik].to_numpy(float), (1, len(sks))))     # noqa: E731
    R, Tn = pan(dohod["R_brutto"]), pan(dohod["T"])
    if (Tn < 0).any():
        raise ValueError("T (оборот дня на единицу позиции) должен быть ≥ 0")
    sek = np.array([sektor[t] for _ in sks for t in tik], dtype=object)
    skid = np.repeat(np.arange(len(sks)), len(tik))
    im = np.nan_to_num(imoex.reindex(daty).to_numpy(float))
    T = len(daty)
    ist = {"ex_sk": np.zeros((T, len(sks))), "ex_port": np.zeros(T), "ex_kniga": np.zeros(T),
           "imoex": im, "kvartal": _kvartal(daty)}
    U, H = np.zeros((T, len(kand))), np.zeros(T)
    for i in range(T):
        it = kniga_msp.kniga_dnya(i, S[i], sg[i], sek, skid, ist, k, rho, plecho, tsel)
        u0 = kniga_msp.vybor_stavok(S[i], sg[i], sek, k, rho)
        for j in range(len(sks)):                                      # история пишется ПОСЛЕ решения дня i
            ist["ex_sk"][i, j] = float((u0 * R[i])[skid == j].sum())
        ist["ex_port"][i] = float((it.u / (it.m * it.vykl)) @ R[i])
        kom_akcii = kniga_msp.KOMISSIYA * float(np.abs(it.u) @ Tn[i])                  # Ж-149: всегда, не зависит от знака
        ist["ex_kniga"][i] = float(it.u @ R[i] + it.hedzh * im[i]) - kom_akcii - it.komissiya   # + оборот хеджа MIX
        U[i], H[i] = it.u, it.hedzh
    return Rezultat(pd.DataFrame(U, index=daty, columns=kand), pd.Series(H, index=daty, name="hedzh"),
                    pd.Series(ist["ex_kniga"], index=daty, name="ex_kniga"), sc)
