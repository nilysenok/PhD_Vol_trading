"""V6-c (Ж-111 пп. 4, 15; спецификация МСП §3.1): ранги силы сигнала и размер ρ/(k_s·σ_d)·m(score).

Чистые функции на numpy. m — массив [N тикеров, T дней] нативной силы сигнала (до обрезки) в кандидат-днях, nan — не кандидат.
Ранг в день d — доля значений окна d−okno…d−1 (день d не входит, данные до d−1) строго меньше m[i, d]; наблюдений < min_nabl → nan
(кандидата нет). ❓ Умолчание: свой ранг, а не `uroven_v3.rang_pul` (#471): тот считает окно с текущим днём и «≤»
(Ж-095), а Ж-111 п. 15 и спец. §3.1 требуют d−1 и «строго меньше».
"""
from __future__ import annotations

import numpy as np

from fastbt import rezhim_e1

OKNO, MIN_NABL = 252, 20
RANG_TIPY = ("tikernyj", "obshchiy", "srednee")     # ось rang_tip
M_TIPY = ("1", "lin", "kvadr")                      # ось m(score): 1; 0,5…1,5; 0,25…2,0
RHO = (0.2, 0.3, 0.5, 0.75, 1.0)                    # % капитала на ставку


def _rang(m: np.ndarray, pul: bool, okno: int, min_nabl: int) -> np.ndarray:
    m = np.asarray(m, float)
    out = np.full(m.shape, np.nan)
    groups = [np.arange(m.shape[0])] if pul else [np.array([i]) for i in range(m.shape[0])]
    for d in range(m.shape[1]):
        for g in groups:
            w = m[g, max(0, d - okno):d]
            v = np.sort(w[np.isfinite(w)])
            ok = np.isfinite(m[g, d])
            if len(v) >= min_nabl and ok.any():
                out[g[ok], d] = np.searchsorted(v, m[g[ok], d], side="left") / len(v)      # строго меньше
    return out


def rang_tikera(m, okno: int = OKNO, min_nabl: int = MIN_NABL) -> np.ndarray:
    """Ранг по паре (стратегия, тикер): пул — прошлые кандидат-дни той же строки."""
    return _rang(m, False, okno, min_nabl)


def rang_obshchiy(m, okno: int = OKNO, min_nabl: int = MIN_NABL) -> np.ndarray:
    """Общий ранг по стратегии: пул — прошлые кандидат-дни всех тикеров."""
    return _rang(m, True, okno, min_nabl)


def rang(m, tip: str, okno: int = OKNO, min_nabl: int = MIN_NABL) -> np.ndarray:
    """Ось rang_tip: tikernyj | obshchiy | srednee (среднее двух рангов; нет любого — nan)."""
    if tip not in RANG_TIPY:
        raise ValueError(f"rang_tip: допустимо {RANG_TIPY}, получено {tip!r}")
    if tip == "tikernyj":
        return rang_tikera(m, okno, min_nabl)
    if tip == "obshchiy":
        return rang_obshchiy(m, okno, min_nabl)
    return (rang_tikera(m, okno, min_nabl) + rang_obshchiy(m, okno, min_nabl)) / 2.0


def mnozhitel(r, m_tip: str) -> np.ndarray:
    """m(score): "1" — 1; "lin" — 0,5 + r (0,5…1,5); "kvadr" — 0,25 + 1,75·r² (0,25…2,0). nan-ранг → nan (при "1" — 1)."""
    r = np.asarray(r, float)
    if m_tip == "1":
        return np.ones(r.shape)
    if m_tip == "lin":
        return 0.5 + r
    if m_tip == "kvadr":
        return 0.25 + 1.75 * r * r
    raise ValueError(f"m_tip: допустимо {M_TIPY}, получено {m_tip!r}")


def ves(sigma, rho_pct: float, k_s: float, m) -> np.ndarray:
    """Вес сделки по дню сигнала: min(1, ρ/(k_s·σ)·m). σ[d] — как у `rezhim_e1.razmer_ves` (WEIGHT_LAG = 0); σ ≤ 0/nan → 0."""
    if not (rho_pct > 0 and k_s > 0):
        raise ValueError(f"ρ и k_s должны быть > 0, получено ρ={rho_pct!r}, k_s={k_s!r}")
    base = rezhim_e1.razmer_ves(sigma, "obratnyj", rho_pct / 100.0 / k_s * np.sqrt(252.0), limit=np.inf)
    return np.nan_to_num(np.minimum(1.0, base * np.asarray(m, float)), nan=0.0)


def sila(ev, t: str, sp: dict, q: float, s0: int, b: int) -> np.ndarray:
    """|нативная сила| Sk до обрезки по дням 0…b окна тикера (как `etap1_plus_eval._strength`). После `ev._positions`."""
    from fastbt.etap1_eval import RM_OFF
    from fastbt.sweep_post import build_view
    td, ind = ev.sk._td[t]
    p = {**ev.base, **sp, "q_sily": q, **RM_OFF[ev.sid]}
    v, _ = build_view(ev.sid, ev._sigs[tuple(sorted(sp.items()))][t], td, ind, p, ev.sk.sig_names)
    st = v.strength[0]
    return np.abs(np.asarray(st[t] if isinstance(st, dict) else st, float)[s0:s0 + b + 1])


def ves_tikerov(ev, pt: dict, sp: dict, q: float, nov: dict, vz: dict, period: str) -> dict:
    """Размер Ж-111 п. 4 по тикерам: {тикер: (w[d], allow[d] | None)}. Ранги — по кандидат-дням (старт сделки) на общем календаре.

    m_tip = "1" — рангов нет, allow = None. Иначе кандидат без ранга (< MIN_NABL наблюдений) не входит: allow[d] = False в день старта.
    """
    if pt["k_s"] == "net" or pt["razmer"] != "ravnyy":
        raise ValueError("rho требует k_s ≠ net и razmer = ravnyy (одно определение размера)")
    tk, info = ev.sk.tickers, {}
    for t in tk:
        m = ev.inp.meta[t]
        s0, b = m["s0"], ev._b(t, period)
        sig = ev._sig2(t, sp, q, False, s0, b, nov, vz, m["dates"][:b + 1])
        sg = ev.inp.sigma[pt["sigma"]][t][:b + 1] * float(vz.get("sigma_mult", 1.0))
        info[t] = (m["dates"][:b + 1], sig, sg, s0, b)
    rho, ks, mt = float(nov["rho"]), float(pt["k_s"]), nov["m_tip"]
    if mt == "1":
        return {t: (ves(i[2], rho, ks, np.ones(len(i[2]))), None) for t, i in info.items()}
    cal = np.unique(np.concatenate([i[0] for i in info.values()]))
    M, ix = np.full((len(tk), len(cal)), np.nan), {}
    for k, t in enumerate(tk):
        dates, sig, _, s0, b = info[t]
        start = (sig != 0) & (sig != np.r_[0.0, sig[:-1]])
        ix[t] = np.searchsorted(cal, dates)
        M[k, ix[t]] = np.where(start, sila(ev, t, sp, q, s0, b), np.nan)
    R = rang(M, nov["rang_tip"])
    out = {}
    for k, t in enumerate(tk):
        r, sg = R[k, ix[t]], info[t][2]
        out[t] = (ves(sg, rho, ks, mnozhitel(r, mt)), ~(np.isfinite(M[k, ix[t]]) & ~np.isfinite(r)))
    return out
