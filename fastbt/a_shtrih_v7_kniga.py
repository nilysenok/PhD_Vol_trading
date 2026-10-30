"""Л3.2 (Ж-146 п. 3, Ж-093): сборка дня A′-расширенной. Движок и `kniga_msp` не правятся — только импорт.

Вход: X[S,N,T] excess пар (стратегия × тикер, как `komponenty` в `progon_urovni.a_shtrih`), poz[S,N,T] знаковые позиции
пары (нужны сектору и плечу), сектора[N], imoex[T] (open→open), mix[T] (доходность фьючерса MIX, склейка `mix_skleyka`),
ks[T] (ключевая ставка, год. доли; None = 0). Порядок дня (строки ≥ i в σ, β, весах, выключателе не читаются):
 1. веса Sk (`vesa_dnya`: паритет с LW или равный риск; `vykl` — поровну);
 2. секторный нетто ≤ 20 %: `kniga_msp.netto` по позициям тикеров сектора (доминирующая сторона режется пропорционально);
 3. таргет 10 %, окно 21, валовое плечо ≤ 2 (`mnozhitel_vola`), маржа сверх 100 % (`marzha_v_den`);
 4. выключатель ×0,5 (`vyklyuchatel`);
 5. хедж MIX = −β·NAV·f (β окно 63 по книге до хеджа).
Комиссия (Ж-149 п. 1, одно определение — `kniga_msp.komissiya_oborota`, позиции держатся через ночь): хедж — ставка·|Δh|;
тикер — доплата сверх оплаченного центрами. Сделки центров уже вычтены в X на |ΔP_i| (P — позиция тикера до сектора,
m, f), в масштабе F·m·f дня; книга платит ставка·|w_i − w_{i−1}|, w = P·F·m·f, а доплачивается
max(0, ставка·|Δw| − ставка·|ΔP|·F·m·f) по каждому тикеру. Знак не важен (у шорта возврата нет), нижняя граница 0
(переплаченное центром не возвращается). Ряда «брутто + оборот» центра в этом входе нет — формула с вычитанием.
Приближения: (а) сектор масштабирует excess пары тем же множителем F, что и позицию тикера; (б) без poz валовая =
1 NAV, комиссии тикеров нет. «Выкл.» всех опций + `vesa="vykl"` = `a_shtrih(ravnyj_bez_targeta)` побайтно.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from fastbt.a_shtrih_v7 import vesa_dnya
from fastbt.kniga_msp import (KOMISSIYA, beta, hedzh_mix, komissiya_oborota, marzha_v_den, mnozhitel_vola,
                              vyklyuchatel)

NETTO_SEK = 0.20
VESA = ("paritet", "ravnyj_risk", "vykl")


@dataclass
class Rezultat:
    ryad: np.ndarray        # [T] excess книги после хеджа
    a: np.ndarray           # [S,T] веса Sk
    m: np.ndarray           # [T] множитель таргета
    f: np.ndarray           # [T] выключатель
    h: np.ndarray           # [T] доля NAV в MIX (минус — короткая)
    kom_hedzh: np.ndarray   # [T]
    marzha: np.ndarray      # [T]
    poz_itog: np.ndarray    # [N,T] позиции тикеров после сектора (до m, f)
    mix_nan: np.ndarray     # [T] bool: MIX неизвестен, хедж 0
    vesa_flagi: list
    kom_tik: np.ndarray     # [T] доплата за оборот тикеров сверх оплаченного центрами (Ж-149)


def _proverit(X, poz, sektora, imoex, mix, ks):
    if X.ndim != 3 or X.shape[2] == 0:
        raise ValueError("пустая лента или X не [S,N,T]")
    S, N, T = X.shape
    if poz is not None and poz.shape != X.shape:
        raise ValueError("poz должна иметь форму X")
    if sektora.shape != (N,) or imoex.shape != (T,) or mix.shape != (T,) or (ks is not None and ks.shape != (T,)):
        raise ValueError("сектора[N], imoex[T], mix[T], ks[T]")


def _vesa_dnej(X, vesa):
    """a[S,T] и флаги: на день i — по ex_sk строк < i."""
    S, _, T = X.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # пустое среднее -> NaN, как в `a_shtrih`
        ex_sk = np.nanmean(X, axis=1).T                          # [T,S]
    a, flagi = np.full((S, T), 1.0 / S), ["vykl"] * T
    if vesa == "vykl":
        return a, flagi
    for i in range(T):
        w, fl = vesa_dnya(ex_sk, i, rezhim=vesa)
        a[:, i], flagi[i] = (w if w.sum() > 0 else np.full(S, 1.0 / S)), fl
    return a, flagi


def sektor_netto(u, sektor, maks=NETTO_SEK):
    """Ж-093 б: |Σ u_i| в каждом секторе ≤ maks·капитал (u — доли NAV). Превышение: доминирующая сторона
    сектора режется пропорционально до |нетто| = maks. Не `kniga_msp.netto` (то — 30 % валовой книги)."""
    u, sektor = np.asarray(u, float), np.asarray(sektor)
    out = u.copy()
    for s in np.unique(sektor):
        idx = np.flatnonzero(sektor == s)
        net = u[idx].sum()
        if abs(net) <= maks:
            continue
        dom = np.sign(net) * u[idx] > 0
        D, O = np.abs(u[idx][dom]).sum(), np.abs(u[idx][~dom]).sum()
        out[idx[dom]] = u[idx][dom] * (O + maks) / D
    return out


def _sektor_den(P, sektora, maks):
    """Множитель тикеров [N] для одного дня: новое/старое P по `sektor_netto`."""
    new = sektor_netto(P, sektora, maks)
    return np.where(P != 0, new / np.where(P != 0, P, 1.0), 1.0)


def _pozicii(X, poz, a, sektora, sektor_netto, maks):
    """F[N,T], poz_itog[N,T], gross[T]: позиция тикера P = Σ_k a_k·poz / Σ_k a_k·n_k (n_k — присутствующие пары)."""
    S, N, T = X.shape
    F, itog, gross = np.ones((N, T)), np.zeros((N, T)), np.ones(T)
    if poz is None:
        return F, itog, gross
    M = ~np.isnan(X)
    den = (a * M.sum(1)).sum(0)
    for i in range(T):
        P = (a[:, i, None] * np.nan_to_num(poz[:, :, i]) * M[:, :, i]).sum(0) / den[i] if den[i] > 0 else np.zeros(N)
        if sektor_netto:
            F[:, i] = _sektor_den(P, sektora, maks)
        itog[:, i], gross[i] = P * F[:, i], np.abs(P * F[:, i]).sum()
    return F, itog, gross


def _knizhnyj(X, a, F):
    """A2[T] — формула `progon_urovni.a_shtrih` без изменений (побайтно при F = 1)."""
    M = ~np.isnan(X)
    num = (a * (np.nan_to_num(X) * F[None]).sum(1)).sum(0)
    den = (a * M.sum(1)).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def kom_tikerov(W, dP, Fc, stavka=KOMISSIYA):
    """[T]: Σ_j max(0, komissiya_oborota(W_j,i, W_j,i−1) − ставка·|dP_j,i|·Fc_j,i). W[N,T] — итоговые знаковые доли
    NAV, dP[N,T] — шаг позиции тикера до сектора и масштаба (центры оплатили его в X), Fc[N,T] = F·m·f."""
    W, dP, Fc = (np.asarray(v, float) for v in (W, dP, Fc))
    N, T = W.shape
    out, prev = np.zeros(T), np.zeros(N)
    oplacheno = stavka * np.abs(dP) * Fc
    for i in range(T):
        out[i] = sum(max(0.0, komissiya_oborota(W[j, i], prev[j], stavka) - oplacheno[j, i]) for j in range(N))
        prev = W[:, i]
    return out


def kniga_ryada(X, poz, sektora, imoex, mix, ks=None, *, vesa="paritet", hedzh=True, sektor_netto=True,
                tsel_vol=True, vykl=True, tsel=0.10, plecho=2.0, netto_sek=NETTO_SEK) -> Rezultat:
    if vesa not in VESA:
        raise ValueError(f"vesa {vesa!r} не из {VESA}")
    X, imoex, mix = (np.asarray(v, float) for v in (X, imoex, mix))
    poz = None if poz is None else np.asarray(poz, float)
    sektora, ks = np.asarray(sektora), None if ks is None else np.asarray(ks, float)
    _proverit(X, poz, sektora, imoex, mix, ks)
    if sektor_netto and poz is None:
        raise ValueError("секторный нетто требует знаковых позиций poz")
    T = X.shape[2]
    a, flagi = _vesa_dnej(X, vesa)
    F, itog, gross = _pozicii(X, poz, a, sektora, sektor_netto, netto_sek)
    ex_port = _knizhnyj(X, a, F)
    m, f, h = np.ones(T), np.ones(T), np.zeros(T)
    ex_kniga, kom, marzha = np.full(T, np.nan), np.zeros(T), np.zeros(T)
    mix_nan = np.isnan(mix) | np.isnan(imoex)
    for i in range(T):
        if tsel_vol:
            m[i] = mnozhitel_vola(ex_port, i, gross[i], tsel, plecho)
        ex_kniga[i] = ex_port[i] * m[i]
        if vykl:
            f[i] = vyklyuchatel(ex_kniga, i, tsel)
        if hedzh and not mix_nan[i]:
            h[i] = (hedzh_mix(beta(ex_kniga, imoex, i)) * f[i]) + 0.0
        if tsel_vol:
            marzha[i] = marzha_v_den(gross[i] * m[i] * f[i], 0.0 if ks is None else ks[i])
    ryad = ex_port.copy()
    if tsel_vol or vykl:
        ryad = ex_kniga.copy() * f - marzha
    kom_tik = np.zeros(T)
    if poz is not None and (sektor_netto or tsel_vol or vykl):
        c = m * f
        P = np.divide(itog, F, out=np.zeros_like(itog), where=F > 0)    # F = 0 только у подмены правила: центр обнулён
        kom_tik = kom_tikerov(itog * c, np.diff(P, axis=1, prepend=0.0), F * c)
        ryad = ryad - kom_tik
    if hedzh:
        h_prev = np.concatenate([[0.0], h[:-1]])
        kom = np.array([komissiya_oborota(h[i], h_prev[i]) for i in range(T)])
        ryad = ryad + np.where(mix_nan, 0.0, h * np.where(mix_nan, 0.0, mix)) - kom
    return Rezultat(ryad, a, m, f, h, kom, marzha, itog, mix_nan, flagi, kom_tik)
