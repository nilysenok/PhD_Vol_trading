"""Кросс-секционные книги S7–S9, голая идея (Ж-093 п. 4, Ж-095 1.1/2.2, Ж-096 п. 2/5б). Только центр, train4.

S7 разворот: x = r_i(d−4…d) − r_IMOEX(d−4…d), скор = −(2·pct − 1). S8 моментум 12−1: x = log(c[d−21]/c[d−252]),
скор = 2·pct − 1. S9 сила к сектору: x = r_i(d−62…d) − r_сектора, скор = 2·pct − 1. pct — перцентиль ранга среди тикеров
сечения на d (только данные до 18:50 d). Книга всегда нейтральна по стоимости: сумма длинных ног = сумма коротких = 0,5
капитала, внутри стороны вес ∝ |скор|/σ_i (ставки равного риска, σ = RV22). Входы и выходы — open 10:00: решение d →
вход open[d+LAG], выход open[d+LAG+τ]; транши 1/n по одному каждые τ/n дней (Jegadeesh–Titman).
Издержки: комиссия KOM с оборота, плата за шорт ZAEM — только за ночь (формула fastbt/zaem_v2: вес × календарные дни / 365),
занятый капитал платит ставку (свободный остаётся под ставкой, избыток над ней = pnl − ставка × занятое).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KOM, ZAEM, DNEI, LAG = 0.0004, 0.05, 365.0, 1
POROG, V_SEKTORE, MIN_TIK = 0.5, 3, 6
TAU = {"S7": (5, 5), "S8": (21, 3), "S9": (21, 3)}      # (τ, число траншей) — проект архитектора v3, центр


def pct(x: pd.DataFrame) -> pd.DataFrame:
    """Перцентиль ранга в сечении на каждый день, [0; 1]; меньше MIN_TIK наблюдений — NaN (входа нет)."""
    n = x.notna().sum(axis=1)
    p = x.rank(axis=1).sub(1).div((n - 1).where(n > 1), axis=0)
    return p.where(n >= MIN_TIK, np.nan)


def skor_s7(c: pd.DataFrame, imoex: pd.Series) -> pd.DataFrame:
    x = (c / c.shift(5) - 1).sub(imoex / imoex.shift(5) - 1, axis=0)
    return -(2 * pct(x) - 1)


def skor_s8(c: pd.DataFrame) -> pd.DataFrame:
    return 2 * pct(np.log(c.shift(21) / c.shift(252))) - 1


def sektor_na_d(karta: pd.DataFrame, dates, tickers) -> np.ndarray:
    """Метки сектора [T, N] по состоянию на каждый день: запись (ticker, indeks, s) действует с даты s (point-in-time)."""
    lab = np.full((len(dates), len(tickers)), "", dtype=object)
    d = pd.DatetimeIndex(dates)
    for _, r in karta.sort_values("s").iterrows():
        if r["ticker"] in tickers:
            lab[d >= pd.Timestamp(r["s"]), list(tickers).index(r["ticker"])] = r["indeks"]
    return lab


def sektor_ryad(r: pd.DataFrame, metki: np.ndarray, rm: pd.Series) -> np.ndarray:
    """Доходность сектора тикера = среднее по ОСТАЛЬНЫМ тикерам сектора на день; нет коллег — IMOEX (AFLT, LSRG, ...)."""
    R = r.to_numpy()
    out = np.repeat(rm.to_numpy()[:, None], R.shape[1], axis=1)
    for lab in {x for x in metki.ravel() if x}:
        M = metki == lab
        ok = M & ~np.isnan(R)
        rr, c = np.where(ok, R, 0.0), ok.sum(axis=1, keepdims=True)
        out = np.where(M & (c - ok >= 1), (rr.sum(axis=1, keepdims=True) - rr) / np.maximum(c - ok, 1), out)
    return out


def skor_s9(c: pd.DataFrame, imoex: pd.Series, metki: np.ndarray) -> pd.DataFrame:
    r, rm = c / c.shift(62) - 1, imoex / imoex.shift(62) - 1
    x = pd.DataFrame(r.to_numpy() - sektor_ryad(r, metki, rm), index=c.index, columns=c.columns)
    return 2 * pct(x) - 1


def k2_skor(skor: np.ndarray, lag: int) -> np.ndarray:
    """Контроль к2: скор циклически сдвинут по времени на lag дней (сечения и распределение целы, привязка ко времени — нет)."""
    return np.roll(skor, lag, axis=0)


def vesa(skor: np.ndarray, sigma: np.ndarray, metki: np.ndarray, ok: np.ndarray, porog: float = POROG,
         cap: int = V_SEKTORE) -> np.ndarray:
    """W[T, N]: лонг +, шорт −; сумма длинных = сумма коротких = 0,5. Нет ног с одной из сторон — строка нулевая."""
    W = np.zeros(skor.shape)
    for t in range(len(skor)):
        s, side = skor[t], {}
        good = np.isfinite(s) & np.isfinite(sigma[t]) & (sigma[t] > 0) & ok[t]
        for sg in (1, -1):
            idx = [i for i in np.flatnonzero(good & (sg * s >= porog))]
            idx.sort(key=lambda i: -abs(s[i]))
            used, keep = {}, []
            for i in idx:                                                   # ≤ cap ставок одной стороны в секторе
                if used.get(metki[t, i], 0) < cap:
                    used[metki[t, i]] = used.get(metki[t, i], 0) + 1
                    keep.append(i)
            side[sg] = np.array(keep, dtype=int)
        if len(side[1]) and len(side[-1]):
            for sg, idx in side.items():
                raw = np.abs(s[idx]) / sigma[t, idx]
                W[t, idx] = sg * 0.5 * raw / raw.sum()
    return W


def kniga(W, P, D, dates, rate, tau: int, n_tr: int, i0: int, kom: float = KOM, zaem: float = None) -> dict:
    """Дневные ряды книги. Транш: решение d (i0, i0+τ/n, …) → вход P[d+LAG], выход P[d+LAG+τ] (на конце данных — закрытие
    по последнему open с комиссией). День t' = open t' → open t'+1. D — дивиденд после НДФЛ на ex-день (в рублях)."""
    zaem = ZAEM if zaem is None else zaem
    T = len(P)
    gap = np.append(np.diff(np.asarray(dates, "datetime64[D]")).astype(float), 1.0)
    z = lambda: np.zeros(T)                                                  # noqa: E731
    o = dict(pnl=z(), kom=z(), zaem=z(), gross=z(), lon=z(), sht=z(), vhody=z())
    for d in range(i0, T, tau // n_tr):
        e, x = d + LAG, min(d + LAG + tau, T - 1)
        w = W[d] / n_tr
        nz = np.flatnonzero(w)
        if e >= x or not len(nz):
            continue
        seg = P[e:x + 1][:, nz]
        o["pnl"][e:x] += ((seg[1:] + D[e + 1:x + 1][:, nz] - seg[:-1]) / seg[0]) @ w[nz]
        o["kom"][e] += kom * np.abs(w[nz]).sum()
        o["kom"][x - 1] += kom * (np.abs(w[nz]) * seg[-1] / seg[0]).sum()
        lo, sh = w[nz][w[nz] > 0].sum(), -w[nz][w[nz] < 0].sum()
        o["lon"][e:x] += lo
        o["sht"][e:x] += sh
        o["gross"][e:x] += lo + sh
        o["zaem"][e:x] += zaem * sh * gap[e:x] / DNEI
        o["vhody"][e] += len(nz)
    o["stavka"] = np.asarray(rate) * o["gross"] * gap / DNEI
    o["excess"] = o["pnl"] - o["kom"] - o["zaem"] - o["stavka"]
    return o
