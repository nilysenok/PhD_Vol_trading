"""Овернайт-разложение (Ж-103 п. 2, 4): чистые функции без данных. Используются scripts/analysis/overnajt.py и тестом.

День позиции = внутридневная часть (open 10:00 → close Н6 того же дня; у дня входа — от цены входа, у дня выхода — до цены выхода)
+ ночная (close Н6 предыдущего дня → open 10:00; дивиденд ex-дня — в ночную). Строка = предыдущий торговый день (как в exec_e).
Доходность сделки — логарифмы цен сделок, как в движке (exec_e_run._exit): сумма частей = gross движка (проверяется сверкой).
"""
from __future__ import annotations

import numpy as np

from fastbt import metriki_e as me


def razlozhit_sdelki(trades, w, o, c, div, nal, ix, n):
    """(intra[n], noch[n]) — вес × лог-доходность частей по строкам. o[i], c[i] — open 10:00 и close Н6 дня i; ix — торговые дни;
    div[i] — полный дивиденд (день гэпа без баров уходит на следующий торговый — как в движке). Сделка без выхода держится до ix[-1]."""
    jx, I, N = {i: j for j, i in enumerate(ix)}, np.zeros(n), np.zeros(n)
    dv = np.zeros(n)
    if div is not None:
        for i in np.flatnonzero(np.asarray(div, float)):
            k = next((x for x in ix if x >= i), None)
            if k is not None:
                dv[k] += float(div[i])
    for tr in trades:
        j0, jl = jx[tr.entry_day], (len(ix) - 1 if tr.exit_day is None else jx[tr.exit_day])
        wt, ref = float(w[tr.d_sig]), tr.entry_px
        for j in range(j0, jl + 1):
            i = ix[j]
            end = tr.exit_px if (tr.exit_day is not None and j == jl) else float(c[i])
            start = tr.entry_px if j == j0 else float(o[i])
            d = 0.0 if j == j0 else dv[i] * nal
            if j > 0:
                intra = tr.side * np.log(end / start)
                I[ix[j - 1]] += wt * intra
                N[ix[j - 1]] += wt * (tr.side * np.log((end + d) / ref) - intra)
            ref = float(c[i])
    return I, N


def kniga_razlozhenie(W, P, C, D, tau: int, n_tr: int, i0: int, lag: int = 1):
    """Части дневного pnl книги kross_seksiya.kniga (день t' = open t' → open t'+1): intra = (close t' − open t')/цена входа,
    noch = (open t'+1 + D − close t')/цена входа. Нет close дня — внутридневная часть 0 (вся доходность в ночь). Сумма частей = pnl книги."""
    T = len(P)
    I, N = np.zeros(T), np.zeros(T)
    for d in range(i0, T, tau // n_tr):
        e, x = d + lag, min(d + lag + tau, T - 1)
        w = W[d] / n_tr
        nz = np.flatnonzero(w)
        if e >= x or not len(nz):
            continue
        seg = P[e:x + 1][:, nz]
        cc = np.where(np.isfinite(C[e:x][:, nz]), C[e:x][:, nz], seg[:-1])
        I[e:x] += ((cc - seg[:-1]) / seg[0]) @ w[nz]
        N[e:x] += ((seg[1:] + D[e + 1:x + 1][:, nz] - cc) / seg[0]) @ w[nz]
    return I, N


def sharpe(x) -> float:
    """Шарп: ddof=1, √252 (Д7, metriki_e)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(me.sr_dnevnoy(x) * np.sqrt(252)) if len(x) > 1 else float("nan")


def dolya_noch(I, N) -> dict:
    """Доли ночной части в дисперсии и доходности валовой (I + N). Дисперсия — аддитивно: Cov(часть, сумма)/Var(сумма) (доли дают 1);
    «простая» — Var(N)/(Var(I)+Var(N)) без ковариации. Доходность: N.sum()/(I+N).sum(), при сумме ≤ 0 — nan."""
    I, N = np.asarray(I, float), np.asarray(N, float)
    tot = I + N
    v = np.var(tot, ddof=1) if len(tot) > 1 else 0.0
    vi, vn = (np.var(I, ddof=1), np.var(N, ddof=1)) if len(tot) > 1 else (0.0, 0.0)
    disp = float(np.cov(N, tot)[0, 1] / v) if v > 1e-30 else float("nan")
    prost = float(vn / (vi + vn)) if vi + vn > 1e-30 else float("nan")
    s = float(tot.sum())
    doh = float(N.sum() / s) if s > 1e-12 else float("nan")
    return dict(dolya_disp_noch=disp, dolya_disp_noch_prostaya=prost, dolya_doh_noch=doh, valovaya_doh=s)


def flag_p2(disp: float, doh: float) -> str:
    """Ж-103 п. 2: овернайт > 50 % дисперсии при < 20 % доходности → «да»; доходности нет (сумма ≤ 0) → «н/д»."""
    if not np.isfinite(disp):
        return "н/д"
    if not np.isfinite(doh):
        return "н/д" if disp > 0.5 else "нет"
    return "да" if (disp > 0.5 and doh < 0.2) else "нет"


def stroka(imya: str, sk: str, I, N, x, pnl, v_dele, god: float = 4.0) -> dict:
    """Строка разложения. I, N — ряды частей валовой доходности (без последней даты окна: день b не разложим), x — excess,
    pnl — торговая компонента (без ФДР, без ставки), v_dele — маска дней с позицией (для x, pnl полной длины)."""
    k = dolya_noch(I, N)
    d = k["dolya_disp_noch"]
    vd = np.asarray(v_dele, bool)
    return dict(kategoriya=imya, sk=sk, dolya_disp_noch=d, dolya_disp_den=1 - d, dolya_disp_noch_prostaya=k["dolya_disp_noch_prostaya"],
                dolya_doh_noch=k["dolya_doh_noch"], dolya_doh_den=1 - k["dolya_doh_noch"],
                flag_zh103_p2=flag_p2(d, k["dolya_doh_noch"]), valovaya_doh_pct_god=k["valovaya_doh"] / god * 100,
                doh_noch_pct_god=float(np.sum(N)) / god * 100, doh_den_pct_god=float(np.sum(I)) / god * 100,
                sharpe=sharpe(x), sharpe_v_dele=sharpe(np.asarray(x, float)[vd]), sharpe_bez_fdr=sharpe(pnl),
                dolya_dnej_v_dele=float(vd.mean()), n_dnej=int(len(x)))
