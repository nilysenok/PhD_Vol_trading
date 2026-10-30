"""Входы A′-расширенной (Л3.3, Ж-146 п. 3): ряды центров по тикерам из исполнителя голой идеи `golaya_v7_isp`.

Для центра: X[N,T] — excess одной пары (вес 1: pnl − комиссия − заём − ставка на единицу цены входа), NaN в дни без цены тикера
(Д4: не заполняется), poz[N,T] — ЗНАКОВАЯ позиция дня (+1/−1/0). Сделки не накрываются, кроме разворота по открытию (состояние
S7–S9): в день разворота позиция — новой сделки (не сумма). Вход d+1 (Д2) обеспечивает исполнитель; здесь вход не пересчитывается.
Среднее X по тикерам равно excess портфеля `golaya_v7.schet` (вес 1/N) — тест. Только d1: вход в T в книгу не идёт."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from fastbt.golaya_v7_isp import ispolnit


def _runy(D: dict, j: int, s, o, c, dts, rate, Dv, EL, ES, tau, sost) -> list:
    """Список результатов исполнителя по тикеру: состояние — два вызова на чередующихся отрезках (как golaya_v7._runy)."""
    kw = dict(dates=dts, rate=rate, D=Dv)
    if not sost:
        fl = [None if x is None else x[:, j] for x in (EL, ES)]
        return [ispolnit(s, o, c, None, "d1", tau, *fl, **kw)]
    from fastbt import progon_kontrol as pk
    res = []
    for par in (0, 1):
        m, fin = np.zeros(len(s), np.int8), np.zeros(len(s), bool)
        for k, (st, ln, val) in enumerate(pk.runs_of(s)):
            if k % 2 == par:
                m[st], fin[min(st + ln, len(s) - 1)] = int(val), st + ln < len(s)
        res.append(ispolnit(m, o, c, None, "d1", None, fin, fin, **kw))
    return res


def _znak_poz(rs: list, n: int) -> np.ndarray:
    """Знаковая позиция [n]: по сделкам в порядке входа, поздняя перекрывает раннюю (день разворота — новая сторона)."""
    p = np.zeros(n)
    for e, x, s in sorted((e, x, s) for r in rs for s, e, x, *_ in r["sdelki"]):
        p[e:x + 1] = s
    return p


def pary(D: dict, sig: np.ndarray, EL=None, ES=None, *, vhod: str = "d1", tau=None, sost: bool = False):
    """(X[N,T], poz[N,T]). sig int8[T,N] в день решения; EL/ES — флаги выхода семьи [T,N] (S1–S6) либо None."""
    if vhod != "d1":
        raise ValueError("в книгу идут центры с входом d1 (вход в T — справочно)")
    n, N = len(D["daty"]), len(D["tk"])
    if np.shape(sig) != (n, N):
        raise ValueError(f"sig должен быть [T={n}, N={N}], получено {np.shape(sig)}")
    cs = np.cumsum(D["div"].to_numpy(), axis=0)
    X, P = np.full((N, n), np.nan), np.zeros((N, n))
    for j in range(N):
        oo, cc = D["o"].iloc[:, j].to_numpy(), D["cr"].iloc[:, j].to_numpy()
        idx = np.flatnonzero((oo > 0) & (cc > 0))                       # дни без цены тикера: NaN, не 0 и не ffill
        X[j, idx] = 0.0
        if not len(idx) or not np.asarray(sig)[idx, j].any():
            continue
        sub = lambda a: None if a is None else np.asarray(a)[idx]       # noqa: E731
        rs = _runy(D, j, np.asarray(sig)[idx, j], oo[idx], cc[idx], D["daty"][idx], D["stavka"][idx],
                   np.diff(np.r_[0.0, cs[idx, j]]), None if EL is None else sub(EL), None if ES is None else sub(ES), tau, sost)
        X[j, idx] = sum(r["excess"] for r in rs)
        P[j, idx] = _znak_poz(rs, len(idx))
    return X, P


def cc_dohodnost(close, daty) -> np.ndarray:
    """IMOEX close→close дня d (close d−1 → close d), как ряды центров; считается на собственных днях ряда, затем на daty (нет дня — NaN).
    Не open→open: книга A′ держит позиции через ночь, рынок обязан быть выровнен по тому же дню (Ж-146, #892)."""
    import pandas as pd
    c = pd.Series(close).astype(float)
    return (c / c.shift(1) - 1.0).reindex(pd.DatetimeIndex(daty)).to_numpy(float)


def mix_cc(daty, katalog, dney: int = 2) -> np.ndarray:
    """MIX close→close дня d внутри активного контракта (ролл как в `mix_skleyka`: за `dney` дней до экспирации); close d−1 того же
    контракта; нет цены на d−1 или d — NaN (хедж 0, Д4), ноль — ошибка данных."""
    import pandas as pd
    from fastbt import mix_skleyka as mx
    daty = pd.DatetimeIndex(daty)
    kont = {}
    for f in sorted(Path(katalog).glob("MX*.csv")):
        df = pd.read_csv(f, usecols=["TRADEDATE", "SECID", "OPEN", "CLOSE"], parse_dates=["TRADEDATE"])
        df = df[(df["TRADEDATE"] < mx.GRANICA) & df["CLOSE"].notna() & df["OPEN"].notna()]
        if len(df):
            kont[df["SECID"].iloc[0]] = df.set_index("TRADEDATE")[["OPEN", "CLOSE"]]
    rolly = sorted((mx.data_rolla(v, daty, dney), s) for s, v in kont.items())
    out = np.full(len(daty), np.nan)
    for i in range(1, len(daty)):
        s = next((s for r, s in rolly if r >= daty[i]), None)
        if s is None:
            continue
        c1, c0 = kont[s]["CLOSE"].get(daty[i], np.nan), kont[s]["CLOSE"].get(daty[i - 1], np.nan)
        if c1 <= 0 or c0 <= 0:
            raise ValueError(f"CLOSE <= 0 у {s} вокруг {daty[i].date()}")
        out[i] = c1 / c0 - 1.0
    return out


def proverka_sdviga(X: np.ndarray, P: np.ndarray, cr: np.ndarray, lags=(-2, -1, 0, 1)) -> dict:
    """Д1 для ряда центра: corr(средний X дня d, Σ_j P[j,d]·r[j,d+lag]) по дням, r — доходность close→close тикера (cr [T,N]).
    Максимум по |corr| обязан быть на lag 0 (результат дня выровнен с позицией дня), иначе ValueError. Возврат {lag: corr}."""
    X, P, cr = np.asarray(X, float), np.asarray(P, float), np.asarray(cr, float)
    T = X.shape[1]
    r = np.vstack([np.full((1, cr.shape[1]), np.nan), cr[1:] / cr[:-1] - 1.0]).T          # r[j,d] — от close d−1 к close d
    x = np.nanmean(X, axis=0)
    out = {}
    for L in lags:
        z = np.full(T, np.nan)
        if L >= 0:
            z[:T - L] = np.nansum(P[:, :T - L] * r[:, L:], axis=0)
        else:
            z[-L:] = np.nansum(P[:, -L:] * r[:, :T + L], axis=0)
        ok = np.isfinite(x) & np.isfinite(z)
        out[L] = float(np.corrcoef(x[ok], z[ok])[0, 1]) if ok.sum() > 5 else float("nan")
    if np.isnan(list(out.values())).any() or max(out, key=lambda k: abs(out[k])) != 0:
        raise ValueError(f"Д1: максимум корреляции не на lag 0 или не определён: {out}")
    return out
