"""Сборщик входа книги МСП v5 (`msp_v5.progon`): сторона и доходность пар (Sk × тикер) из движка ТФ `tf_os_v5`.

Контракт `msp_v5` (Ж-149 п. 1, #859): строка d — решение по данным к 18:50 дня d, исполнение d+1 (Д2). Поэтому
  storona[Sk][d]  — сигнал семьи Sk на ПОСЛЕДНЕМ баре дня d (его `tf_os_v5.pozicii` сегодня не исполняет);
  R_brutto[d]     — знаковая доходность позиции +1 в день d+1: единица держится с open первого бара до close
                    последнего (Σ по барам `tf_os_v5.dohod` при pos = 1), БЕЗ комиссии; знак ставит книга (u);
  T[d]            — оборот того же дня на единицу, ≥ 0 (вход + закрытие по close = 2 на каждый день с барами).
Комиссия не внутри R: ex = u·R_brutto − ставка·Σ|u|·T (шорт платит так же, как лонг).
↺ день-ставка = «держим сторону последнего бара вчера весь день», а не внутридневной автомат `pozicii` (он сам
ловит вход по барам дня d+1; сторона и сила для МСП определены только на 18:50) · откат: R по `pozicii` (31.10).
R и T не зависят от Sk (одна и та же единица), поэтому панели [d × тикер] — как ждёт `msp_v5` (tile по Sk).
Сила и g членов здесь не строятся (вызывающий: `reyting.g_*`, сила семьи). Данные ≥ 2020 не читаются (ValueError).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import tf_os_v5 as tf

END = pd.Timestamp("2020-01-01")
STATUS = "разработка 2016-2019, до аудита; потолок, не результат"


def ryad_dnya(sig: np.ndarray, o: np.ndarray, c: np.ndarray, den: np.ndarray):
    """По барам одного тикера -> (дни, storona, R_unit, T_unit): сторона = sig на последнем баре дня;
    R_unit/T_unit — доход и оборот единичной позиции, которую держат ВЕСЬ день (все бары)."""
    g, _, turn = tf.dohod(np.ones(len(sig), np.int8), o, c, den, 0.0)
    last = np.r_[den[1:] != den[:-1], True]
    dni = pd.DatetimeIndex(den[last].astype("datetime64[ns]"))
    return (dni, pd.Series(sig[last].astype(float), index=dni),
            pd.Series(g, index=den).groupby(level=0).sum().set_axis(dni),
            pd.Series(turn, index=den).groupby(level=0).sum().set_axis(dni))


def sdvig_d_plus_1(r_unit: pd.Series, t_unit: pd.Series):
    """Строка d получает день d+1 (Д2). Последний день — NaN: исполнения нет в окне."""
    return r_unit.shift(-1), t_unit.shift(-1)


def tiker(sid: str, tf_name: str, tochka: dict, b10: pd.DataFrame, tik: str = "X"):
    """(storona, R_brutto, T) тикера по дням (всё индексировано днём решения d)."""
    f = tf.bary(b10, tf_name)
    td = tf.ticker_data(f, tik)
    ss = tf.signaly(sid, td, [tochka], tf_name)
    den = tf.den_id(td)
    dni, st, r, t = ryad_dnya(ss.sig[0], td.open, td.close, den)
    r, t = sdvig_d_plus_1(r, t)
    return st, r, t


def sobrat(sk_list: list, tikery: list, period: tuple, tf_name: str, tochki: dict, b10: dict | None = None,
           main=None) -> dict:
    """{'R_brutto': [d×тикер], 'T': [d×тикер], 'storona': {Sk: [d×тикер]}, 'status'} за дни [a, b) period.
    tochki = {Sk: точка сетки v5 семьи}; b10 = {тикер: 10-минутки} (по умолчанию `bars_agg.load_10m`, до b)."""
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    if b > END:
        raise ValueError("данные ≥ 2020 не читаются")
    from fastbt import bars_agg as ba
    R, T, S = {}, {}, {sk: {} for sk in sk_list}
    for k in tikery:
        bars = b10[k] if b10 is not None else ba.load_10m(k, str(b.date()), main)
        for sk in sk_list:
            st, r, t = tiker(sk, tf_name, tochki[sk], bars[bars["date"] < b], k)
            S[sk][k] = st.loc[a:]
            R[k], T[k] = r.loc[a:], t.loc[a:]            # r, t не зависят от Sk
    fr = lambda d: pd.DataFrame(d).sort_index()           # noqa: E731
    return dict(R_brutto=fr(R), T=fr(T), storona={sk: fr(v) for sk, v in S.items()}, status=STATUS)
