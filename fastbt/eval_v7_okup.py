"""V7-3: ось k_okup в оценщике v7 (Ж-153, Ж-156 п. 3: k_okup ∈ {net, 3}; 2 — добором). Миксин `OkupMixin` для `EvaluatorV7`.

k_okup = net (или оси нет) — путь v6/v7 побайтно: ось снимается с точки до v6, ключ кэша не меняется.
k_okup = k: теневой прогон точки БЕЗ маски → журнал сделок (все тикеры точки, пул) → основной прогон, где вход в день старта сигнала
разрешён по `k_okup_v7.maska_vhoda` (канон маски — там). День входа d = день после дня решения i (вход не раньше d+1, Д2): в окно идут сделки,
закрытые не позже d−1 = i (M-d1). Терциль силы дня i — по |силе| дней i−252…i−1 (строго до i; как scripts/analysis/golaya_v7_tablicy.tercil), < 30 наблюдений — пусто
(вход разрешён с флагом). Маска трогает только дни старта сигнала (позиция, открытая сигналом, не закрывается — как gate_trades).
Журнал: gross = side·(выход/вход − 1)·100 %, kom = (fee_in + fee_out)·100 %, short = 0 (платы за заём по сделке нет, ↺), частичные выходы (parent) и открытые на конце не берутся."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import k_okup_v7 as K

OKNO_TERCIL, MIN_TERCIL = 252, 30
KOLS = ["vyhod", "tercil", "uderzhanie", "gross", "kom", "short"]


def k_iz_tochki(pt: dict) -> tuple:
    """(точка без оси k_okup, k или None): 'net' и отсутствие оси — None; допустимы k из k_okup_v7.KS (2, 3); иное — ValueError."""
    if "k_okup" not in pt:
        return pt, None
    k, q = pt["k_okup"], {n: v for n, v in pt.items() if n != "k_okup"}
    if k is None or (isinstance(k, str) and k == "net"):
        return q, None
    if isinstance(k, bool) or not isinstance(k, (int, float)) or k not in K.KS:
        raise ValueError(f"k_okup: допустимо net | {' | '.join(str(x) for x in K.KS if x)}, получено {k!r}")
    return q, int(k)


def tercil(sila: np.ndarray, i: int) -> float:
    """Терциль силы дня i по окну i−252…i−1 (строго до i); < 30 положительных наблюдений или v не число — nan."""
    w = sila[max(0, i - OKNO_TERCIL):i]
    w, v = w[np.isfinite(w) & (w > 0)], sila[i]
    if len(w) < MIN_TERCIL or not np.isfinite(v):
        return np.nan
    return float(1 + (v > np.quantile(w, 1 / 3)) + (v > np.quantile(w, 2 / 3)))


def den_vhoda(dates, i: int):
    """День входа для дня решения i: следующий (Д2). Окно маски — сделки, закрытые не позже его предыдущего дня."""
    return dates[i + 1]


def zhurnal(evr, rows) -> pd.DataFrame:
    """Журнал теневых сделок по строкам `_rows` (ti, E1Rows, a, b): колонки KOLS, см. docstring модуля."""
    out = []
    for ti, x, _, b in rows:
        t = evr.sk.tickers[ti]
        dates, sila = evr.inp.meta[t]["dates"], evr._sila(t, b + 1)
        for tr in x.trades:
            if tr.parent is not None or tr.exit_day is None or tr.exit_px is None:
                continue
            e, v = pd.Timestamp(dates[tr.entry_day]), pd.Timestamp(dates[tr.exit_day])
            out.append(dict(vyhod=v, tercil=tercil(sila, tr.d_sig), uderzhanie=(v - e).days,
                            gross=tr.side * (tr.exit_px / tr.entry_px - 1.0) * 100.0, kom=(tr.fee_in + tr.fee_out) * 100.0, short=0.0))
    return pd.DataFrame(out, columns=KOLS)


class OkupMixin:
    """Подмешивается перед EvaluatorV6: `_rows` (тень → основной прогон), `_allow` (маска), evaluate_v2 (снять ось), ключ кэша."""
    _okup_k = None        # k активной точки; None — ось выключена
    _okup_zh = None       # журнал теневых сделок; None — идёт теневой прогон (маски нет)

    def _okup_key(self) -> tuple:
        return () if self._okup_k is None else (("k_okup", self._okup_k),)

    def _sila(self, t, n: int) -> np.ndarray:
        """|сила| сигнала тикера на днях 0…n−1 окна (как EvaluatorV5._ranki)."""
        s0 = self.inp.meta[t]["s0"]
        return np.abs(np.asarray(self._view(t, self.r5.sp, 0.0)[0].strength[0], float))[s0:s0 + n]

    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None):
        q, self._okup_k = k_iz_tochki(pt)
        return super().evaluate_v2(q, period, vozm)

    def _rows(self, pt, period, anchor):
        if self._okup_k is None:
            return super()._rows(pt, period, anchor)
        self._okup_zh = None
        ten = super()._rows(pt, period, anchor)                  # тень: та же точка без маски
        self._okup_zh = zhurnal(self, ten)
        try:
            return super()._rows(pt, period, anchor)
        finally:
            self._okup_zh = None

    def _allow(self, pt, t, dates, sig, f):
        a = super()._allow(pt, t, dates, sig, f)
        if self._okup_k is None or self._okup_zh is None:
            return a
        sig = np.asarray(sig, float)
        st = np.flatnonzero((sig != 0) & (sig != np.r_[0.0, sig[:-1]]))
        if not len(st):
            return a
        sila = self._sila(t, len(sig))
        vh = pd.DataFrame({"vhod": [den_vhoda(dates, int(i)) for i in st], "tercil": [tercil(sila, int(i)) for i in st]})
        a = np.array(a, bool, copy=True)
        a[st] &= K.maska_vhoda(self._okup_zh, vh, self._okup_k).razresheno.to_numpy()
        return a
