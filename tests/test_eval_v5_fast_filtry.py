"""Ф-сл4b: оценщик v5 с numba-фильтрами (#629) = оценщик с эталонами (maska_atr_expand, calc_adx, rangi_v6.rang) побайтно.

Допуск 0. Точка v5 — строка ledger-вида без t_sek (r без `t_sek`), ряд портфеля x и ряд t. 50 точек каждого Sk (train4,
≤ 2019, как tests/test_eval_v5.py). Мутации № 21/22 (окно с текущим днём) ловятся на маске и на точке.
"""
import random

import numpy as np
import pytest

from fastbt import canon_v5, eval_v5, rangi_v6
from fastbt.eval_v5_osi import yakor
from fastbt.indicators import calc_adx, calc_atr
from fastbt.signals.v5_filtr import maska_atr_expand, primenit
from tests import test_eval_v5 as T


def maska_etalon(td, tip_, filtr, sig):
    """Прежний eval_v5.maska: всё через signals.v5_filtr.primenit (эталонные calc_adx, maska_atr_expand)."""
    import pandas as pd
    df = pd.DataFrame({"high": td.high, "low": td.low, "close": td.close, "volume": td.volume}, index=pd.to_datetime(td.dates))
    return primenit(df, tip_, filtr, sig=np.asarray(sig))


def _td(seed, n=500):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(size=n))
    from types import SimpleNamespace
    return SimpleNamespace(high=c + np.abs(r.normal(size=n)), low=c - np.abs(r.normal(size=n)), close=c,
                           volume=r.integers(0, 1000, n).astype(float), dates=np.arange("2015-01-01", n, dtype="datetime64[D]")[:n])


FILTRY = [dict(adx_th=25), dict(atr_lo=10), dict(adx_th=15, atr_lo=40, vol_th=0.7, dow=1), dict(atr_lo=90, squeeze_pct=40)]


@pytest.mark.parametrize("tip_", ["kontr", "trend"])
@pytest.mark.parametrize("filtr", FILTRY)
@pytest.mark.parametrize("seed", range(3))
def test_maska_sintetika_pobajtno(seed, filtr, tip_):
    td = _td(seed)
    sig = np.sign(np.random.default_rng(seed).normal(size=len(td.close))).astype(np.int8)
    a, b = eval_v5.maska(td, tip_, filtr, sig), maska_etalon(td, tip_, filtr, sig)
    assert a.dtype == b.dtype and np.array_equal(a, b) and 0 < a.sum() < len(a)


def test_maska_bez_adx_atr_ne_menyaetsya():
    td, sig = _td(7), np.zeros(500, np.int8)
    f = dict(vol_th=0.7, dow=2)
    assert np.array_equal(eval_v5.maska(td, "kontr", f, sig), maska_etalon(td, "kontr", f, sig))


def test_mutaciya_atr_s_tekushchim_dnem_lovitsya(monkeypatch):
    """№ 21/22: порог ATR по [0..d] вместо [0..d-1] — маска обязана отличаться от эталона."""
    def s_dnem(atr, pct):
        out = np.zeros(len(atr), bool)
        for d in range(len(atr)):
            w = atr[:d + 1][~np.isnan(atr[:d + 1])]
            out[d] = len(w) >= 51 and not np.isnan(atr[d]) and atr[d] > np.percentile(w, pct)
        return out
    monkeypatch.setattr(eval_v5, "maska_atr_expand_nb", s_dnem)
    td, sig = _td(1), np.zeros(500, np.int8)
    assert not np.array_equal(eval_v5.maska(td, "kontr", {"atr_lo": 30}, sig), maska_etalon(td, "kontr", {"atr_lo": 30}, sig))


def test_mutaciya_adx_v_dne_d_plus_1_lovitsya(monkeypatch):
    orig = eval_v5.calc_adx_nb
    monkeypatch.setattr(eval_v5, "calc_adx_nb", lambda h, l, c, p=14: np.r_[orig(h, l, c, p)[1:], np.nan])
    td, sig = _td(2), np.zeros(500, np.int8)
    assert not np.array_equal(eval_v5.maska(td, "trend", {"adx_th": 20}, sig), maska_etalon(td, "trend", {"adx_th": 20}, sig))


def test_etalony_sovpadayut_s_importom():
    assert maska_atr_expand is not None and calc_adx is not None and calc_atr is not None   # эталоны на месте


# ---------------- точка v5 на реальных данных train4 ----------------

def _tochki(sid, n, seed=20261030):
    """n точек v5: якорь + сигнал из середины сетки + случайные оси filtr/vhod/vyhod (в т. ч. rang_tip, m_sily)."""
    k, rng, out = canon_v5.load_kub(sid), random.Random(seed + int(sid[1])), []
    sp = T._sp5(sid)
    while len(out) < n:
        q = {**yakor(sid), **sp}
        for sec in ("filtr", "vhod", "vyhod"):
            for name, v in k[sec].items():
                if rng.random() < 0.5:
                    q[name] = rng.choice(list(v["setka"]))
        if canon_v5.canon_v5(sid, q) is not None:
            out.append(q)
    return out


def _b(a):
    """Байты ряда (nan на тех же местах = те же байты); None/не ряд — repr."""
    return np.asarray(a, float).tobytes() if isinstance(a, np.ndarray) else repr(a)


def _sravnit(sid, n, fix=None):
    from fastbt import eval_v5 as E
    _, e5 = T._ocenshchiki(sid, n_tik=99, semya="v5")        # все тикеры train4: ранг требует ≥ MIN_NABL наблюдений в окне
    pts, bystr, medl, rasho = [{**q, **(fix or {})} for q in _tochki(sid, n)], [], [], 0
    if fix is None:                                           # каждая 3-я точка — с рангом (общий / средний / тикерный)
        tips = ("obshchiy", "srednee", "tikernyj")
        pts = [{**q, "razmer_tip": "risk_sigma", "m_sily": "lin", "rang_tip": tips[i // 3 % 3]} if i % 3 == 0 else q
               for i, q in enumerate(pts)]
    for q in pts:
        r, x, t = e5.evaluate_v2(q, "train4")
        bystr.append(({k: v for k, v in r.items() if k != "t_sek"}, x, t))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "maska", maska_etalon)
        mp.setattr(E, "rang_nb", rangi_v6.rang)
        for q in pts:
            r, x, t = e5.evaluate_v2(q, "train4")
            medl.append(({k: v for k, v in r.items() if k != "t_sek"}, x, t))
    for (r1, x1, t1), (r2, x2, t2) in zip(bystr, medl):
        same = repr(r1) == repr(r2) and all(_b(a) == _b(b) for a, b in ((x1, x2), (t1, t2)))
        rasho += not same
    return rasho, sum(r["status"] == "ok" for r, _, _ in bystr)


@T.real
@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_tochka_v5_pobajtno(sid):
    rasho, n_ok = _sravnit(sid, 50)
    print(f"{sid}: точек 50, status ok {n_ok}, расхождений {rasho}")
    assert rasho == 0 and n_ok >= 10


@T.real
def test_mutaciya_rang_s_d_plus_1_lovitsya(monkeypatch):
    """Ранг по дню d+1 (сдвиг матрицы) — хотя бы одна точка S1 обязана разойтись с эталоном."""
    monkeypatch.setattr(eval_v5, "rang_nb", lambda M, tip: rangi_v6.rang(np.roll(M, -1, axis=1), tip))
    rasho, n_ok = _sravnit("S1", 8, fix=dict(razmer_tip="risk_sigma", m_sily="lin", rang_tip="obshchiy"))   # общий ранг: у тикера < 20 стартов
    assert n_ok >= 4
    assert rasho > 0
