"""Supertrend (fastbt/signals/s34.py) на ряду с NaN прогрева (относительный ряд sektor/rynok): старт от первого конечного бара + period.

Баг: старт с фиксированного индекса `period` → при 54 NaN в начале весь ST = NaN и у S4 нет сигналов (в syroy те же точки — сотни).
Эталон — медленная чистая реализация на питоне (формулы calc_supertrend c22a4b3, ATR = pandas rolling.mean).
Причинность: ST[t] не зависит от баров > t; мутации «старт от середины ряда» и «засев будущим» ловятся."""
import os

import numpy as np
import pandas as pd
import pytest

from fastbt.signals import s34

NAN54 = 54


def ryad(n=600, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    h, l = c * (1 + rng.uniform(0, 0.01, n)), c * (1 - rng.uniform(0, 0.01, n))
    return h, l, c


def s_nan(k=NAN54, **kw):
    h, l, c = (x.copy() for x in ryad(**kw))
    h[:k] = l[:k] = c[:k] = np.nan
    return h, l, c


def etalon(h, l, c, period, mult):
    """Медленный путь: ST на срезе с первого конечного бара, NaN до f+period."""
    ok = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    f = int(np.argmax(ok))
    h, l, c, n = h[f:], l[f:], c[f:], len(c) - f
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = pd.Series(tr).rolling(period).mean().to_numpy()
    st = np.full(n, np.nan)
    fu, fl, d = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    bu, bl = (h[period] + l[period]) / 2 + mult * atr[period], (h[period] + l[period]) / 2 - mult * atr[period]
    fu[period], fl[period], d[period] = bu, bl, (1.0 if c[period] > bu else -1.0)
    st[period] = fl[period] if d[period] == 1 else fu[period]
    for t in range(period + 1, n):
        bu, bl = (h[t] + l[t]) / 2 + mult * atr[t], (h[t] + l[t]) / 2 - mult * atr[t]
        fu[t] = bu if (bu < fu[t - 1] or c[t - 1] > fu[t - 1]) else fu[t - 1]
        fl[t] = bl if (bl > fl[t - 1] or c[t - 1] < fl[t - 1]) else fl[t - 1]
        d[t] = (-1.0 if c[t] < fl[t] else 1.0) if d[t - 1] == 1 else (1.0 if c[t] > fu[t] else -1.0)
        st[t] = fl[t] if d[t] == 1 else fu[t]
    return np.concatenate([np.full(f, np.nan), st])


def sig_dney(h, l, c, period=14, mult=3.0):
    """Число сигнал-дней S4 (вход/смена знака ST) без фильтров: пересечение close через ST."""
    st = s34.supertrend(h, l, c, period, mult)
    ok = np.isfinite(st[1:]) & np.isfinite(st[:-1])
    up = (c[1:] > st[1:]) != (c[:-1] > st[:-1])
    return int((ok & up).sum())


@pytest.mark.parametrize("period,mult", [(10, 2.0), (14, 3.0), (20, 3.5)])
def test_54_nan_dayut_signaly(period, mult):
    h, l, c = s_nan()
    st = s34.supertrend(h, l, c, period, mult)
    assert np.isfinite(st[NAN54 + period:]).all() and np.isnan(st[:NAN54 + period]).all()
    assert sig_dney(h, l, c, period, mult) > 0


@pytest.mark.parametrize("period,mult", [(10, 2.0), (14, 3.0), (20, 3.5)])
def test_pobaytno_s_medlennym_putyom(period, mult):
    h, l, c = s_nan()
    assert s34.supertrend(h, l, c, period, mult).tobytes() == etalon(h, l, c, period, mult).tobytes()


def test_konechnyj_start_pobaytno_prezhnee():
    h, l, c = ryad()
    st = s34.supertrend(h, l, c, 14, 3.0)
    assert st.tobytes() == s34._supertrend0(h, l, c, 14, 3.0).tobytes()
    assert np.isnan(st[:14]).all() and np.isfinite(st[14:]).all()


def test_prichinnost_buduschee_ne_vlijaet():
    h, l, c = s_nan()
    full = s34.supertrend(h, l, c, 14, 3.0)
    for cut in (150, 300, 450):
        h2, l2, c2 = h.copy(), l.copy(), c.copy()
        h2[cut:], l2[cut:], c2[cut:] = h[cut:] * 1.7, l[cut:] * 0.4, c[cut:] * 1.3
        part = s34.supertrend(h2, l2, c2, 14, 3.0)
        assert part[:cut].tobytes() == full[:cut].tobytes()


def test_mutaciya_start_ot_serediny_lovitsya():
    """Мутация: ST стартует с середины ряда (индекс 300) — расходится с эталоном и ломается на причинности/покрытии."""
    h, l, c = s_nan()
    bad = np.full(len(c), np.nan)
    bad[300:] = s34._supertrend0(h[300:], l[300:], c[300:], 14, 3.0)
    assert bad.tobytes() != s34.supertrend(h, l, c, 14, 3.0).tobytes()
    assert np.isnan(bad[NAN54 + 14:300]).all() and not np.isnan(s34.supertrend(h, l, c, 14, 3.0)[NAN54 + 14:300]).any()


def test_mutaciya_zasev_budushchim_lovitsya():
    """Мутация: засев первым конечным баром из будущего (ST старта берётся по последнему бару) — причинность падает."""
    h, l, c = s_nan()
    ok = s34.supertrend(h, l, c, 14, 3.0)
    h2, l2, c2 = h[::-1].copy(), l[::-1].copy(), c[::-1].copy()
    rev = s34.supertrend(h2, l2, c2, 14, 3.0)[::-1]
    assert rev[100:].tobytes() != ok[100:].tobytes()


def test_vse_nan_ryad():
    n = 80
    x = np.full(n, np.nan)
    assert np.isnan(s34.supertrend(x, x, x, 14, 3.0)).all()


def test_syroy_ravno_otnositelnomu_s_nan_na_konechnom_otrezke():
    """Отрезок после прогрева даёт ST, равный ST отдельного ряда этого отрезка (старт от первого конечного бара)."""
    h, l, c = s_nan()
    a = s34.supertrend(h, l, c, 14, 3.0)[NAN54:]
    b = s34.supertrend(h[NAN54:], l[NAN54:], c[NAN54:], 14, 3.0)
    assert a.tobytes() == b.tobytes()


# ---------------- реальные данные ≤ 2021: S4 + sektor ----------------
MAIN = os.environ.get("FASTBT_MAIN")


def _est():
    from fastbt import daily_1850
    return bool(MAIN) and daily_1850.out_path().exists()


@pytest.mark.skipif(not _est(), reason="нет FASTBT_MAIN / дневного ряда 18:50")
def test_real_s4_sektor_signal_dni_bolshe_nulya():
    from fastbt import etap1, istochniki_v6
    from fastbt.eval_v6_osi import rel_td_ind
    inp = etap1.build_inputs("S4", period="train4")
    tik = list(inp.sk.tickers)[:5]
    ist = istochniki_v6.zagruzit("train4", tik)
    res = {}
    for t in tik:
        td, _ = inp.sk._td[t]
        tr, _ = rel_td_ind(ist.rel(t, "sektor"), td, True)
        res[t] = (sig_dney(td.high, td.low, td.close), sig_dney(tr.high, tr.low, tr.close), int(np.isnan(tr.close).sum()))
    print(res)
    assert all(v[1] > 0 for v in res.values()), res
