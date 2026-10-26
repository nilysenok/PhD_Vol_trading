"""20. Сигналы A: индикатор и сигнал бара d — только из данных ≤ d (FA2b).

Проверка «префиксом»: ряд, обрезанный по бару t, обязан дать на барах ≤ t те же индикаторы
compute_base и те же сигналы, выходы, силу и уровни (S5/S6) всех шести семейств, что и полный ряд.
Любое заглядывание вперёд (сдвиг на +1, центрированное окно, high/low дня d+1) на конце префикса
даёт другое значение. Ответ вручную: окно SMA бара t = среднее close[t−w+1..t]; rh_N бара t —
максимум high[t−N..t]; pivot недели — по H/L/C предыдущей недели.
"""
import numpy as np
import pandas as pd
import pytest

from synth_fa2b import bars, prefix
from fastbt import indicators as ind_mod
from fastbt.signals.interface import get_family

SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
CUTS = (230, 287, 344, 401, 458, 559)            # бары после WARMUP=200, разные дни недели/месяца


def _base(td):
    return ind_mod.compute_base(td.close, td.high, td.low, td.open, td.volume)


def _eq(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "f":
        assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), what
    else:
        assert np.array_equal(a, b), what


@pytest.fixture(scope="module")
def full():
    td = bars()
    return td, _base(td)


def test_indicators_by_hand(full):
    td, ind = full
    t = 300
    assert np.isclose(ind_mod.calc_sma(td.close, 20)[t], td.close[t - 19:t + 1].mean())
    assert np.isclose(ind["rh_10"][t], td.high[t - 10:t + 1].max())
    assert np.isclose(ind["rl_20"][t], td.low[t - 20:t + 1].min())
    tr = np.maximum.reduce([td.high[1:] - td.low[1:], np.abs(td.high[1:] - td.close[:-1]),
                            np.abs(td.low[1:] - td.close[:-1])])
    assert np.isclose(ind["atr14"][t], tr[t - 14:t].mean())          # tr[i] здесь = TR бара i+1
    v20 = td.volume[t - 19:t + 1].mean()
    assert ind["vf_pass"][t] == (not td.volume[t] < 0.5 * v20)


def test_indicators_prefix_invariant(full):
    td, ind = full
    for t in CUTS:
        p = _base(prefix(td, t))
        for k, v in ind.items():
            _eq(p[k], np.asarray(v)[:t + 1], f"{k} @ {t}")


@pytest.mark.parametrize("sid", SIDS)
def test_signals_prefix_invariant(full, sid):
    td, ind = full
    fam = get_family(sid)
    grid = fam.params()
    ss = fam.generate(td, ind, grid)
    assert (ss.sig != 0).any(), "синтетика без сигналов — тест пустой"
    for t in CUTS:
        tp = prefix(td, t)
        sp = fam.generate(tp, _base(tp), grid)
        _eq(sp.sig, ss.sig[:, :t + 1], f"{sid} sig @ {t}")
        _eq(sp.exit_l, ss.exit_l[:, :t + 1], f"{sid} exit_l @ {t}")
        _eq(sp.exit_s, ss.exit_s[:, :t + 1], f"{sid} exit_s @ {t}")
        _eq(sp.strength, ss.strength[:, :t + 1], f"{sid} strength @ {t}")
        for k, v in ss.extra.items():
            _eq(sp.extra[k], v[:, :t + 1], f"{sid} {k} @ {t}")


def test_weekly_pivot_by_hand(full):
    """S5: P недели = (H+L+C)/3 предыдущей ISO-недели ряда."""
    td, ind = full
    fam = get_family("S5")
    grid = [g for g in fam.params() if g["pivot_period"] == "weekly"
            and g["pivot_type"] == "classic"][:1]
    P = fam.generate(td, ind, grid).extra["P"][0]
    wk = pd.DatetimeIndex(td.dates).isocalendar()
    key = (wk["year"] * 100 + wk["week"]).values
    t = 333
    prev = np.where(key == np.unique(key[key < key[t]])[-1])[0]
    hand = (td.high[prev].max() + td.low[prev].min() + td.close[prev[-1]]) / 3.0
    assert np.isclose(P[t], hand)


# ---------- мутации ----------

def _fwd(x):
    return np.r_[np.asarray(x, float)[1:], np.nan]


def _sma_center(arr, w):
    return pd.Series(arr).rolling(w, min_periods=w, center=True).mean().values


def _mut_center(mp):
    from fastbt.signals import s12, s56
    for m in (ind_mod, s12, s56):
        mp.setattr(m, "calc_sma", _sma_center)


def _mut_hl_next(mp):
    orig = ind_mod.compute_base

    def cb(close, high, low, open_arr, volume):
        nx = lambda a: np.r_[np.asarray(a)[1:], np.asarray(a)[-1]]
        return orig(close, nx(high), nx(low), open_arr, volume)
    mp.setattr(ind_mod, "compute_base", cb)


_atr = ind_mod.calc_atr
MUTATIONS = {
    "ind_shift_plus1": lambda mp: mp.setattr(ind_mod, "calc_atr",
                                             lambda h, l, c, period=14: _fwd(_atr(h, l, c, period))),
    "rolling_center": _mut_center,
    "hl_next_day": _mut_hl_next,
}
