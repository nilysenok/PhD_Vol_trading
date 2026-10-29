"""Л4: уровни S5@v5 против независимо переписанных формул Р @6a7a4636 (сдвиг массивов, не группы):
calc_pivot_daily, calc_pivot_demark (strategies_walkforward.py:470-510, grid_scan_phase3.py:121-148),
gen_s5_extended (:150-178). Данные и фикстуры — из test_v5_s56.py."""
import numpy as np
import pandas as pd
import pytest

from fastbt.signals import v5_s56 as v5
from fastbt.signals.interface import expand_grid
from test_v5_s56 import TIPY, eq, ind, td  # noqa: F401  (td, ind — фикстуры)


def ref_daily(o, h, l, c, tip):
    """Формулы Р: уровни из H/L/C(/O) бара t-1 (сдвиг [1:]). Возврат P,S1,R1,S2,R2,S3,R3."""
    n = len(c); out = [np.full(n, np.nan) for _ in range(7)]
    P, S1, R1, S2, R2, S3, R3 = out
    H, L, C = h[:-1], l[:-1], c[:-1]
    rng = H - L
    if tip == "classic":
        P[1:] = (H + L + C) / 3.0
        S1[1:], R1[1:] = 2 * P[1:] - H, 2 * P[1:] - L
        S2[1:], R2[1:] = P[1:] - rng, P[1:] + rng
        S3[1:], R3[1:] = L - 2 * (H - P[1:]), H + 2 * (P[1:] - L)
    elif tip == "fibonacci":
        P[1:] = (H + L + C) / 3.0
        S1[1:], R1[1:] = P[1:] - 0.382 * rng, P[1:] + 0.382 * rng
        S2[1:], R2[1:] = P[1:] - 0.618 * rng, P[1:] + 0.618 * rng
        S3[1:], R3[1:] = P[1:] - 1.000 * rng, P[1:] + 1.000 * rng
    elif tip == "woodie":
        P[1:] = (H + L + 2 * C) / 4.0
        S1[1:], R1[1:] = 2 * P[1:] - H, 2 * P[1:] - L
        S2[1:], R2[1:] = P[1:] - rng, P[1:] + rng
        S3[1:], R3[1:] = S2[1:] - rng, R2[1:] + rng
    elif tip == "camarilla":
        P[1:] = (H + L + C) / 3.0
        for a, b, k in ((S1, R1, 12.0), (S2, R2, 6.0), (S3, R3, 4.0)):
            a[1:], b[1:] = C - 1.1 * rng / k, C + 1.1 * rng / k
    else:  # demark, цикл как calc_pivot_demark
        for i in range(1, n):
            h_, l_, c_, o_ = h[i - 1], l[i - 1], c[i - 1], o[i - 1]
            x = h_ + 2 * l_ + c_ if c_ < o_ else (2 * h_ + l_ + c_ if c_ > o_ else h_ + l_ + 2 * c_)
            P[i] = x / 4.0; S1[i] = x / 2.0 - h_; R1[i] = x / 2.0 - l_
            S2[i], R2[i] = P[i] - (h_ - l_), P[i] + (h_ - l_)
            S3[i], R3[i] = P[i] - 2 * (h_ - l_), P[i] + 2 * (h_ - l_)
    return out


def ref_period(td, per, tip):
    """Пивот по завершённому прошлому периоду: группы pandas, уровни как у дневных формул
    на агрегате периода (H=max, L=min, C=последний, O=первый), раздаются барам следующего."""
    ix = pd.DatetimeIndex(td.dates)
    key = (ix.isocalendar().year.values * 100 + ix.isocalendar().week.values) if per == "weekly" \
        else ix.year.values * 100 + ix.month.values
    df = pd.DataFrame({"k": key, "o": td.open, "h": td.high, "l": td.low, "c": td.close})
    g = df.groupby("k", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    lv = np.array(ref_daily(g.o.values, g.h.values, g.l.values, g.c.values, tip))   # 7 x ngroups
    pos = pd.Series(np.arange(len(g)), index=g.index).loc[key].values
    return [lv[j][pos] for j in range(7)]


def mapped(ref):          # P,S1,R1,S2,R2,S3,R3 -> порядок v5: P,S1,S2,S3,R1,R2,R3
    P, S1, R1, S2, R2, S3, R3 = ref
    return [P, S1, S2, S3, R1, R2, R3]


@pytest.mark.parametrize("tip", TIPY)
def test_pivot_den_ravny_formulam_R(td, tip):
    got = v5.pivots(v5._group_ids(td.dates, "den"), td.open, td.high, td.low, td.close, tip)
    for g, r in zip(got, mapped(ref_daily(td.open, td.high, td.low, td.close, tip))):
        assert eq(g, r)


@pytest.mark.parametrize("per", ["weekly", "monthly"])
@pytest.mark.parametrize("tip", TIPY)
def test_pivot_nedelya_mesyac_ravny_formulam_po_proshlomu_periodu(td, tip, per):
    name = {"weekly": "nedelya", "monthly": "mesyac"}[per]
    got = v5.pivots(v5._group_ids(td.dates, name), td.open, td.high, td.low, td.close, tip)
    for g, r in zip(got, mapped(ref_period(td, per, tip))):
        assert np.allclose(g, r, rtol=0, atol=1e-12, equal_nan=True)
    assert np.isnan(got[0][0])      # первый период: прошлого нет


def ref_s5_extended(close, atr, S, R, f, warmup=200):
    """gen_s5_extended @6a7a4636 (grid_scan_phase3.py:150-178)."""
    sig = np.zeros(len(close), np.int8)
    if f > 0:
        atr_ok = ~np.isnan(atr) & (atr > 1e-12)
        s_ok = atr_ok & ~np.isnan(S) & (np.abs(close - S) > f * atr)
        r_ok = atr_ok & ~np.isnan(R) & (np.abs(close - R) > f * atr)
        sig[s_ok & (close < S)] = 1; sig[r_ok & (close > R)] = -1
    else:
        sig[~np.isnan(S) & (close < S)] = 1; sig[~np.isnan(R) & (close > R)] = -1
    sig[:warmup] = 0
    return sig


def test_s5_uroven_atr_filtr_ravny_gen_s5_extended(td, ind):
    g = expand_grid(dict(pivot_type=TIPY, pivot_period=["den"], uroven=[1, 2, 3],
                         atr_filtr=[0, 0.5, 1.0, 1.5, 2.0], filter_set=["net"]))
    ss = v5.generate5(td, ind, g)
    total = 0
    for i, p in enumerate(g):
        _, S1, R1, S2, R2, S3, R3 = ref_daily(td.open, td.high, td.low, td.close, p["pivot_type"])
        S, R = {1: (S1, R1), 2: (S2, R2), 3: (S3, R3)}[p["uroven"]]
        want = ref_s5_extended(td.close, ind["atr14"], S, R, p["atr_filtr"])
        assert eq(ss.sig[i], want), p
        total += int((want != 0).sum())
    assert total > 500
