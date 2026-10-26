"""2. Ранг C по окну без текущего дня: rank_t = доля s[t−W..t−1] строго меньше s_t.

s9_stats.past_rank (C1/C2/D С9) и ранг масок C основного дизайна (masks._c_masks, u02).
"""
import numpy as np
import pytest
from numba import njit

from fastbt import design, grids, masks, s9_stats
from fastbt.signals.interface import TickerData


@njit(cache=False)
def _rank_incl_current(s, W, minp):            # мутация: окно [t−W+1, t] — с текущим днём
    n = len(s)
    out = np.full(n, np.nan)
    for t in range(n):
        v = s[t]
        if v != v:
            continue
        lo = t - W + 1 if t - W + 1 > 0 else 0
        c = 0
        k = 0
        for j in range(lo, t + 1):
            x = s[j]
            if x == x:
                c += 1
                if x < v:
                    k += 1
        if c >= minp:
            out[t] = k / c
    return out


MUTATIONS = {"rank_incl_current": lambda mp: mp.setattr(s9_stats, "past_rank", _rank_incl_current)}


def test_past_rank_by_hand():
    s = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    # t=2: окно [1,3], 2 больше одного → 1/2; t=3: [1,3,2] → 3/3; t=4: [3,2,5] → 2/3
    got = s9_stats.past_rank(s, 3, 2)
    assert np.isnan(got[:2]).all()
    assert np.allclose(got[2:], [0.5, 1.0, 2 / 3])


def test_past_rank_ignores_future_and_nan():
    rng = np.random.default_rng(0)
    s = rng.lognormal(size=400)
    s[50] = np.nan
    r = s9_stats.past_rank(s.copy(), 126, 50)
    s2 = s.copy()
    s2[300:] = rng.lognormal(size=100) * 5
    r2 = s9_stats.past_rank(s2, 126, 50)
    assert np.allclose(r[:300], r2[:300], equal_nan=True)
    assert np.isnan(r[50]) and np.isnan(r[:50]).all() and np.isfinite(r[51:]).all()


def _td(n=700, seed=1):
    rng = np.random.default_rng(seed)
    sg = {h: rng.lognormal(-4, 0.4, n) for h in ("h1", "h5", "h22")}
    c = np.ones(n)
    return TickerData("ZZZ", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"),
                      c, c, c, c, c, np.zeros(n), sg)


WANT = dict(thr=0.55, lookback=126, horizon="h1", direction="OFF", term_filter=False, hysteresis=0)


def _trend_mask(dsg):
    design.apply(dsg)
    td = _td()
    rows, idx, params = masks.c_masks(td, "Trend", False)
    design.apply(())
    return rows[idx[params.index(WANT)]], td


def test_main_c_mask_rank_excludes_current_day():
    """Н1 исправлено: основной дизайн (main.yaml, c_rank_past: true) — ранг без текущего дня."""
    from fastbt import dev
    from conftest import ROOT
    d, _ = dev.load_design(ROOT / "fastbt" / "designs" / "final" / "main.yaml")
    cfg = dev.make_config(d, "dev")
    assert ("c_rank_past", True) in cfg.design
    m, td = _trend_mask(cfg.design)
    pr = s9_stats.past_rank(td.sigma["h1"].copy(), 126, max(50, 126 // 5))
    assert (m == (~np.isnan(pr) & (pr > 0.55))).all()


def test_gate_default_keeps_c22a4b3_rank():
    """Без ручки (ворота c22a4b3) ранг прежний — pandas rolling.rank с текущим днём."""
    import pandas as pd
    m, td = _trend_mask(())
    pr = pd.Series(td.sigma["h1"]).rolling(126, min_periods=50).rank(pct=True).values
    assert (m == (~np.isnan(pr) & (pr > 0.55))).all()
