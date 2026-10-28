"""25. Подход B (σ-стопы, fastbt.approach_b): σ_med динамического удержания — только прошлое (FA2b).

fix_b_median (по умолчанию включено): σ_med года Y = медиана σ_h > 1e-6 по барам до 1 января Y,
отдельно для h1/h5/h22; σ входа — строка с датой ≤ бара входа (lvl_run/trend_run: sigma[e]).
Ответ вручную: σ_h1 = 0,01 до 2020 года и 0,05 после → σ_med(2020) = 0,01, а не медиана всего ряда;
σ_h5 = 0,02 / 0,08 → 0,02; σ_h22 = NaN до 2018, затем 0,03 / 0,09 → 0,03.
"""
import types

import numpy as np
import pandas as pd
import pytest

from fastbt import approach_b
from fastbt.config import Config
from fastbt.data import year_bounds

DATES = pd.bdate_range("2017-01-02", "2021-12-30").values.astype("datetime64[ns]")


def _unit():
    n = len(DATES)
    ys = year_bounds(DATES, 2020)[0]
    sg = {}
    for h, (a, b) in {"h1": (0.01, 0.05), "h5": (0.02, 0.08), "h22": (0.03, 0.09)}.items():
        s = np.where(np.arange(n) < ys, a, b).astype(float)
        sg[h] = s
    sg["h22"][:year_bounds(DATES, 2018)[0]] = np.nan
    z = np.zeros(n)
    return types.SimpleNamespace(
        cat="Contrarian", trend=False, n=n, dates=DATES, td=types.SimpleNamespace(sigma=sg),
        rh=z[None], rl=z[None], sigs=None, exl=None, exs=None, nxt=None, close=z, high=z, low=z,
        adx=z, atr=z, atr0=z, sdl=None, sds=None, tdl=None, tds=None, strength=None, lr=z,
        start=200)


@pytest.fixture
def captured(monkeypatch):
    got = {}

    def b_eval(*a):
        got["med3"] = np.array(a[14])
        got["sig3"] = np.array(a[13])
        a[-2][:] = 0.0

    monkeypatch.setattr(approach_b, "b_eval", b_eval)
    monkeypatch.setattr(approach_b, "_test_ensemble", lambda u, *a: np.zeros(u.n))
    return got


def test_b_median_past_only(captured):
    u = _unit()
    cfg = Config(years_bcd=(2020,), val_start=2018)
    out = approach_b.run_b(u, cfg, {2020: ([(0, 0, 0)], None)})
    assert 2020 in out
    assert np.allclose(captured["med3"], [0.01, 0.02, 0.03])
    assert np.array_equal(captured["sig3"][0], u.td.sigma["h1"])       # σ без сдвига


def test_b_median_gate_mode_whole_series(captured):
    """Режим ворот (fix_b_median=False, c22a4b3): медиана всего ряда — для контраста."""
    u = _unit()
    cfg = Config(years_bcd=(2020,), val_start=2018, fix_b_median=False)
    approach_b.run_b(u, cfg, {2020: ([(0, 0, 0)], None)})
    full = [float(np.nanmedian(u.td.sigma[h])) for h in ("h1", "h5", "h22")]
    assert np.allclose(captured["med3"], full)


MUTATIONS = {
    # медиана σ по ряду с годом Y и будущим (граница 1 янв Y потеряна)
    "b_median_future": lambda mp: mp.setattr(approach_b, "year_bounds",
                                             lambda d, y: (len(d), len(d))),
}
