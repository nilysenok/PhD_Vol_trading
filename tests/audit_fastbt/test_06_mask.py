"""6. Маска _year_bounds(...)[0]: окно отбора года Y кончается до 1 января Y (не до 1 января Y+1).

Config.bound / val_bounds (B/C/D, фильтр оборота), обучение A (turnover.apply, ApproachA.year)
и окна С9 (s9.Engine.vwin).
"""
import inspect

import numpy as np

from conftest import bdays
from fastbt import grids, s9, turnover
from fastbt.config import Config
from fastbt.data import year_bounds
from fastbt.stats import sharpe_window

MUTATIONS = {"mask_bound1": lambda mp: mp.setattr(Config, "bound", lambda self: 1)}

DATES = bdays("2016-01-04", 1100)


def test_year_bounds_by_hand():
    s, e = year_bounds(DATES, 2019)
    assert DATES[s - 1] < np.datetime64("2019-01-01") <= DATES[s]
    assert DATES[e - 1] < np.datetime64("2020-01-01") <= DATES[e]


def test_config_windows_end_before_test_year():
    cfg = Config()
    assert cfg.bound() == 0
    for Y in (2018, 2019):
        vs, ve = cfg.val_bounds(DATES, Y)
        assert DATES[vs] >= np.datetime64(f"{cfg.val_start}-01-01")
        assert DATES[ve - 1] < np.datetime64(f"{Y}-01-01") <= DATES[ve]


def test_turnover_choice_ignores_test_year(monkeypatch):
    """Синтетика, где год Y меняет лучший (удержание, N): выбор обязан совпасть с выбором до Y."""
    monkeypatch.setattr(grids, "TF_HOLD", [1, 5])
    monkeypatch.setattr(grids, "TF_REENTRY", [0, 5])

    class U:
        pass
    Y = 2019
    ys, ye = year_bounds(DATES, Y)
    for seed in range(200):
        rng = np.random.default_rng(seed)
        u = U()
        u.dates, u.lr = DATES, rng.normal(0, 0.01, len(DATES))
        pos = np.sign(rng.normal(size=len(DATES))) * (rng.random(len(DATES)) > 0.4)
        grid = [(h, nb) for h in grids.TF_HOLD for nb in grids.TF_REENTRY]
        pick = lambda e: max((sharpe_window(turnover.tfilter(pos, h, nb), u.lr, 200, e, 0.0005),
                              -i) for i, (h, nb) in enumerate(grid))[1]
        if pick(ys) != pick(ye):
            break
    else:
        raise AssertionError("не нашлось синтетики, различающей окна")
    h, nb = grid[-pick(ys)]
    out = turnover.apply({Y: ("p", pos)}, u, Config(), "A")
    assert out[Y][0]["tf"] == dict(min_hold=h, reentry_ban=nb)


def test_s9_and_approach_a_use_bound0():
    src = inspect.getsource(s9.Engine.__init__)
    assert "year_bounds(dates, VAL_START)[0], year_bounds(dates, ye)[0]" in src
    from fastbt.approach_a import ApproachA
    assert "year_bounds(u.dates, Y)[self.cfg.bound()]" in inspect.getsource(ApproachA.year)
