"""5. Фильтр оборота: удержание ≥ min_hold баров, запрет входа N баров после выхода; (удержание, N)
выбирается по Шарпу на окне отбора (A — обучение до 1 янв Y).
"""
import numpy as np
import pandas as pd

from conftest import bdays
from fastbt import grids, turnover
from fastbt.config import Config
from fastbt.signals.interface import WARMUP
from fastbt.stats import sharpe_window

MUTATIONS = {"tf_off": lambda mp: mp.setattr(turnover, "tfilter",
                                             lambda raw, h, nb: np.asarray(raw, float).copy())}


def test_hold_and_ban_by_hand():
    raw = np.array([1, 1, 0, 0, 0, 1, 0, -1, -1, -1, 0, 0, 1], float)
    # вход t0; держим до held=3 (t3 — выход); запрет 2 бара: t5 (5−3=2) нельзя; t7 вход −1;
    # выход t10; t12 (12−10=2) нельзя
    want = [1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0]
    assert np.array_equal(turnover.tfilter(raw, 3, 2), want)


def test_reversal_without_ban():
    raw = np.array([1, -1, -1, -1, 0], float)
    assert np.array_equal(turnover.tfilter(raw, 2, 0), [1, 1, -1, -1, 0])


def test_apply_selects_on_train_window(monkeypatch):
    monkeypatch.setattr(grids, "TF_HOLD", [1, 3, 5])
    monkeypatch.setattr(grids, "TF_REENTRY", [0, 2, 5])
    dates = bdays("2016-01-01", 1000)
    rng = np.random.default_rng(3)

    class U:
        pass
    u = U()
    u.dates, u.lr = dates, rng.normal(0, 0.01, len(dates))
    pos = np.sign(rng.normal(size=len(dates))) * (rng.random(len(dates)) > 0.3)
    cfg = Config()
    Y = 2019
    out = turnover.apply({Y: ("p", pos)}, u, cfg, "A")
    e = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
    best = max(((sharpe_window(turnover.tfilter(pos, h, nb), u.lr, WARMUP, e, cfg.comm), -i, h, nb)
                for i, (h, nb) in enumerate((h, nb) for h in grids.TF_HOLD
                                            for nb in grids.TF_REENTRY)))
    assert out[Y][0]["tf"] == dict(min_hold=best[2], reentry_ban=best[3])
    f = out[Y][1]
    assert np.array_equal(f, turnover.tfilter(pos, best[2], best[3]))
    assert (np.diff(f) != 0).sum() < (np.diff(pos) != 0).sum()          # оборот упал
