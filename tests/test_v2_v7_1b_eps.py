"""V7 лист 1b, G21: ε метки предобучения = 0,01·медиана положительных r² баров обучающих дней (решение [V7] класса B), только сессии < 01.01.Y.
Реальный путь `train_encoder` (мета `eps`). Мутация «ε по всей выборке» (отбор строк без границы года) обязана ловиться."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_1e_net as NET, v7_1e_pre as P, v7_1e_syn as SYN

Y = 2020
KW = dict(max_epochs=1, min_rows=30)


@pytest.fixture(scope="module")
def pack():
    return SYN.synth_pack(n=130, W=5, seed=0, start="2019-08-01")


def fresh(pack):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pack.items()}


def test_eps_known_answer_median_of_positive_only():
    r2 = np.zeros((4, NET.SLOTS))
    r2[0, :3] = [1.0, 4.0, 9.0]                                  # положительные: 1, 4, 9 -> медиана 4; нули в медиану не входят
    assert P.eps_pos(dict(r2=r2), np.arange(4)) == pytest.approx(0.04)
    assert P.eps_pos(dict(r2=np.full((2, NET.SLOTS), np.nan)), np.arange(2)) == P.EPS_REL      # нет положительных
    assert P.eps_pos(dict(r2=r2), np.array([1, 2])) == P.EPS_REL                                # строки без положительных


def test_train_eps_equals_formula_on_training_days(pack):
    m = P.train_encoder("tcn", 5, Y, 0, pack, **KW)
    rows = P.train_rows(pack, Y)
    v = pack["r2"][rows]
    assert m.meta["eps"] == pytest.approx(0.01 * np.median(v[np.isfinite(v) & (v > 0)]), rel=1e-12)


def test_eps_ignores_bars_from_year_y(pack):
    base = P.train_encoder("tcn", 5, Y, 0, pack, **KW).meta["eps"]
    pk = fresh(pack)
    later = pd.DatetimeIndex(pk["days"]) >= pd.Timestamp(Y, 1, 1)
    pk["r2"][later] = np.where(np.isfinite(pk["r2"][later]), 1e3, np.nan)
    assert P.train_encoder("tcn", 5, Y, 0, pk, **KW).meta["eps"] == base


def test_mutation_eps_over_whole_sample_is_caught(pack, monkeypatch):
    pk = fresh(pack)
    later = pd.DatetimeIndex(pk["days"]) >= pd.Timestamp(Y, 1, 1)
    pk["r2"][later] = np.where(np.isfinite(pk["r2"][later]), 1e3, np.nan)
    honest = P.train_encoder("tcn", 5, Y, 0, pk, **KW).meta["eps"]
    monkeypatch.setattr(P, "eps_pos", lambda p, rows: P.EPS_POS_REL * float(np.median(p["r2"][np.isfinite(p["r2"]) & (p["r2"] > 0)])))      # мутант: вся выборка
    assert P.train_encoder("tcn", 5, Y, 0, pk, **KW).meta["eps"] != honest
