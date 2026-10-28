"""11. Статистика отсева: доля сигналов A, которые фильтр X (C/D) отсеял целиком, — от сигналов A
с данными X (scripts/analysis/etap3_filtr.py); отсев фазы 1 A — от конфигураций грубой сетки
(chapter4_a.otsev); для затворов С9 — доля позиций-дней A, закрытых затвором (s9.Engine.gate_stats).
"""
import numpy as np
import pandas as pd
import pytest

import etap3_filtr as ef
from fastbt import chapter4_a, s9

_orig_sum = ef.summarize


def _from_all(ep, x):
    r = _orig_sum(ep, x)
    r["доля отсеянных, %"] = 100 * r["отсеяно целиком"] / r["сигналов A"]
    return r


_orig_gs = s9.Engine.gate_stats


def _gate_all_days(self, g):
    r = _orig_gs(self, g)
    r["closed_A"] = r["closed_dev"]
    return r


MUTATIONS = {"otsev_all_signals": lambda mp: mp.setattr(ef, "summarize", _from_all),
             "otsev_gate_all_days": lambda mp: mp.setattr(s9.Engine, "gate_stats", _gate_all_days)}

N = 7
A1 = [1, 1, 0, -1, -1, 0, 1]          # 3 сигнала: [0,1] +, [3,4] −, [6] +
X1 = [1, 0, 0, 0, 0, 0, 1]            # 1-й частично, 2-й отсеян целиком, 3-й оставлен
G1 = [0.01, 0.0, 0.0, -0.01, -0.02, 0.0, 0.03]
A2 = [0, 1, 1, 0, 0, 0, 0]            # 1 сигнал, у X пропуск данных → исключён
X2 = [0, np.nan, 1, 0, 0, 0, 0]


def _frames():
    cols = pd.MultiIndex.from_tuples([("A", "S1", "AAA"), ("A", "S2", "BBB"),
                                      ("C", "S1", "AAA"), ("C", "S2", "BBB")])
    idx = pd.bdate_range("2021-03-01", periods=N)
    pos = pd.DataFrame(np.c_[A1, A2, X1, X2], idx, cols)
    gr = pd.DataFrame(np.c_[G1, np.zeros(N), np.zeros(N), np.zeros(N)], idx, cols)
    return pos, gr


def test_signal_dropout_by_hand():
    pos, gr = _frames()
    ep = ef.episodes(pos, gr, "C")
    s = ef.summarize(ep, "C")
    assert s["сигналов A"] == 4 and s["без данных X (исключены)"] == 1
    assert s["сигналов в расчёте"] == 3 and s["отсеяно целиком"] == 1
    assert np.isclose(s["доля отсеянных, %"], 100 / 3)
    assert s["сокращено частично"] == 1
    assert np.isclose(s["убыточных среди отсеянных, % (по A)"], 100.0)
    # позиции-дни A (без пары с пропуском): 5, X того же знака — 2 → 60 % без X
    assert np.isclose(s["доля позиций-дней A без X, %"], 60.0)


def test_phase1_dropout_by_hand():
    inf = lambda c, p: dict(n_coarse=c, n_sh_pos=6, n_mdd_ok=5, n_exp_ok=8, n_pass=p)
    metas = {"AAA": {"S1": {"A_info": {2019: inf(10, 4)}}},
             "BBB": {"S1": {"A_info": {2019: inf(10, 0)}}}}
    g = chapter4_a.otsev(metas).set_index(["sid", "year"])
    assert np.isclose(g.loc[("S1", 2019), "pass_share"], 4 / 20)
    assert g.loc[("S1", 2019), "n_units_fallback"] == 1


def _engine(a_rows):
    e = object.__new__(s9.Engine)
    n = len(a_rows)
    e.r = {"pair": np.zeros(n, int)}
    e.first = np.r_[True, np.zeros(n - 1, bool)]
    e.A_row = np.asarray(a_rows, float)
    return e


def test_gate_dropout_from_A_positions():
    """Н3 исправлено: closed_A — от позиций-дней A, closed_dev — от всех пар-дней."""
    a = [0, 1, 1, 1, 0, 0, -1, -1]
    g = [0, 1, 0, 1, 0, 1, 0, 1]          # закрыто 4 из 8 баров; из 5 дней в позиции A — 2
    st = s9.Engine.gate_stats(_engine(a), np.asarray(g, float))
    assert np.isclose(st["closed_dev"], 0.5)
    assert np.isclose(st["closed_A"], 2 / 5)
