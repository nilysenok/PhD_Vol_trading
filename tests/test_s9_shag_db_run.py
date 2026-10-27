"""Шаг 3′ (D, B): константы, сквозной run() на синтетическом движке, справочный выбор при 0,05 %.
python3 -m pytest tests/test_s9_shag_db_run.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts" / "analysis")]
from test_s9_shag0 import _world  # noqa: E402

import s9_shag_common as C  # noqa: E402
import s9_shag_db as DB  # noqa: E402


def test_constants_pinned():
    assert C.COMM_MAIN == 0.0004                      # отбор при 0,04 %
    assert (C.COMM_MAIN, C.COMM_ALT) == (0.0004, 0.00035)          # основные: 0,04 и 0,035
    assert C.COMM_APP == (0.0005, 0.0006)                           # приложение: 0,05 и 0,06
    assert C.COMM_REF == 0.0005 and C.ALPHA == 0.05


def _eng():
    ns = _world()
    ns.g_row = lambda G: G
    return ns


def _pts(ns):
    good = (ns.A_row * ns.lr_row > 0).astype(float)           # знает знак дня: Шарп высокий, сдвиг его рушит
    noise = np.random.RandomState(4).permutation(good)
    return lambda *a: [("D1", "good", good, False, 0.3, True), ("D1", "noise", noise, False, 0.3, True)]


@pytest.fixture
def ran(monkeypatch, tmp_path):
    ns = _eng()
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(DB, "points", _pts(ns))
    comms = []
    orig = DB.run_grid
    monkeypatch.setattr(DB, "run_grid", lambda p, e, s, ser, comm, reps: (comms.append(comm), orig(p, e, s, ser, comm, reps))[1])
    DB.run("D", "naive", 50, eng=ns)
    rd = lambda n: pd.read_csv(tmp_path / n, sep=";", decimal=",", encoding="utf-8-sig")   # noqa: E731
    return comms, rd


def test_run_selects_at_004_recomputes_at_0035_and_ref_at_005(ran):
    comms, rd = ran
    assert comms == [C.COMM_MAIN, C.COMM_ALT, C.COMM_REF]
    ot = rd("shag3D_otbor_naive.csv").query("sigma == 'naive_h1'")       # второй σ-ряд — те же синтетические точки
    by = {c: ot[ot["comm_pct"].round(4) == c] for c in (0.04, 0.035)}
    assert set(by[0.04]["point"]) == set(by[0.035]["point"]) == {"good", "noise"}      # те же точки, две комиссии
    assert by[0.04].set_index("point")["sharpe"].ne(by[0.035].set_index("point")["sharpe"]).all()
    assert bool(by[0.04].set_index("point").loc["good", "passed_v2"])
    assert not bool(by[0.04].set_index("point").loc["noise", "passed_v2"])
    ust = rd("shag3D_ustoychivost_naive.csv")
    assert ust.iloc[0]["point"] == "good"                                               # выбор — по 0,04 %
    kom = rd("shag3D_komissiya_naive.csv")
    assert list(kom["comm_pct"].round(4)) == [0.04, 0.035, 0.05, 0.06] and set(kom["point"]) == {"good"}
    ref = rd("shag3D_vybor005_naive.csv")
    assert len(ref) == 1 and {"vybor_005", "sovpal", "obe_na_test"} <= set(ref.columns)


def _frame(pt, passed=True, sharpe=1.0):
    return pd.DataFrame([dict(family="D1", point=pt, sigma="s", sharpe=sharpe, passed_v2=passed)])


def test_select_ref_independent_and_flag():
    sel = C.select(_frame("a"))
    same = C.select_ref(_frame("a"), sel)
    assert same["sovpal"] is True and same["obe_na_test"] is False
    diff = C.select_ref(_frame("b"), sel)                   # при 0,05 другой выбор — обе точки на тест
    assert diff["sovpal"] is False and diff["obe_na_test"] is True and diff["vybor_005"] == "D1:b:s"
    none = C.select_ref(_frame("b", passed=False), sel)
    assert none["sovpal"] is False and none["vybor_005"] == "нет прошедших"
