"""К-wf7: таблица сравнений §6. Синтетика; мутации № 12 (ddof=0) и № 13 (Δ из ЛВ) роняют тесты."""
import math

import numpy as np
import pandas as pd
import pytest

from fastbt import sravneniya_wf as S
from fastbt.msp import sharpe
from fastbt.s9_stats import lw_sharpe_diff

IDX = pd.date_range("2020-01-01", periods=300, freq="B")
PARY = (("a", "b"), ("a", "c"), ("b", "c"))


def _ryady():
    g = np.random.RandomState(1)
    return {k: pd.Series(g.normal(m, 0.01, len(IDX)), index=IDX) for k, m in (("a", 4e-4), ("b", 0), ("c", 1e-4))}


def _proverka_delta(t, r):
    """Δ каждой строки обязана быть разностью табличных Шарпов (ddof=1)."""
    for w in t["rows"]:
        want = sharpe(r[w["a"]]) - sharpe(r[w["b"]])
        assert math.isclose(w["delta"], want, abs_tol=1e-9), (w["delta"], want)


def test_delta_raznost_tablichnyh_sharpov():
    r = _ryady()
    t = S.sravnit(r, PARY, reps=50)
    _proverka_delta(t, r)
    lw = lw_sharpe_diff(r["a"].values, r["b"].values, 50, 5)
    assert t["rows"][0]["p_lw"] == lw[2]
    assert abs(t["rows"][0]["delta"] - lw[0]) > 1e-6  # ЛВ-оценка другая: определения различимы


def test_holm_m_parametr():
    r = _ryady()
    t = S.sravnit(r, PARY, reps=50)
    p = [x["p_lw"] for x in t["rows"]]
    assert t["m"] == 3
    assert t["rows"][int(np.argmin(p))]["p_holm"] == pytest.approx(min(1.0, 3 * min(p)))
    assert S.sravnit(r, PARY[:2], reps=50)["m"] == 2
    assert S.sravnit(r, PARY[:1], reps=50)["rows"][0]["p_holm"] == p[0]
    assert S.sravnit.__defaults__[0] == S.PARY_V2 and len(S.PARY_V2) == 5


def test_format_i_oshibki():
    r = _ryady()
    t = S.sravnit(r, PARY[:1], reps=20)
    assert t["v"] == "1" and set(t["rows"][0]) >= {"delta", "p_lw", "p_holm", "otkl", "n_dney"}
    with pytest.raises(ValueError):  # пустая лента
        S.sravnit({"a": r["a"].iloc[:0], "b": r["b"]}, PARY[:1], reps=20)
    with pytest.raises(ValueError):  # нулевая дисперсия
        S.sravnit({"a": r["a"], "b": r["b"] * 0}, PARY[:1], reps=20)
    with pytest.raises(ValueError):  # нет общих дней
        S.sravnit({"a": r["a"].iloc[:100], "b": r["b"].iloc[150:]}, PARY[:1], reps=20)
    with pytest.raises(ValueError):
        S.sravnit(r, (), reps=20)
    with pytest.raises(KeyError):
        S.sravnit(r, (("a", "zzz"),), reps=20)


def test_znak_delta_pri_perestanovke():
    r = _ryady()
    d1 = S.sravnit(r, (("a", "b"),), reps=20)["rows"][0]["delta"]
    d2 = S.sravnit(r, (("b", "a"),), reps=20)["rows"][0]["delta"]
    assert d1 > 0 and d1 == pytest.approx(-d2)


def test_mutaciya_ddof0_lovitsya(monkeypatch):
    r = _ryady()
    monkeypatch.setattr(S, "sr_table", lambda x: float(np.mean(x) / np.std(x, ddof=0) * math.sqrt(252)))
    with pytest.raises(AssertionError):
        _proverka_delta(S.sravnit(r, PARY[:1], reps=20), r)


def test_mutaciya_delta_iz_lw_lovitsya():
    r = _ryady()
    t = S.sravnit(r, PARY[:1], reps=20)
    w = t["rows"][0]
    w["delta"] = lw_sharpe_diff(r["a"].values, r["b"].values, 20, 5)[0]
    with pytest.raises(AssertionError):
        _proverka_delta(t, r)
