"""ZD: AUC — известный ответ на синтетике (идеальный признак → 1, обратный → 0, шум → ~0,5, ничьи → 0,5); ранговая корреляция; тождество Σ D по состояниям = по годам = всего, выброшенная дата и подмена суммы его ломают; формула D."""
from __future__ import annotations

import numpy as np
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_zd_core as C, x1_zd_tab as T


def test_auc_known_answers() -> None:
    rng = np.random.default_rng(0)
    pos = rng.random(4000) < 0.4
    assert T.auc(pos.astype(float) + 0.0, pos) == 1.0                    # идеальный признак
    assert T.auc(-pos.astype(float), pos) == 0.0                         # обратный знак
    assert abs(T.auc(rng.normal(size=4000), pos) - 0.5) < 0.03           # шум
    assert T.auc(np.zeros(10), np.arange(10) < 5) == 0.5                 # все ничьи
    assert np.isnan(T.auc(np.arange(5.0), np.ones(5, bool)))             # один класс


def test_spearman_known() -> None:
    a = np.arange(20.0)
    assert T.spearman(a, a ** 3) == pytest.approx(1.0) and T.spearman(a, -a) == pytest.approx(-1.0) and np.isnan(T.spearman(a, np.ones(20)))


def test_dloss_known_answer_and_sign() -> None:
    assert C.dloss(1.0, 1.0, 1.0) == 0.0
    # RV = 1: X1 = 2 → l = 0,5 + ln 2 − 1; F = 1 (точно) → l = 0: поправка помогла, D < 0
    assert C.dloss(1.0, 1.0, 2.0) == pytest.approx(-(0.5 + np.log(2.0) - 1.0)) and C.dloss(1.0, 1.0, 2.0) < 0
    assert C.dloss(1.0, 2.0, 1.0) > 0                                    # перепутанный знак: поправка ухудшила


def mk(n: int = 400) -> tuple:
    rng = np.random.default_rng(3)
    year = np.sort(rng.integers(2018, 2027, n))
    return {"Y2": rng.normal(size=n)}, (rng.random(n) < 0.2).astype(int), year


def test_identity_holds_and_dropped_date_breaks_it() -> None:
    Ds, e, year = mk()
    assert T.identity(T.states_table(Ds, e, year), Ds) == []
    keep = np.arange(len(e)) != 17                                       # выброшенная дата в таблице при полном ряду D
    bad = T.identity(T.states_table({"Y2": Ds["Y2"][keep]}, e[keep], year[keep]), Ds)
    assert bad and "Y2" in bad[0]


def test_identity_catches_substituted_sum_and_wrong_state_split() -> None:
    Ds, e, year = mk()
    t = T.states_table(Ds, e, year)
    t2 = t.copy()
    t2.loc[(t2["state"] == "E") & (t2["period"] == "2020"), "sum_D"] += 1.0
    assert T.identity(t2, Ds)
    assert T.identity(T.states_table(Ds, 1 - e, year), Ds) == []     # другое разбиение на E/N сумму не меняет


def test_states_table_known_answer() -> None:
    D, e, year = np.array([1.0, -2.0, 3.0, 4.0]), np.array([1, 0, 1, 0]), np.array([2018, 2018, 2019, 2019])
    t = T.states_table({"Y2": D}, e, year).set_index(["state", "period"])
    assert t.loc[("E", "2018"), "sum_D"] == 1.0 and t.loc[("N", "2019"), "mean_D"] == 4.0 and t.loc[("all", "all"), "sum_D"] == 6.0 and t.loc[("E", "2022-2025"), "n"] == 0
