"""Блок ранней остановки не входит в обучение деревьев (замечание проверяющего этапа 2).

Перехват аргументов `fit`: строки обучения и ES не пересекаются, между ними эмбарго не меньше одной пары, ES — ровно 252 строки.
Проверяется s2 (`boost.run_block`) и s3 (`gen.run_design`). Мутация `tr = union(tr, es)` после `split_pairs` обязана ловиться.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import gen  # noqa: E402

ID = 10        # столбец dow используется как идентификатор строки (в cols входит, в CORE не входит)
N, T = 700, 650


def synth():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 12))
    X[:, ID] = np.arange(N)
    rv = np.exp(rng.normal(size=N))
    return ft.Fp(np.datetime64("2018-01-01") + np.arange(N), np.zeros(N, int), rv, X)


def capture(monkeypatch):
    got = {}
    orig = boost.fit

    def spy(spec, depth, delta, Xtr, ytr, Xes, yes):
        got["tr"], got["es"] = Xtr[:, ID].astype(int), Xes[:, ID].astype(int)
        return orig(spec, depth, delta, Xtr, ytr, Xes, yes)

    monkeypatch.setattr(boost, "fit", spy)
    return got


def violations(g) -> list:
    tr, es = g["tr"], g["es"]
    v = []
    if np.intersect1d(tr, es).size:
        v.append("пересечение")
    if len(es) != boost.ES_N:
        v.append("длина ES")
    if tr.max() + 1 + boost.EMBARGO > es.min():
        v.append("нет эмбарго")
    return v


def mutate_union(monkeypatch):
    orig = boost.split_pairs
    monkeypatch.setattr(boost, "split_pairs", lambda t: (np.union1d(orig(t)[0], orig(t)[1]), orig(t)[1]))


RUNNERS = {
    "s2_run_block": lambda F: boost.run_block(F, boost.Spec("xgb", "mse", False, False), 3, 0.5, T, [T]),
    "s3_run_design": lambda F: gen.run_design(gen.Design(F.X[:, :11], F.seg, np.log(F.rv), tuple(range(11))),
                                              boost.Spec("lgbm", "mse", False, False), 3, 0.5, T, [T]),
}


@pytest.mark.parametrize("name", list(RUNNERS))
def test_es_block_disjoint_from_train_with_embargo(monkeypatch, name):
    g = capture(monkeypatch)
    RUNNERS[name](synth())
    assert violations(g) == [] and len(g["tr"]) > 0


@pytest.mark.parametrize("name", list(RUNNERS))
def test_mutation_es_block_in_train_is_caught(monkeypatch, name):
    g = capture(monkeypatch)
    mutate_union(monkeypatch)
    RUNNERS[name](synth())
    assert "пересечение" in violations(g)
