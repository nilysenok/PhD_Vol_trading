"""Ж-212 (HS2, HS3, G6, HS13): реестр — 521 строка побайтно, лимиты, «архитектурное отличие» для C, N Холма, причинное правило выбора, подложенное «закрыто + 1»."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s5.d import stats_d
from models_v2.s8 import hs_ledger as L, hs_paths as P, hs_pit, hs_rules


@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ALL", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(L, "CHECK", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(P, "REP", tmp_path)
    return tmp_path


def test_first_521_rows_are_byte_identical_and_n_is_521(tmp_ledger):
    L.init()
    assert L.ALL.read_bytes() == L.SRC.read_bytes() and L.n_final() == 521


def test_limit_exceeded_raises_and_repeat_is_not_a_trial(tmp_ledger):
    for i in range(6):
        L.register("moe", {"i": i})
    k = L.register("moe", {"i": 0})
    assert L.used(L.load())["moe"] == 6 and k.startswith("Ж-212:moe:")
    with pytest.raises(ValueError):
        L.register("moe", {"i": 99})
    assert L.n_final() == 527


def test_total_limit_50_raises(tmp_ledger, monkeypatch):
    monkeypatch.setattr(L, "LIMITS", {k: 12 for k in L.LIMITS})
    for d in ("har_features", "residual_qlike", "component", "conditional"):
        for i in range(12):
            L.register(d, {"i": i, L.ARCH: "x"})
    L.register("moe", {"i": 1})
    L.register("moe", {"i": 2})       # 48 + 2 = 50
    assert L.used(L.load())["moe"] == 2 and sum(L.used(L.load()).values()) == L.TOTAL
    with pytest.raises(ValueError):
        L.register("moe", {"i": 3})   # 51-й trial при лимите направления 12


def test_c_without_architecture_difference_is_refused(tmp_ledger):
    with pytest.raises(ValueError):
        L.register("residual_qlike", {"cfg": "Ca"})
    with pytest.raises(ValueError):
        L.register("residual_qlike", {"cfg": "Ca", L.ARCH: "  "})
    assert L.register("residual_qlike", {"cfg": "Ca", L.ARCH: "Ж-206:...: отличие"})


def test_holm_n_must_use_full_registry_not_number_of_tests():
    p = np.array([0.001, 0.01, 0.03])
    assert stats_d.holm_n(p, 521)[0] == pytest.approx(0.521)           # 521·0,001
    assert stats_d.holm_n(p, 3)[0] == pytest.approx(0.003)             # мутация «N = число тестов» даёт другое число
    with pytest.raises(ValueError):
        stats_d.holm_n(p, 2)


def _grid(n=600):
    d = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return d, d + np.timedelta64(1, "D")


def test_causal_pick_known_answer_and_closed_plus_one_mutation():
    d, td = _grid()
    Q2 = np.column_stack([np.ones(600), np.r_[np.full(300, 1.5), np.zeros(300)]])   # сумма кандидата 1 = 450 после 300-й строки; кандидата 0 = m
    a, b = hs_rules.causal_pick(Q2, d, td), hs_rules.causal_pick(Q2, d, td, extra=1)
    assert (a[:300] == 0).all() and (a[460:] == 1).all()
    assert a[450] == 0 and b[450] == 1       # k = 450: закрыто m = 450, ничья -> раньше в списке; «закрыто + 1» даёт 451 -> другой выбор (ловится)


def test_causal_pick_ignores_future_rows():
    d, td = _grid()
    rng = np.random.default_rng(0)
    Q = rng.random((600, 3))
    k = 400
    Q2 = Q.copy()
    Q2[int(hs_pit.closed_count(d, td)[k]):] = rng.random((600 - int(hs_pit.closed_count(d, td)[k]), 3)) * 100
    assert hs_rules.causal_pick(Q, d, td)[k] == hs_rules.causal_pick(Q2, d, td)[k]
