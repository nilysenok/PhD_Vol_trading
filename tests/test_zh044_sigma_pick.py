"""Ж-044: тесты sigma126 и pick_best (zh043_lib). Синтетика, данных не читает; мутации — через monkeypatch / правку исходника функции."""
import inspect
import math
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [".", "scripts"]
import zh043_lib as L                                   # noqa: E402
import zh043_itog as I                                  # noqa: E402


def mutated(mod, name, old, new):
    """Копия функции модуля с подложенной заменой в исходнике (старый текст обязан присутствовать)."""
    src = textwrap.dedent(inspect.getsource(getattr(mod, name)))
    assert old in src, old
    ns = dict(vars(mod))
    exec("from __future__ import annotations\n" + src.replace(old, new), ns)
    return ns[name]


def _periodic(n_rep=4, seed=1):
    base = np.random.default_rng(seed).normal(0.001, 0.01, 126)
    x = np.tile(base, n_rep)
    return x[: len(x) // 2], x[len(x) // 2:]


def _ref_sigma(x18, x19):
    s = pd.Series(np.concatenate([x18, x19]))
    r = [s[i:i + 126].mean() / s[i:i + 126].std(ddof=1) * math.sqrt(252) for i in range(len(s) - 125)]
    return float(np.std(r, ddof=1))


def test_window_is_126_trading_days():
    assert L.WIN == 126


def test_sigma126_zero_when_rolling_sharpe_constant():
    """Ряд с периодом 126: любое окно из 126 дней содержит те же значения -> скользящий Шарп постоянен -> σ = 0."""
    assert L.sigma126(*_periodic()) == pytest.approx(0.0, abs=1e-9)


def test_sigma126_mutation_window_63_caught(monkeypatch):
    monkeypatch.setattr(L, "WIN", 63)
    assert L.sigma126(*_periodic()) > 1e-3


def test_sigma126_matches_independent_formula():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.0005, 0.01, 250), rng.normal(0.0008, 0.012, 252)
    assert L.sigma126(a, b) == pytest.approx(_ref_sigma(a, b), rel=1e-9)


def test_sigma126_mutation_ddof0_caught():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.0005, 0.01, 250), rng.normal(0.0008, 0.012, 252)
    ref = _ref_sigma(a, b)
    bad_outer = mutated(L, "sigma126", "r.std(ddof=1)", "r.std(ddof=0)")
    bad_inner = mutated(L, "sigma126", "rolling(WIN).std(ddof=1)", "rolling(WIN).std(ddof=0)")
    assert bad_outer(a, b) != pytest.approx(ref, rel=1e-6)
    assert bad_inner(a, b) != pytest.approx(ref, rel=1e-6)


def test_sigma126_annual_scale_sqrt252():
    """Годовая шкала: σ растёт ровно в √252 раз по сравнению с дневным Шарпом."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.0005, 0.01, 250), rng.normal(0.0008, 0.012, 252)
    bad = mutated(L, "sigma126", "math.sqrt(252)", "1.0")
    assert L.sigma126(a, b) / bad(a, b) == pytest.approx(math.sqrt(252), rel=1e-9)


def test_sigma126_short_series_nan():
    assert math.isnan(L.sigma126(np.zeros(50), np.zeros(50)))


def _rows(*t):
    return [dict(score=s, obj=s, plato=p, sigma126=g, name=n) for n, s, p, g in t]


def test_pick_best_large_gap_picks_higher_target_even_if_noisy():
    r = _rows(("a", 1.00, 1.0, 5.0), ("b", 0.90, 0.9, 0.1))          # разница 0,10 > 0,05
    assert L.pick_best(r)["name"] == "a"


def test_pick_best_tie_within_005_picks_lower_sigma():
    r = _rows(("a", 1.00, 1.0, 5.0), ("b", 0.96, 0.9, 0.1))          # разница 0,04 <= 0,05
    assert L.pick_best(r)["name"] == "b"


def test_pick_best_tie_ignores_sigma_of_clearly_worse_point():
    r = _rows(("a", 1.00, 1.0, 5.0), ("b", 0.94, 0.9, 0.0), ("c", 0.98, 0.9, 2.0))
    assert L.pick_best(r)["name"] == "c"


def test_pick_best_nan_score_and_empty():
    assert L.pick_best([]) is None
    assert L.pick_best(_rows(("a", float("nan"), 1.0, 1.0))) is None
    r = _rows(("a", float("nan"), 9.0, 0.0), ("b", 0.5, 0.5, 3.0))
    assert L.pick_best(r)["name"] == "b"


def test_pick_best_nan_sigma_loses_tie():
    r = _rows(("a", 1.00, 1.0, float("nan")), ("b", 0.98, 0.9, 7.0))
    assert L.pick_best(r)["name"] == "b"


def test_pick_best_mutation_tolerance_caught(monkeypatch):
    """Мутация: допуск 0,5 вместо 0,05 — точка с отрывом 0,10 ошибочно уходит в ничью."""
    r = _rows(("a", 1.00, 1.0, 5.0), ("b", 0.90, 0.9, 0.1))
    assert L.pick_best(r)["name"] == "a"
    assert L.pick_best(r, tol=0.5)["name"] == "b"
    monkeypatch.setattr(L, "TIE", 0.5)
    assert mutated(L, "pick_best", "tol: float = TIE", "tol: float = 0.5")(r)["name"] == "b"


def test_pick_best_mutation_max_sigma_caught():
    r = _rows(("a", 1.00, 1.0, 5.0), ("b", 0.96, 0.9, 0.1))
    bad = mutated(L, "pick_best", "key=lambda r: (r[\"sigma126\"]", "key=lambda r: (-r[\"sigma126\"]")
    assert bad(r)["name"] == "a" and L.pick_best(r)["name"] == "b"


def test_tie_constant_is_005():
    assert L.TIE == 0.05


def test_itog_compares_by_target_not_plato():
    """Толкование Ж-043 п. 4: ничья и выбор — по целевой точки (`obj`), а не по плато."""
    src = inspect.getsource(I.itog)
    assert 'pick_best(full, "plato")' not in src and 'pick_best(full, "obj")' in src
