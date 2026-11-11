"""W4-A, лист W3 (PIT): мутация «закрыто + 1» в DFTL, AH, HEDGE0 ловится через боевой путь `scheme`; порча целей с t⁺ > d не меняет π дня d;
мутации, видящие будущее, ловятся. Подмены — только monkeypatch (класс № 43)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w3_core as C

NAMES = ("DFTL126", "AH", "HEDGE0")
K = (60, 150)


def world(n: int = 240, gap: int = 3, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = np.datetime64("2020-01-06") + np.arange(n)
    M = np.exp(0.1 * rng.standard_normal(n))
    return pd.DataFrame(dict(d=d.astype("datetime64[ns]"), t_plus=(d + gap).astype("datetime64[ns]"), rv=M * np.exp(0.25 * rng.standard_normal(n)),
                             M=M, C=M * np.exp(0.3 * rng.choice([-1.0, 1.0], n)), s=rng.integers(0, 2, n)))   # C вокруг M в обе стороны: нет дрейфа, честный π не насыщается


def pis(x: pd.DataFrame) -> dict:
    L = C.grid_losses(x["rv"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy())          # боевой путь: потери по сетке из rv, затем scheme
    return {nm: C.scheme(nm, x, L) for nm in NAMES}


def corrupt(x: pd.DataFrame, mask) -> pd.DataFrame:
    y = x.copy()
    y.loc[mask, "rv"] = y.loc[mask, "rv"] * 1e3
    return y


def future_mask(x: pd.DataFrame, k: int):
    return (x["t_plus"] > x["d"].iloc[k]).to_numpy()


def unchanged_to(k: int, a: dict, b: dict) -> dict:
    return {nm: bool(np.array_equal(a[nm][: k + 1], b[nm][: k + 1])) for nm in NAMES}


@pytest.mark.parametrize("k", K)
def test_corrupting_targets_after_d_does_not_change_pi_of_day_d(k) -> None:
    x = world()
    base, bad = pis(x), pis(corrupt(x, future_mask(x, k)))
    assert all(unchanged_to(k, base, bad).values())
    assert not any(np.array_equal(base[nm], bad[nm]) for nm in NAMES)                     # порча действительно влияет на более поздние π — тест не пустой


@pytest.mark.parametrize("k", K)
def test_corrupting_closed_targets_does_change_pi(k) -> None:
    x = world()
    closed = ~future_mask(x, k)
    closed[: k // 2] = False                                                              # порча закрытых пар, видимых дню k
    base, bad = pis(x), pis(corrupt(x, closed))
    assert not any(unchanged_to(k, base, bad).values())


@pytest.mark.parametrize("k", K)
def test_closed_plus_one_caught_through_combat_path(k, monkeypatch) -> None:
    x = world()
    monkeypatch.setattr(C, "closed_date", lambda tp, d: d)                                # мутация: пара доступна в день своего d (закрыто + 1 и раньше)
    base, bad = pis(x), pis(corrupt(x, future_mask(x, k)))
    caught = {nm: not eq for nm, eq in unchanged_to(k, base, bad).items()}
    assert all(caught.values()), caught


def test_seeing_the_whole_sample_caught(monkeypatch) -> None:
    x, k = world(), 60
    monkeypatch.setattr(C, "closed_date", lambda tp, d: np.full_like(tp, tp.min()))       # мутация: все пары доступны с первого дня (будущее)
    base, bad = pis(x), pis(corrupt(x, future_mask(x, k)))
    assert not any(unchanged_to(k, base, bad).values())


def future_leader(L, *_a, **_k):
    """Мутант «выбор по будущему»: π дня i — минимум суммы потерь пар i, i+1, … (исходы после d)."""
    return C.GRID[np.argmin(np.cumsum(L[::-1], axis=0)[::-1], axis=1)]


@pytest.mark.parametrize("k", K)
def test_future_selection_mutants_caught(k, monkeypatch) -> None:
    x = world()
    monkeypatch.setattr(C, "dftl", future_leader)
    monkeypatch.setattr(C, "adahedge_fs", future_leader)
    base, bad = pis(x), pis(corrupt(x, future_mask(x, k)))
    eq = unchanged_to(k, base, bad)
    assert not eq["DFTL126"] and not eq["AH"] and eq["HEDGE0"]                            # HEDGE0 не подменён — честный остаётся честным


def test_pit_helper_not_vacuous_and_mutations_restored() -> None:
    x = world()
    assert C.closed_date(x["t_plus"].to_numpy(), x["d"].to_numpy()) is not None
    t = np.array([3, 4])
    assert C.closed_date(t, np.array([1, 2])) is t                                        # подмены не остались в модуле
    assert C.dftl.__module__ == C.adahedge_fs.__module__ == C.__name__
