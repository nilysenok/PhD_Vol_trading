"""W4-A, лист W3 (core): известный ответ схем слоя доверия (DFTL, AH, HEDGE0) на синтетике; концы сетки π = M и C побитно; X1 воспроизводится `de_e1_state.run` побитно.
Подмены — только monkeypatch (класс № 43); реальные файлы не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from models_v2.s16 import x1_w3_core as C


def world(n: int = 240, gap: int = 2, seed: int = 0, one_state: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = np.datetime64("2020-01-06") + np.arange(n)
    M = np.exp(0.1 * rng.standard_normal(n))
    return pd.DataFrame(dict(d=d.astype("datetime64[ns]"), t_plus=(d + gap).astype("datetime64[ns]"), rv=1.4 * np.exp(0.3 * rng.standard_normal(n)),
                             M=M, C=2.0 * M * np.exp(0.1 * rng.standard_normal(n)), s=np.zeros(n, int) if one_state else rng.integers(0, 2, n)))


def arrs(x: pd.DataFrame):
    return x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy()


def quad(n: int, centers) -> np.ndarray:
    """Потери, квадратичные по π, минимум в centers[i] (без шума: ответ известен точно)."""
    return (C.GRID[None, :] - np.asarray(centers, float)[:, None]) ** 2


def test_grid_ends_are_M_and_C_bitwise(monkeypatch) -> None:
    x = world()
    M, Cc, rv = x["M"].to_numpy(), x["C"].to_numpy(), x["rv"].to_numpy()
    L = C.grid_losses(rv, M, Cc)
    assert np.array_equal(L[:, 0], C.qlike(rv, M)) and np.array_equal(L[:, -1], C.qlike(rv, Cc))
    assert np.array_equal(C.combine(M, Cc, np.zeros(len(M))), M) and np.array_equal(C.combine(M, Cc, np.ones(len(M))), Cc)
    mid = C.combine(M, Cc, np.full(len(M), 0.5))
    assert np.allclose(mid, np.sqrt(M * Cc), rtol=1e-12, atol=0)
    pure = lambda M_, C_, p: np.exp((1.0 - p) * np.log(M_) + p * np.log(C_))        # noqa: E731  мутация: без явных концов
    assert not np.array_equal(pure(M, Cc, np.ones(len(M))), Cc)                       # тест «побитно» различает такую подмену


def test_dftl_known_answer_and_states_are_separate() -> None:
    n = 120
    s, d, tp = np.zeros(n, int), (np.datetime64("2020-01-06") + np.arange(n)), np.datetime64("2020-01-06") + np.arange(n) + 1
    L = quad(n, np.full(n, 0.3))
    pi = C.dftl(L, s, d, tp, 126.0)
    assert pi[0] == 0.5 and np.all(pi[1:] == 0.3)                                    # нет закрытых пар — 0,5; дальше — минимум потерь
    s[-1] = 1
    assert C.dftl(L, s, d, tp, 126.0)[-1] == 0.5                                     # у состояния 1 закрытых пар нет, у состояния 0 их много


@pytest.mark.parametrize("hl", [126.0, 63.0, 252.0, np.inf])
def test_dftl_equals_independent_reference(hl) -> None:
    x = world(seed=3)
    s, d, tp = arrs(x)
    L = C.grid_losses(x["rv"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy())
    lam = 1.0 if not np.isfinite(hl) else 0.5 ** (1.0 / hl)
    ref = np.empty(len(s))
    for i in range(len(s)):
        js = [j for j in range(len(s)) if tp[j] <= d[i] and s[j] == s[i]]           # закрытые пары своего состояния, по порядку прихода
        if not js:
            ref[i] = 0.5
            continue
        w = lam ** np.arange(len(js) - 1, -1, -1)
        ref[i] = C.GRID[int(np.argmin((w[:, None] * L[js]).sum(0)))]
    assert np.array_equal(C.dftl(L, s, d, tp, hl), ref)


def test_dftl_regime_shift_discount_matters() -> None:
    n = 200
    d = np.datetime64("2020-01-06") + np.arange(n)
    L = quad(n, np.r_[np.full(100, 0.1), np.full(100, 0.9)])
    s = np.zeros(n, int)
    inf, h63 = C.dftl(L, s, d, d + 1, np.inf), C.dftl(L, s, d, d + 1, 63.0)
    assert inf[150] == 0.35                                                          # среднее (100·0,1 + 50·0,9)/150 = 0,367 → ближайшая точка сетки
    assert h63[150] > inf[150] and h63[199] > inf[199]                               # дисконт следует за новым режимом быстрее


def test_ah_known_answer_range_and_states() -> None:
    n = 400
    rng = np.random.default_rng(0)
    d = np.datetime64("2020-01-06") + np.arange(n)
    L = quad(n, np.full(n, 0.3)) + 0.01 * rng.standard_normal((n, 1))
    s = np.zeros(n, int)
    pi = C.adahedge_fs(L, s, d, d + 1, 0.01)
    assert pi[0] == pytest.approx(0.5, abs=1e-12) and abs(pi[-50:].mean() - 0.3) < 0.01
    assert np.all((pi >= 0) & (pi <= 1))
    s[-1] = 1
    assert C.adahedge_fs(L, s, d, d + 1, 0.01)[-1] == pytest.approx(0.5, abs=1e-12)  # состояние без закрытых пар — равные веса


def test_hedge0_known_answer_expit() -> None:
    n = 40
    x = world(n=n, gap=1, one_state=True)
    x["rv"] = x["M"]                                                                 # QLIKE(M) = 0, QLIKE(C) = QLIKE при r = 1/2 > 0
    s, d, tp = arrs(x)
    qc, qm = C.qlike(x["rv"], x["C"]), C.qlike(x["rv"], x["M"])
    assert np.all(qm == 0.0) and qc.min() > 0
    pi = C.scheme("HEDGE0", x, C.grid_losses(x["rv"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy()))
    ref = C.fs_pass(qc, qm, s, d, tp, np.zeros((n, 2)))[0]
    assert np.array_equal(pi, ref) and pi[0] == 0.5 and np.all(np.diff(pi) < 0)       # C хуже M всегда — доверие к C только падает
    xc = x.copy()
    xc["C"] = xc["M"]
    same = C.scheme("HEDGE0", xc, C.grid_losses(xc["rv"].to_numpy(), xc["M"].to_numpy(), xc["C"].to_numpy()))
    assert np.all(same == 0.5)                                                       # C ≡ M: разности потерь нет, π остаётся ½


def test_hedge0_is_x1_with_zero_alpha_and_alpha_matters() -> None:
    x = world(seed=5)
    s, d, tp = arrs(x)
    qc, qm = C.qlike(x["rv"], x["C"]), C.qlike(x["rv"], x["M"])
    a0 = C.fs_pass(qc, qm, s, d, tp, np.zeros((len(s), 2)))[0]
    a5 = C.fs_pass(qc, qm, s, d, tp, np.full((len(s), 2), 0.05))[0]
    assert not np.array_equal(a0, a5)
    assert a5.min() >= 0.05 / 2 - 1e-12 and a5.max() <= 1 - 0.05 / 2 + 1e-12         # fixed share: π не уходит ближе α/2 к концам
    assert a0.max() > 1 - 0.05 / 2 and a0.min() < 0.05 / 2                           # без подмешивания (α = 0) — уходит


def x1_pieces(x: pd.DataFrame):
    import models_v2.s15  # noqa: F401  (torch первым, как в x1_w3_diag.check_repro)
    from models_v2.s15 import de_e1_state as E
    s, d, tp = arrs(x)
    return E.run(x["rv"].to_numpy(), x["M"].to_numpy(), x["C"].to_numpy(), s, d, tp), (C.qlike(x["rv"], x["C"]), C.qlike(x["rv"], x["M"]), s, d, tp)


def test_x1_reproduced_bitwise_by_de_e1_state_run(monkeypatch) -> None:
    x = world(n=300, gap=3, seed=7)
    r, (qc, qm, s, d, tp) = x1_pieces(x)
    assert len(np.unique(np.r_[r["a0"], r["a1"]])) > 1                               # α действительно выбирается, не константа
    pi, eta, pis = C.fs_pass(qc, qm, s, d, tp, np.column_stack([r["a0"], r["a1"]]))
    assert np.array_equal(pi, r["pi"]) and np.array_equal(pis, r["pis"]) and np.array_equal(eta, r["eta"], equal_nan=True)
    monkeypatch.setattr(C, "closed_date", lambda t, dd: dd)                          # мутация «закрыто + 1» в проходе X1
    assert not np.array_equal(C.fs_pass(qc, qm, s, d, tp, np.column_stack([r["a0"], r["a1"]]))[0], r["pi"])


def test_closed_date_restored_after_mutations() -> None:
    t = np.array([5, 6])
    assert C.closed_date(t, np.array([1, 2])) is t
