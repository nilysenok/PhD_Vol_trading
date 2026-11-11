"""S-A-IMPL-1, мутации листа (S_A_SPEC разд. 8): каждая из семи подложенных ошибок обязана ронять проверку (mutations.md № 21/44, 142). Только monkeypatch (№ 43)."""
import copy

import numpy as np
import pytest
import torch

from models_v2.s18 import a_feat as A
import s18_a_feat_checks as C
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)

LAG0, WEIGHTS0, RVB0 = A.lag_idx, A.lag_weights, A.rv_b


@pytest.fixture(scope="module")
def F():
    return C.synth2()


def mut_lag(kind):
    def lag_idx(F, rows):
        rows = np.asarray(rows)
        out = []
        for L in A.LAGS:
            lg = np.arange(L)
            if kind == "plus1":                       # окно с сессией d + 1
                ok = lg[None, :] <= F.pos[rows][:, None]
                ix = np.minimum(rows[:, None] + 1 - lg[None, :], len(F.dates) - 1)
            elif kind == "no_d":                      # окно «без d»: от d − 1
                ok = lg[None, :] <= F.pos[rows][:, None] - 1
                ix = rows[:, None] - 1 - lg[None, :]
            else:                                     # «notrunc»: склейка через остановку 2022
                ok = lg[None, :] <= rows[:, None]
                ix = rows[:, None] - lg[None, :]
            out.append((np.where(ok, ix, 0), ok))
        return out
    return lag_idx


def test_mutation_window_with_session_d_plus_1_caught(F, monkeypatch):
    C.check_eq(F), C.check_pit(F)
    monkeypatch.setattr(A, "lag_idx", mut_lag("plus1"))
    with pytest.raises(AssertionError):
        C.check_pit(F)
    with pytest.raises(AssertionError):
        C.check_eq(F)


def test_mutation_normalization_over_whole_sample_caught(F, monkeypatch):
    def rv_b_all(F, E):                               # среднее exp(h) по всем 53 слотам, а не по присутствующим
        r2, pres, n = F.tens()
        E = torch.as_tensor(E)
        return A.SCALE * (r2 @ E.T) / E.mean(1)[None, :]

    def rv_b_sample(F, E):                            # средняя доля присутствия по ВСЕЙ выборке (будущие дни входят)
        r2, pres, n = F.tens()
        E = torch.as_tensor(E)
        return A.SCALE * (r2 @ E.T) / ((pres.mean(0) @ E.T)[None, :] * 1.0)
    C.check_bmean(F)
    for mut in (rv_b_all, rv_b_sample):
        monkeypatch.setattr(A, "rv_b", mut)
        with pytest.raises(AssertionError):
            C.check_bmean(F)
    monkeypatch.setattr(A, "rv_b", rv_b_sample)
    with pytest.raises(AssertionError):
        C.check_pit(F)


def test_mutation_a_outside_simplex_caught(F, monkeypatch):
    def unnormalized_g(ok, k, omega, psi):
        L = A.LAGS[k]
        m = torch.as_tensor(ok, dtype=torch.float64)
        x = torch.arange(L, dtype=torch.float64) / (L - 1)
        R = m * (torch.arange(L) < A.RECT[k]).to(torch.float64)
        return (1.0 - omega) * R / R.sum(1, keepdim=True) + omega * m * torch.exp(psi[0] * x + psi[1] * x * x)

    def omega_outside(ok, k, omega, psi):             # ω ∉ [0, 1]: коэффициент при R отрицателен
        return WEIGHTS0(ok, k, omega * 3.0, psi)
    for mut in (unnormalized_g, omega_outside):
        monkeypatch.setattr(A, "lag_weights", mut)
        with pytest.raises(AssertionError):
            C.check_simplex(F)


def test_mutation_a_har_differs_from_k1_caught(F, monkeypatch):
    monkeypatch.setattr(A, "lag_idx", mut_lag("no_d"))
    with pytest.raises(AssertionError):
        C.check_eq(F)
    monkeypatch.setattr(A, "lag_idx", LAG0)

    def mean_of_logs(F, theta, lag, tau=A.TAU):       # среднее логарифмов вместо логарифма среднего
        rb = A.rv_b(F, A.slot_exp(theta["c"], tau))
        cols = []
        for k, (ix, ok) in enumerate(lag):
            a = A.lag_weights(ok, k, theta["omega"][k], theta["psi"][k])
            cols.append((a * torch.log(rb[torch.as_tensor(ix), k])).sum(1))
        return torch.stack(cols, dim=1)
    monkeypatch.setattr(A, "ln_S", mean_of_logs)
    with pytest.raises(AssertionError):
        C.check_eq(F)


def test_mutation_slot_by_bar_number_instead_of_time_caught():
    G = A.build(C.bars_missing_first(), index_days=None)
    C.check_slots(G)
    with pytest.raises(AssertionError):
        C.check_slots(C.pack_left(G))


def test_mutation_truncation_by_segment_removed_caught(F, monkeypatch):
    monkeypatch.setattr(A, "lag_idx", mut_lag("notrunc"))
    with pytest.raises(AssertionError):
        C.check_eq(F)


def test_mutation_float32_store_instead_of_float64_bars_caught(F):
    G = copy.copy(F)                                   # r2 пропущен через float32 (склад M10 V7), rv — float64 как у K1
    G.r2 = F.r2.astype(np.float32).astype(np.float64)
    G._t = None
    with pytest.raises(AssertionError):
        C.check_eq(G)
