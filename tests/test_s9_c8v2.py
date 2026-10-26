"""К (Ъ владельца 27.10, 00:10): сетка C + p50/p60 и контроли без прогноза к1/к2 (fastbt.fast_kontrol).

(а) расширение сетки (setka_C.yaml: dop.rassh): число точек = yaml; формулы = C1/C2 (S9.c_points);
    детектор утечек — σ после t → шум не меняет затворы на барах ≤ t, сдвиг +1 ловится;
(б) к2: перестановки внутри пар, доля и уровни 1 / 0,5 / 0 сохраняются; Шарп ядра на тождественной
    перестановке = s9.Engine.port → metrics; p = (1 + #≥) / (N + 1);
(в) синтетика: маска C случайная (та же доля) → p к к2 распределён равномерно;
(г) мутация: маска к2 берёт другую долю → ловится (check_share и неравномерность p).
Данные не читаются. python3 -m pytest tests/test_s9_c8v2.py -q
"""
import types

import numpy as np
import pytest
import yaml
from scipy import stats

from fastbt import fast_kontrol as fk
from fastbt import s9
from fastbt.s9_stats import minp, past_rank

DOP = yaml.safe_load(s9.C_DOP_YAML.read_text())["dop"]
RSPEC = dict(min_periods=DOP["min_periods"], semi=DOP["rassh"])
RNG = np.random.RandomState(27)
T, N = 3, 1300
n_t = np.array([N, N - 120, N - 400])
S0 = np.full((T, N), np.nan)
for i in range(T):
    S0[i, 25:n_t[i]] = np.exp(RNG.normal(-4, 0.3, n_t[i] - 25))


# ---------------- (а) расширение сетки ----------------

def _flat(S):
    return [(f, lab, G) for f, pts in s9.c_dop_gates(S, None, n_t, RSPEC) for lab, G in pts]


def test_rassh_point_count_matches_yaml():
    got = {f: len(p) for f, p in s9.c_dop_gates(S0, None, n_t, RSPEC)}
    want = {k: v for k, v in DOP["tochek_rassh_na_sigma"].items() if k != "vsego"}
    assert got == want and sum(got.values()) == DOP["tochek_rassh_na_sigma"]["vsego"] == 18
    assert DOP["tochek_rassh_vsego"] == 18 * 6
    labs = [lab for _, lab, _ in _flat(S0)]
    assert len(set(labs)) == len(labs)
    assert {"W126_p50", "W252_p60", "W504_p50", "W252_p50_60", "W126_p60_80", "W504_p50_70"} <= set(labs)


def test_rassh_formulas_equal_c1_c2():
    """Те же формулы, что C1/C2 (S9.c_points): p70 через путь расширения = C1 W252 p70 и т. п."""
    class E:
        pass
    e = E()
    e.n_t = n_t
    e.sigma = lambda src: S0
    f = E()
    f.e = e
    pts1, pts2 = s9.S9.c_points(f, "x")
    old = dict(pts1 + pts2)
    spec = dict(min_periods=DOP["min_periods"],
                semi={"C1": dict(signal="sigma", okna=[126, 252], filtr=[70, 90]),
                      "C2": dict(signal="sigma", okna=[252], chastichnaya=[[70, 90], [70, 80]])})
    for _, pts in s9.c_dop_gates(S0, None, n_t, spec):
        for lab, G in pts:
            assert np.array_equal(G, old[lab]), lab
    R = np.full((T, N), np.nan)
    for ti in range(T):
        R[ti, :n_t[ti]] = past_rank(S0[ti, :n_t[ti]].copy(), 252, minp(252))
    new = dict((lab, G) for _, lab, G in _flat(S0))
    nan = np.isnan(R)
    with np.errstate(invalid="ignore"):
        assert np.array_equal(new["W252_p60"][0], np.where(nan | (R > 0.4), 1.0, 0.0))
        assert np.array_equal(new["W252_p50"][2], np.where(nan | ((R >= 0.5) & (R <= 0.5)), 1.0, 0.0))


def _noise_after(A, t):
    rng = np.random.RandomState(1000 + t)
    B = A.copy()
    B[:, t + 1:] = np.exp(rng.normal(-3, 1.0, B[:, t + 1:].shape))
    return B


def _shift1(A):
    B = np.full_like(A, np.nan)
    B[:, :-1] = A[:, 1:]
    return B


@pytest.mark.parametrize("t", [400, 700, 880])
def test_rassh_future_noise(t):
    base, pert = _flat(S0), _flat(_noise_after(S0, t))
    assert len(base) == len(pert) == 18
    for (f, lab, a), (_, _, b) in zip(base, pert):
        assert np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1]), (f, lab)


def test_rassh_shift_plus1_caught():
    base = _flat(_shift1(S0))
    caught = np.zeros(len(base), bool)
    for t in range(560, 860, 30):
        pert = _flat(_shift1(_noise_after(S0, t)))
        caught |= [not np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1])
                   for (_, _, a), (_, _, b) in zip(base, pert)]
    assert caught.all(), [(f, lab) for (f, lab, _), c in zip(base, caught) if not c]


# ---------------- синтетический движок ----------------

def _engine(G=12, D=500, drift=0.0, seed=0):
    """Пары × дни, строки по парам; A — позиции ±1/0 с инерцией; доходности — шум (+ дрейф по A)."""
    rng = np.random.RandomState(seed)
    n = G * D
    A = np.zeros(n)
    for g in range(G):
        s = np.sign(rng.normal(size=D // 20)).repeat(20)[:D]
        on = rng.random_sample(D) < 0.8
        A[g * D:(g + 1) * D] = s * on
    lr = rng.normal(0, 0.01, n) + drift * A
    pair = np.repeat(np.arange(G), D)
    date = np.tile(np.arange(D), G)
    first = np.r_[True, pair[1:] != pair[:-1]]
    kr = np.full(D, 0.07)
    ns = types.SimpleNamespace(A_row=A, lr_row=lr, first=first, dates=np.arange(D),
                               cnt=np.full(D, float(G)), kr=kr, rf=kr / 365.0,
                               cal=rng.choice([1.0, 3.0], D, p=[0.8, 0.2]),
                               r=dict(pair=pair, date=date))
    import pandas as pd
    ns.dates = pd.date_range("2018-01-01", periods=D, freq="B")
    ns._finish = lambda *a: s9.Engine._finish(ns, *a)
    return ns


def _gate_like_C(ns, seed, share0=0.3, share_half=0.1):
    """Затвор с уровнями 1 / 0,5 / 0 (доли 0,6 / 0,1 / 0,3 по дням с A ≠ 0 в каждой паре)."""
    rng = np.random.RandomState(seed)
    g = np.ones(len(ns.A_row))
    for p in np.unique(ns.r["pair"]):
        idx = np.flatnonzero((ns.r["pair"] == p) & (ns.A_row != 0))
        k0, kh = int(round(share0 * len(idx))), int(round(share_half * len(idx)))
        sel = rng.permutation(idx)
        g[sel[:k0]] = 0.0
        g[sel[k0:k0 + kh]] = 0.5
    return g


def test_perm_index_within_groups():
    groups = np.repeat(np.arange(5), [7, 1, 30, 12, 50])[np.random.RandomState(1).permutation(100)]
    P = fk.perm_index(groups, 50, 3)
    assert P.shape == (50, 100)
    for b in range(50):
        assert np.array_equal(np.sort(P[b]), np.arange(100))
        assert np.array_equal(groups[P[b]], groups)
    assert np.array_equal(P, fk.perm_index(groups, 50, 3))          # seed фиксирован
    assert not np.array_equal(P, fk.perm_index(groups, 50, 4))


def test_identity_perm_equals_engine_port():
    ns = _engine(seed=5)
    k = fk.Kontrol(ns, 0.0004, n_perm=20)
    pos = ns.A_row * _gate_like_C(ns, 9)
    ref = s9.metrics(s9.Engine.port(ns, pos, comm=0.0004))["sharpe"]
    assert abs(k.identity_sharpe(pos) - ref) < 1e-10
    # к1: константа — отношение средних |позиций|
    assert abs(k.k1_const(pos) - np.abs(pos).mean() / np.abs(ns.A_row).mean()) < 1e-15
    fk.check_share(k.gate_act(pos), k.P, k.groups)
    assert fk.p_perm(1.0, np.array([0.5, 1.0, 2.0])) == 3 / 4


def _p_null(ns, k, reps, mutate=False):
    ps = []
    for s in range(reps):
        pos = ns.A_row * _gate_like_C(ns, 100 + s)
        P = k.P
        if mutate:
            P = _mutated_P(k, pos)
        ps.append(fk.p_perm(s9.metrics(s9.Engine.port(ns, pos, comm=k.comm))["sharpe"],
                            k.k2_sharpes(pos, P)))
    return np.array(ps)


def _mutated_P(k, pos):
    """Мутация: маска к2 берёт другую долю — половина нулевых дней заменена днями с g = 1."""
    gr = k.gate_act(pos)
    P = k.P.copy()
    rng = np.random.RandomState(0)
    for gi in np.unique(k.groups):
        loc = np.flatnonzero(k.groups == gi)
        ones = loc[gr[loc] == 1.0]
        for b in range(P.shape[0]):
            row = P[b, loc]
            z = np.flatnonzero(gr[row] == 0.0)
            z = z[rng.random_sample(len(z)) < 0.5]
            P[b, loc[z]] = rng.choice(ones, len(z))
    return P


def test_k2_p_uniform_under_random_mask():
    """Маска C — случайная той же доли: p к к2 ~ U(0, 1) (KS), доля p < 0,05 около 5 %."""
    ns = _engine(seed=11, drift=0.002)
    k = fk.Kontrol(ns, 0.0004, n_perm=199, seed=fk.SEED)
    ps = _p_null(ns, k, 300)
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert 0.01 <= np.mean(ps < 0.05) <= 0.10


def test_mutation_other_share_caught():
    ns = _engine(seed=11, drift=0.002)
    k = fk.Kontrol(ns, 0.0004, n_perm=199, seed=fk.SEED)
    pos = ns.A_row * _gate_like_C(ns, 100)
    with pytest.raises(AssertionError):
        fk.check_share(k.gate_act(pos), _mutated_P(k, pos), k.groups)
    ps = _p_null(ns, k, 60, mutate=True)
    assert stats.kstest(ps, "uniform").pvalue < 1e-3


def test_gate_outside_A_rejected():
    ns = _engine(seed=2)
    k = fk.Kontrol(ns, 0.0004, n_perm=5)
    pos = ns.A_row.copy()
    pos[np.flatnonzero(ns.A_row == 0)[0]] = 1.0
    with pytest.raises(AssertionError):
        k.gate_act(pos)
