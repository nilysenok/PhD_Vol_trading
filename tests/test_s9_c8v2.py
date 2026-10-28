"""К (Ъ владельца 27.10, 00:10): расширение сетки C точками p50/p60 (setka_C.yaml: dop.rassh):
число точек = yaml; формулы = C1/C2 (S9.c_points); детектор утечек — σ после t → шум не меняет
затворы на барах ≤ t, сдвиг +1 ловится. Контроли к2/к3 — tests/test_s9_kontrol.py.
Данные не читаются. python3 -m pytest tests/test_s9_c8v2.py -q
"""
import numpy as np
import pytest
import yaml

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


