"""С9, новые точки сетки C (Ъ 22:30: C2W, C*1, C*2 — fastbt.s9.c_dop_gates): детектор утечек.

(а) «Будущее не влияет»: σ_pred и √RV_w после бара t заменены шумом → затворы на барах ≤ t
    совпадают побайтно; (б) контроль: σ_d := σ_{d+1} (сдвиг +1) обязан ловиться проверкой (а)
    во всех точках; (в) окно ранга без текущего дня, W504 — разгон от 252; (г) число точек = yaml;
    (д) формулы фильтра и частичной экспозиции совпадают с C1/C2 (S9.c_points) на тех же окнах.
python3 -m pytest tests/test_s9_cdop_leak.py -q
"""
import numpy as np
import pytest
import yaml

from fastbt import s9
from fastbt.s9_stats import minp, past_rank

SPEC = yaml.safe_load(s9.C_DOP_YAML.read_text())["dop"]
RNG = np.random.RandomState(13)
T, N = 3, 1300
n_t = np.array([N, N - 120, N - 400])
S0 = np.full((T, N), np.nan)
RV0 = np.full((T, N), np.nan)
for i in range(T):
    S0[i, 25:n_t[i]] = np.exp(RNG.normal(-4, 0.3, n_t[i] - 25))
    RV0[i, 25:n_t[i]] = np.exp(RNG.normal(-4, 0.4, n_t[i] - 25))


def _flat(S, RV):
    return [(f, lab, G) for f, pts in s9.c_dop_gates(S, RV, n_t, SPEC) for lab, G in pts]


def _noise_after(A, t, seed=0):
    rng = np.random.RandomState(1000 + t + seed)
    B = A.copy()
    B[:, t + 1:] = np.exp(rng.normal(-3, 1.0, B[:, t + 1:].shape))
    return B


def _shift1(A):
    B = np.full_like(A, np.nan)
    B[:, :-1] = A[:, 1:]
    return B


@pytest.mark.parametrize("t", [400, 700, 880])
def test_future_noise(t):
    base, pert = _flat(S0, RV0), _flat(_noise_after(S0, t), _noise_after(RV0, t))
    assert len(base) == len(pert) > 0
    for (f, lab, a), (_, _, b) in zip(base, pert):
        assert np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1]), (f, lab)


def test_shift_plus1_caught():
    """При сдвиге +1 шум после t виден только на баре t; бинарный затвор на одном баре может не
    переключиться, поэтому точка считается пойманной, если расхождение есть хотя бы на одной из
    10 дат t (как детектор на позициях: много пар и дат)."""
    base = _flat(_shift1(S0), _shift1(RV0))
    caught = np.zeros(len(base), bool)
    for t in range(560, 860, 30):
        pert = _flat(_shift1(_noise_after(S0, t)), _shift1(_noise_after(RV0, t)))
        caught |= [not np.array_equal(a[:, :, :t + 1], b[:, :, :t + 1])
                   for (_, _, a), (_, _, b) in zip(base, pert)]
    missed = [(f, lab) for (f, lab, _), c in zip(base, caught) if not c]
    assert not missed, missed


def test_window_excludes_current_day_and_w504_warmup():
    s = np.arange(1.0, 700.0)
    r = past_rank(s.copy(), 504, int(SPEC["min_periods"][504]))
    assert SPEC["min_periods"][504] == 252
    assert np.isnan(r[251]) and r[252] == 1.0 and r[600] == 1.0


def test_point_count_matches_yaml():
    got = {f: len(p) for f, p in s9.c_dop_gates(S0, RV0, n_t, SPEC)}
    want = {k: v for k, v in SPEC["tochek_na_sigma"].items() if k != "vsego"}
    assert got == want and sum(got.values()) == SPEC["tochek_na_sigma"]["vsego"]
    naive = {f: len(p) for f, p in s9.c_dop_gates(S0, None, n_t, SPEC)}
    assert naive == SPEC["tochek_na_naive"]


def test_same_formulas_as_c1_c2():
    """Фильтр/частичная из _c_gate = C1/C2 из S9.c_points при тех же рангах (W252, minp 50)."""
    class E:
        pass
    e = E()
    e.n_t = n_t
    e.sigma = lambda src: S0
    f = E()
    f.e = e
    pts1, pts2 = s9.S9.c_points(f, "x")
    old = dict(pts1 + pts2)
    R = np.full((T, N), np.nan)
    for ti in range(T):
        R[ti, :n_t[ti]] = past_rank(S0[ti, :n_t[ti]].copy(), 252, minp(252))
    for p in (70, 80, 90):
        assert np.array_equal(s9._c_gate(R, p / 100), old[f"W252_p{p}"])
    for p1, p2 in ((70, 90), (80, 90), (70, 80)):
        assert np.array_equal(s9._c_gate(R, (p1 / 100, p2 / 100)), old[f"W252_p{p1}_{p2}"])
