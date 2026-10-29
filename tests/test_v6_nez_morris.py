"""Независимые тесты тестировщика к V6-14 (fastbt/morris_v6.py): траектории, элементарные эффекты, μ*, σ, U95.

Известные ответы: индекс уровня против дробной арифметики (Fraction), структура траектории (каждая ось ровно один раз,
шаг +-2 уровня), расстояние Кампалонго циклом, отбор против перебора на малом наборе, функции с известным μ*
(линейная, нулевая ось, взаимодействие, дочерняя ось), ручной эффект/статистика, граничные случаи (ось без активных шагов,
один активный шаг, нулевые эффекты), детерминизм seed.
"""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from fastbt import morris_v6 as M


# ---------------- индекс уровня ----------------
def test_indeks_vs_fractions_all_L():
    for L in range(1, 201):
        for j in range(4):
            x = Fraction(j, 3) * (L - 1)
            fl = x.numerator // x.denominator
            frac = x - fl
            assert frac != Fraction(1, 2)                        # ничьих нет (знаменатель 3)
            exp = fl + (1 if frac > Fraction(1, 2) else 0)
            assert int(M.indeks(j, L)) == exp, (j, L)


def test_indeks_hand_values_and_vectorised():
    assert [int(M.indeks(j, 2)) for j in range(4)] == [0, 0, 1, 1]
    assert [int(M.indeks(j, 3)) for j in range(4)] == [0, 1, 1, 2]
    assert [int(M.indeks(j, 4)) for j in range(4)] == [0, 1, 2, 3]
    assert [int(M.indeks(j, 5)) for j in range(4)] == [0, 1, 3, 4]
    assert [int(M.indeks(j, 1)) for j in range(4)] == [0, 0, 0, 0]
    J = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    L = np.array([2, 3, 4, 5])
    assert M.indeks(J, L).tolist() == [[0, 1, 2, 4], [1, 1, 1, 0]]


def test_partner_step_changes_every_nontrivial_axis():
    """Шаг +2 уровня (mod 4) обязан менять индекс значения для любой оси с L >= 2 - иначе шаг был бы «неактивным»."""
    for L in range(2, 60):
        for j in range(4):
            assert int(M.indeks(j, L)) != int(M.indeks((j + 2) % 4, L)), (j, L)


# ---------------- траектории ----------------
LS = [2, 2, 3, 5, 13, 4, 2, 7]


def test_trajectory_structure():
    I, por = M.traektorii(LS, 12, seed=11)
    k = len(LS)
    assert I.shape == (12, k + 1, k) and por.shape == (12, k)
    for tr in range(12):
        assert sorted(por[tr].tolist()) == list(range(k))               # каждая ось ровно один раз
        for t in range(1, k + 1):
            diff = np.flatnonzero(I[tr, t] != I[tr, t - 1])
            assert diff.tolist() == [por[tr, t - 1]]                     # меняется только заявленная ось
        assert np.all(I >= 0) and all(I[:, :, a].max() <= LS[a] - 1 for a in range(k))


def test_trajectory_levels_come_from_four_level_grid():
    I, _ = M.traektorii(LS, 10, seed=3)
    for a, L in enumerate(LS):
        allowed = {int(M.indeks(j, L)) for j in range(4)}
        assert set(np.unique(I[:, :, a]).tolist()) <= allowed


def test_trajectory_determinism_and_seed_sensitivity():
    a, pa = M.traektorii(LS, 10, seed=5)
    np.random.seed(123)
    b, pb = M.traektorii(LS, 10, seed=5)
    c, _ = M.traektorii(LS, 10, seed=6)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)
    assert not np.array_equal(a, c)
    d, _ = M.traektorii(LS, 10)                    # seed по умолчанию = SEED
    e, _ = M.traektorii(LS, 10, seed=M.SEED)
    assert np.array_equal(d, e) and M.SEED == 20261101


def test_selected_are_subset_of_candidates_and_mnozh1_takes_first():
    L = np.asarray(LS)
    J, por = M.kandidaty(len(LS), 10 * 6, np.random.default_rng(9))
    I, p = M.traektorii(LS, 6, seed=9)
    cand = {tuple(M.indeks(J[i], L).astype(int).ravel()) for i in range(len(J))}
    for tr in range(6):
        assert tuple(I[tr].astype(int).ravel()) in cand
    I1, p1 = M.traektorii(LS, 6, seed=9, mnozh=1)         # mnozh=1: кандидатов ровно r, отбора нет
    J6, por6 = M.kandidaty(len(LS), 6, np.random.default_rng(9))
    assert np.array_equal(I1, M.indeks(J6, L).astype(np.int16))
    assert np.array_equal(p1, por6)


# ---------------- расстояние и отбор ----------------
def _campolongo_loop(J):
    u = J.astype(float) / 3.0
    m, n, _ = u.shape
    out = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a != b:
                out[a, b] = sum(np.linalg.norm(u[a, i] - u[b, j]) for i in range(n) for j in range(n))
    return out ** 2


def test_razneseniye_vs_direct_loop():
    J, _ = M.kandidaty(5, 7, np.random.default_rng(1))
    # расхождение ~1e-9 отн.: sqrt от разности квадратов (кэнселяция) на почти совпадающих точках - не дефект
    assert np.allclose(M.razneseniye(J), _campolongo_loop(J), rtol=1e-6, atol=1e-4)
    d2 = M.razneseniye(J)
    assert np.allclose(d2, d2.T) and np.all(np.diag(d2) == 0)


def test_razneseniye_identical_trajectories_zero_and_scale():
    J = np.zeros((2, 3, 2), np.int8)
    assert np.all(M.razneseniye(J) == 0)
    J2 = np.zeros((2, 2, 1), np.int8)
    J2[1] = 3                                   # u = 0 против u = 1; 4 пары точек по расстоянию 1 -> d = 4 -> d² = 16
    assert M.razneseniye(J2)[0, 1] == pytest.approx(16.0)


def test_otbor_returns_r_unique_sorted_and_beats_random():
    J, _ = M.kandidaty(6, 40, np.random.default_rng(2))
    keep = M.otbor(J, 8)
    assert len(keep) == 8 and len(set(keep.tolist())) == 8 and keep.tolist() == sorted(keep.tolist())
    d2 = M.razneseniye(J)
    score = lambda idx: d2[np.ix_(idx, idx)].sum()      # noqa: E731
    rng = np.random.default_rng(0)
    rnd = [score(rng.choice(40, 8, replace=False)) for _ in range(200)]
    assert score(keep) > max(rnd)


def test_otbor_close_to_bruteforce_optimum_small():
    J, _ = M.kandidaty(4, 9, np.random.default_rng(4))
    d2 = M.razneseniye(J)
    best = max(d2[np.ix_(c, c)].sum() for c in combinations(range(9), 4))
    got = d2[np.ix_(M.otbor(J, 4), M.otbor(J, 4))].sum()
    assert got <= best + 1e-9
    assert got >= 0.9 * best          # жадное удаление - эвристика; сильное отставание = дефект


def test_otbor_r_equals_m_keeps_all():
    J, _ = M.kandidaty(3, 5, np.random.default_rng(8))
    assert M.otbor(J, 5).tolist() == [0, 1, 2, 3, 4]


# ---------------- эффекты: ручной пример ----------------
def test_effekty_hand_example():
    # r=2 траектории, k=3 оси. Порядок осей: траектория 0: [2,0,1]; траектория 1: [1,2,0].
    por = np.array([[2, 0, 1], [1, 2, 0]])
    Y = np.array([[10.0, 13.0, 12.0, 20.0], [1.0, 1.0, 4.0, 3.0]])
    keys = np.array([[0, 1, 2, 3], [0, 0, 5, 6]])       # у траектории 1 шаг 1 неактивен (ключ тот же)
    E = M.effekty(Y, keys, por)
    # тр.0: ось2: +3, ось0: -1, ось1: +8; тр.1: ось1: неактивен (NaN), ось2: +3, ось0: -1
    exp = np.array([[-1.0, 8.0, 3.0], [-1.0, np.nan, 3.0]])
    assert np.allclose(E, exp, equal_nan=True)


def test_effekty_nan_y_makes_both_adjacent_steps_inactive():
    por = np.array([[0, 1, 2]])
    Y = np.array([[0.0, np.nan, 5.0, 7.0]])
    keys = np.array([[0, 1, 2, 3]])
    E = M.effekty(Y, keys, por)
    assert np.isnan(E[0, 0]) and np.isnan(E[0, 1]) and E[0, 2] == 2.0


def test_effekty_inf_y_not_counted():
    por = np.array([[0, 1]])
    E = M.effekty(np.array([[0.0, np.inf, 1.0]]), np.array([[0, 1, 2]]), por)
    assert np.isnan(E).all()


# ---------------- статистика: ручной пример ----------------
def test_statistika_hand_example():
    nan = np.nan
    E = np.array([[1.0, -2.0, nan, 0.0],
                  [3.0, 2.0, nan, 0.0],
                  [-2.0, nan, nan, 0.0],
                  [nan, 4.0, 5.0, 0.0]])
    s = M.statistika(E, B=500, seed=1)
    assert s["n_akt"].tolist() == [3, 3, 1, 4]
    assert s["mu_star"][0] == pytest.approx((1 + 3 + 2) / 3) and s["mu_star"][1] == pytest.approx((2 + 2 + 4) / 3)
    assert s["mu_star"][2] == pytest.approx(5.0) and s["mu_star"][3] == 0.0
    assert s["sigma"][0] == pytest.approx(np.std([1.0, 3.0, -2.0], ddof=1))
    assert s["sigma"][1] == pytest.approx(np.std([-2.0, 2.0, 4.0], ddof=1))
    assert np.isnan(s["sigma"][2]) and s["sigma"][3] == 0.0
    assert s["U95"][3] == 0.0                         # нулевая ось: граница ровно 0
    for i in (0, 1, 2):
        absv = np.abs(E[:, i][np.isfinite(E[:, i])])
        assert absv.min() - 1e-12 <= s["U95"][i] <= absv.max() + 1e-12


def test_statistika_axis_without_active_steps_is_nan_not_zero():
    E = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    s = M.statistika(E, B=200, seed=2)
    assert s["n_akt"].tolist() == [3, 0]
    assert np.isnan(s["mu_star"][1]) and np.isnan(s["sigma"][1]) and np.isnan(s["U95"][1])


def test_statistika_constant_abs_effect_gives_u95_equal_mu():
    E = np.array([[0.7], [-0.7], [0.7], [0.7], [-0.7], [0.7]])
    s = M.statistika(E, B=300, seed=3)
    assert s["mu_star"][0] == pytest.approx(0.7) and s["U95"][0] == pytest.approx(0.7)


def test_statistika_u95_at_least_mu_for_noisy_axis_and_deterministic():
    r = np.random.default_rng(1)
    E = r.normal(0, 1, (40, 3))
    a = M.statistika(E, B=1000, seed=7)
    b = M.statistika(E, B=1000, seed=7)
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True)
    assert np.all(a["U95"] > a["mu_star"])            # 95-й процентиль среднего больше самого среднего для шумной оси
    assert np.all(a["U95"] < np.abs(E).max(0))


def test_statistika_sigma_uses_signed_effects():
    E = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    s = M.statistika(E, B=50, seed=0)
    assert s["mu_star"][0] == 1.0 and s["sigma"][0] == pytest.approx(np.std([1, -1, 1, -1], ddof=1))
    assert s["sigma"][0] > 1.0                        # по |EE| было бы 0


# ---------------- сквозные известные ответы на функциях ----------------
def _run(f, L, r=40, seed=M.SEED, keyf=None, B=300):
    I, por = M.traektorii(L, r, seed=seed)
    flat = I.reshape(-1, len(L))
    Y = np.array([f(x) for x in flat], float).reshape(r, len(L) + 1)
    keys = np.array([hash(keyf(x) if keyf else tuple(x)) for x in flat]).reshape(r, len(L) + 1)
    return M.statistika(M.effekty(Y, keys, por), B=B, seed=1), I, por


def test_linear_function_mu_star_equals_abs_coefficient_times_step():
    L = [2, 2, 5, 13, 1, 4]                      # ось 4 (L=1) не меняет индекс -> ключ тот же -> неактивна
    c = np.array([1.5, -0.04, 2.0, 0.25, 9.0, -1.0])
    s, I, _ = _run(lambda x: float(c @ x), L, keyf=lambda x: tuple(x))
    step = [1, 1, 3, 8, None, 2]                 # |Δ индекса| при шаге +-2 уровня: L=2:1, L=5:3, L=13:8, L=4:2
    for a in (0, 1, 2, 3, 5):
        assert s["mu_star"][a] == pytest.approx(abs(c[a]) * step[a])
        assert s["U95"][a] == pytest.approx(abs(c[a]) * step[a])       # |EE| постоянен -> граница = среднее
        assert s["n_akt"][a] == 40
    assert s["n_akt"][4] == 0 and np.isnan(s["mu_star"][4]) and np.isnan(s["U95"][4])


def test_zero_axis_has_zero_mu_and_u95():
    L = [2, 2, 2]
    s, _, _ = _run(lambda x: 3.0 * x[0] + 2.0 * x[2], L, keyf=lambda x: tuple(x))
    assert s["mu_star"].tolist() == [3.0, 0.0, 2.0] and s["U95"][1] == 0.0 and s["n_akt"][1] == 40


def test_interaction_mu_star_matches_bruteforce_on_same_trajectories():
    L = [2, 2, 2]
    s, I, por = _run(lambda x: float(x[0] * x[1]), L, keyf=lambda x: tuple(x))
    for a in (0, 1):
        ee = []
        for tr in range(len(I)):
            t = int(np.flatnonzero(por[tr] == a)[0]) + 1
            ee.append(abs(float(I[tr, t, 0] * I[tr, t, 1]) - float(I[tr, t - 1, 0] * I[tr, t - 1, 1])))
        assert s["mu_star"][a] == pytest.approx(np.mean(ee))
        assert 0.25 < s["mu_star"][a] < 0.75          # ожидание 0,5: при шаге оси 0 эффект = x1 (равновероятно 0/1)
    assert s["mu_star"][2] == 0.0
    assert s["sigma"][0] > 0                          # взаимодействие -> эффект зависит от остальных осей


def test_child_axis_active_only_with_parent_on():
    """Ось 1 влияет на Y только при x[0]=1; ключ точки включает x1 только при x0=1 -> шаги оси 1 при x0=0 неактивны."""
    L = [2, 2]
    keyf = lambda x: (x[0], x[1] if x[0] == 1 else -1)          # noqa: E731
    s, I, por = _run(lambda x: 4.0 * x[0] * x[1] + 1.0 * x[0], L, keyf=keyf, r=60)
    n_on = 0
    for tr in range(len(I)):
        t = int(np.flatnonzero(por[tr] == 1)[0]) + 1
        n_on += int(I[tr, t - 1, 0] == 1)
    assert s["n_akt"][1] == n_on and 0 < n_on < 60
    assert s["mu_star"][1] == pytest.approx(4.0) and s["U95"][1] == pytest.approx(4.0)
    assert s["n_akt"][0] == 60                        # ось 0 всегда меняет ключ


def test_scale_invariance_mu_scales_linearly():
    L = [2, 5, 3]
    f = lambda x: float(x[0] + 0.5 * x[1] + x[0] * x[2])    # noqa: E731
    s1, _, _ = _run(f, L, keyf=lambda x: tuple(x))
    s2, _, _ = _run(lambda x: 10 * f(x), L, keyf=lambda x: tuple(x))
    assert np.allclose(s2["mu_star"], 10 * s1["mu_star"]) and np.allclose(s2["U95"], 10 * s1["U95"])
