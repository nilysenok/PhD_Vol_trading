"""9. Бутстрап Ледуа–Вольфа (2008) для ΔШарп: HAC-с.о. (Бартлетт), студентизированный круговой
блочный бутстрап, блок 5 во всех местах основного p (final_test.lw_p, s9 тест, chapter4_a).
Эталон ниже написан заново по статье (разд. 3.1–3.2), не копией кода.
"""
import math

import numpy as np

from fastbt import chapter4_a, final_test, s9, s9_stats


def _force_block1(f):
    return lambda x1, x2, reps=1000, block=5, seed=42: f(x1, x2, reps, 1, seed)


MUTATIONS = {
    "lw_block1": lambda mp: mp.setattr(s9_stats, "lw_sharpe_diff",
                                       _force_block1(s9_stats.lw_sharpe_diff)),
    "lw_block1_callsite": lambda mp: mp.setattr(final_test, "BLOCK", 1),
}


def ref_lw(x1, x2, reps, b, seed):
    T = len(x1)

    def parts(a, c):
        mu, g = np.array([a.mean(), c.mean()]), np.array([(a * a).mean(), (c * c).mean()])
        v = g - mu ** 2
        d = mu[0] / math.sqrt(v[0]) - mu[1] / math.sqrt(v[1])
        grad = np.array([g[0] / v[0] ** 1.5, -g[1] / v[1] ** 1.5,
                         -mu[0] / (2 * v[0] ** 1.5), mu[1] / (2 * v[1] ** 1.5)])
        return d, grad, np.stack([a - mu[0], c - mu[1], a * a - g[0], c * c - g[1]], 1)

    d, grad, Y = parts(x1, x2)
    L = int(4 * (T / 100) ** (2 / 9))
    S = np.zeros((4, 4))
    for k in range(-L, L + 1):
        w = 1 - abs(k) / (L + 1)
        G = sum(np.outer(Y[t], Y[t - k]) for t in range(max(k, 0), T + min(k, 0))) / T
        S += w * G
    se = math.sqrt(grad @ S @ grad / T)
    rng = np.random.RandomState(seed)
    l = T // b
    cnt = 0
    for _ in range(reps):
        starts = rng.randint(0, T, l)
        idx = np.concatenate([np.arange(s, s + b) % T for s in starts])
        ds, gs, Ys = parts(x1[idx], x2[idx])
        zeta = np.array([Ys[j * b:(j + 1) * b].sum(0) / math.sqrt(b) for j in range(l)])
        psi = zeta.T @ zeta / l
        ses = math.sqrt(gs @ psi @ gs / (l * b))
        cnt += abs(ds - d) / ses >= abs(d) / se
    return d * math.sqrt(252), se * math.sqrt(252), (cnt + 1) / (reps + 1)


def _data(T=300, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(T + 4, 2))
    ma = (e[4:] + e[3:-1] + e[2:-2] + e[1:-3] + e[:-4]) / 5          # зависимость до лага 4
    return 0.0006 + 0.01 * ma[:, 0], 0.0002 + 0.01 * (0.6 * ma[:, 0] + 0.8 * ma[:, 1])


def test_against_reference_block5():
    x1, x2 = _data()
    got = s9_stats.lw_sharpe_diff(x1, x2, 200, 5, 42)
    ref = ref_lw(x1, x2, 200, 5, 42)
    assert np.allclose(got, ref, rtol=1e-9)
    sr = lambda x: x.mean() / x.std() * math.sqrt(252)
    assert np.isclose(got[0], sr(x1) - sr(x2))


def test_identical_series_p_is_one():
    x1, _ = _data()
    assert s9_stats.lw_sharpe_diff(x1, x1.copy(), 50, 5)[2] == 1.0


def test_call_sites_use_block5(monkeypatch):
    seen = []
    orig = s9_stats.lw_sharpe_diff

    def spy(x1, x2, reps=1000, block=5, seed=42):
        seen.append(block)
        return orig(x1, x2, reps, block, seed)
    monkeypatch.setattr(s9_stats, "lw_sharpe_diff", spy)
    x1, x2 = _data()
    final_test.lw_p(x1, x2, 20)
    fake = type("F", (), {"reps": 20, "boot": lambda self, a, b: (0.0, 0.5)})()
    monkeypatch.setattr(s9, "SCOPE", "test")
    import pandas as pd
    r = {}
    s9.S9.delta(fake, r, "vs_A", pd.DataFrame({"excess": x1}), pd.DataFrame({"excess": x2}))
    assert seen == [5, 5] and chapter4_a.LW_BLOCK == 5 and "p_vs_A_iid" in r
