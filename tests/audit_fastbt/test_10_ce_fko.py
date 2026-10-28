"""10. Формулы CE и FKO.

CE (final_test.ce): CRRA, CE = (mean (1+r)^{1−γ})^{1/(1−γ)} − 1.
FKO (s9_stats.fko_fee, Fleming–Kirby–Ostdiek 2001): Δ из mean U(1+r1−Δ) = mean U(1+r0),
U(R) = R − γ/(2(1+γ))·R²; в % годовых; на полных доходностях total (после комиссий).
"""
import numpy as np
import pandas as pd

from fastbt import final_test, s9, s9_stats


def _ce_sign(r, gamma):
    w = np.maximum(1.0 + r, 1e-10)
    return np.mean(w ** (gamma - 1), axis=-1) ** (1 / (gamma - 1)) - 1


_orig_fko = s9_stats.fko_fee


def _fko_gamma(r1, r0, gamma):                 # коэффициент γ/(1+γ) вместо γ/(2(1+γ))
    return _orig_fko(r1, r0, 2 * gamma / (1 - gamma))
_orig_p3 = s9.S9.run_p3


def _p3_gross(self, pairs):
    return _orig_p3(self, {k: (a.assign(total=a["gross"]), b.assign(total=b["gross"]))
                           for k, (a, b) in pairs.items()})


MUTATIONS = {
    "ce_sign": lambda mp: mp.setattr(final_test, "ce", _ce_sign),
    "fko_gamma": lambda mp: mp.setattr(s9_stats, "fko_fee", _fko_gamma),
    "fko_gross": lambda mp: mp.setattr(s9.S9, "run_p3", _p3_gross),
}


def test_ce_by_hand():
    r = np.array([0.01, -0.01])
    # γ = 3: mean(1.01^-2, 0.99^-2) = 1.000300…; CE = that^(−1/2) − 1
    want = (0.5 * (1.01 ** -2 + 0.99 ** -2)) ** -0.5 - 1
    assert np.isclose(final_test.ce(r, 3), want)
    assert want < r.mean()                                           # штраф за риск
    assert np.isclose(final_test.ce(np.full(5, 0.002), 5), 0.002)     # без риска CE = r
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 1000)
    c = [final_test.ce(x, g) for g in (3, 5, 10)]
    assert c[0] > c[1] > c[2]                                        # убывает по γ


def test_fko_by_hand():
    rng = np.random.default_rng(1)
    r0 = rng.normal(0.0003, 0.01, 500)
    assert np.isclose(s9_stats.fko_fee(r0 + 0.0001, r0, 5), 0.0001 * 252 * 100)   # сдвиг = плата
    assert np.isclose(s9_stats.fko_fee(r0, r0, 5), 0.0)
    r1 = 0.0004 + 0.6 * (r0 - r0.mean())                            # выше среднее, ниже риск
    for g in (3, 10):
        dl = s9_stats.fko_fee(r1, r0, g) / 252 / 100
        a = g / (2 * (1 + g))
        U = lambda R: np.mean(R - a * R ** 2)
        assert np.isclose(U(1 + r1 - dl), U(1 + r0), rtol=0, atol=1e-12)
        assert dl > 0.0001


def test_p3_uses_net_total(monkeypatch):
    seen = []
    monkeypatch.setattr(s9_stats, "fko_fee", lambda a, b, g: seen.append((a, b)) or 0.0)
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    rng = np.random.default_rng(2)

    def port():
        g = rng.normal(0.0004, 0.01, len(idx))
        return pd.DataFrame({"gross": g, "r": g - 0.0002, "total": g - 0.0002 + 1e-4,
                             "excess": g - 0.0002, "exposure": 0.3, "turnover": 0.1}, index=idx)
    ph, pj = port(), port()
    fake = type("F", (), {"reps": 20})()
    s9.S9.run_p3(fake, {"x": (ph, pj)})
    assert len(seen) == 3
    for a, b in seen:
        assert np.array_equal(a, ph["total"].values) and np.array_equal(b, pj["total"].values)
