"""G29: DM (Ньюи–Уэст, лаг 5) на синтетике с известным ответом; Холм."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import metrics  # noqa: E402
from models_v2.s3 import dm  # noqa: E402


def test_nw_variance_matches_manual_lag5_and_iid_limit():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0, 1.0])
    u = x - x.mean()
    n = len(x)
    g = [float((u[l:] * u[:n - l]).sum()) / n for l in range(6)]
    want = (g[0] + 2 * sum((1 - l / 6) * g[l] for l in range(1, 6))) / n
    assert dm.nw_variance(x) == pytest.approx(want, rel=1e-12)
    z = np.random.default_rng(0).normal(size=20000)
    assert dm.nw_variance(z) == pytest.approx(1 / 20000, rel=0.15)


def test_dm_sign_and_p_on_known_difference():
    rng = np.random.default_rng(1)
    rv = np.exp(rng.normal(0, 0.5, 2000))
    good, bad = rv * np.exp(rng.normal(0, 0.1, 2000)), rv * np.exp(rng.normal(0, 0.6, 2000))
    t = dm.dm(rv, bad, good)  # модель (good) лучше базы (bad): плюс
    assert t["sign"] == 1 and t["mean_d"] > 0 and t["p"] < 1e-6
    assert dm.dm(rv, good, bad)["sign"] == -1
    same = dm.dm(rv, good, good * np.exp(rng.normal(0, 1e-3, 2000)))
    assert same["p"] > 0.01 and metrics.qlike(1.0, 1.0) == 0.0


def test_dm_null_rejection_rate_is_near_nominal():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(200):
        rv = np.exp(rng.normal(0, 0.5, 400))
        a, b = rv * np.exp(rng.normal(0, 0.4, 400)), rv * np.exp(rng.normal(0, 0.4, 400))
        hits += dm.dm(rv, a, b)["p"] < 0.05
    assert 2 <= hits <= 22  # номинал 5 % от 200 = 10; широкие границы


def test_dm_p_matches_normal_and_degenerate_cases():
    rv = np.array([1.0, 2.0, 1.5, 0.7, 1.2, 1.1, 0.9, 1.3])
    k = np.array([1.5, 0.8, 1.2, 1.7, 0.9, 1.1, 1.4, 0.6])
    r = dm.dm(rv, rv * k, rv * 0.9)
    assert math.isnan(dm.dm(rv, rv * 1.5, rv * 0.9)["p"])  # постоянная разность: дисперсия 0, статистики нет
    assert r["p"] == pytest.approx(math.erfc(abs(r["stat"]) / math.sqrt(2)))
    e = dm.dm(np.array([]), np.array([]), np.array([]))
    assert e["n"] == 0 and math.isnan(e["p"])


def test_holm_monotone_and_not_below_p():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    h = dm.holm(p)
    assert (h >= p).all() and h.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.5]) and dm.holm(np.array([0.9]))[0] == 0.9
