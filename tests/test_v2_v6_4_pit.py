"""V6-4, G36: пул аналогов и состояние Z_D (PIT). Пул дня d — только пары с t⁺ ≤ d, нормировка Z_D — по пулу дня d; мутации ловятся через боевой `v6_4_an.run`: «закрыто + 1», нормировка по всей выборке, аналог из будущего.
На синтетике численный минимум QLIKE по постоянному множителю равен среднему R (≤ 1e-8), и это же значение отдаёт K9."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models_v2.s13 import v6_0_q as Q, v6_1_wf as WF, v6_4_an as AN

N, D_POS = 260, 200


def panel(n: int = N, seed: int = 1):
    rng = np.random.default_rng(seed)
    r = pd.bdate_range("2015-07-07", periods=n + 1)
    idx, tp = r[:-1], r[1:]
    tab = pd.DataFrame({"t_plus": tp, "rv_next": np.exp(rng.normal(0, 0.5, n)), "h": np.exp(rng.normal(0, 0.3, n))}, index=idx)
    S = pd.DataFrame(rng.normal(size=(n, 9)), index=idx, columns=["s%d" % i for i in range(1, 10)])
    return tab, S


def perturbed(tab, S, pos: int = D_POS):
    """Будущее дня d = idx[pos]: цель строки d (RV на d+1) и всё после; признаки и H строк после d."""
    t2, s2 = tab.copy(), S.copy()
    t2.iloc[pos:, t2.columns.get_loc("rv_next")] *= 7.0
    t2.iloc[pos + 1:, t2.columns.get_loc("h")] *= 3.0
    s2.iloc[pos + 1:] += 50.0
    return t2, s2


def m_of(tab, S, d):
    return AN.run(tab, S, [d]).iloc[0]


def invariant(tab, S, pos: int = D_POS) -> bool:
    d = tab.index[pos]
    t2, s2 = perturbed(tab, S, pos)
    a, b = m_of(tab, S, d), m_of(t2, s2, d)
    return all(a[c] == b[c] for c in ("m_glob", "m1_50", "m2_100", "h"))


def test_forecast_of_day_d_ignores_future_rows():
    tab, S = panel()
    assert invariant(tab, S)


def test_pool_is_exactly_pairs_closed_by_d():
    tab, S = panel()
    for pos in (120, 200):
        d = tab.index[pos]
        assert int(m_of(tab, S, d)["n_pool"]) == int((tab["t_plus"] <= d).sum()) == pos            # строка d-1 закрыта в d, строка d — нет


def test_mutation_closed_plus_one_is_caught(monkeypatch):
    tab, S = panel()
    monkeypatch.setattr(AN, "closed_rows", lambda tp, d: WF.closed_rows(tp, pd.Timestamp(d) + pd.offsets.BDay(1)))
    assert not invariant(tab, S)                         # пара с t⁺ = d + 1 попала в пул: прогноз d зависит от RV следующего дня


def test_mutation_analog_from_future_is_caught(monkeypatch):
    tab, S = panel()
    monkeypatch.setattr(AN, "closed_rows", lambda tp, d: np.arange(len(tp)))
    assert not invariant(tab, S)


def test_normalisation_uses_pool_stats_and_whole_sample_mutation_is_caught(monkeypatch):
    tab, S = panel()
    Zp = S[AN.ZN].to_numpy()[:150]
    mu, sd = AN.norm_stats(Zp)
    assert np.allclose(mu, Zp.mean(0)) and np.allclose(sd, Zp.std(0))
    t2, s2 = perturbed(tab, S)
    d = tab.index[D_POS]

    def whole(S_):                                                  # МУТАНТ: среднее и СКО по всей выборке панели (вместе с будущим)
        Z = S_[AN.ZN].to_numpy()
        monkeypatch.setattr(AN, "norm_stats", lambda Zp_: (Z.mean(0), Z.std(0)))
        return m_of(tab, S_, d)

    assert whole(S)["h"] != whole(s2)["h"]                          # под мутантом прогноз дня d зависит от будущих признаков


def test_numeric_qlike_minimum_over_constant_multiplier_equals_mean_r():
    rng = np.random.default_rng(3)
    rv, h = np.exp(rng.normal(0, 0.7, 400)), np.exp(rng.normal(0, 0.4, 400))
    res = minimize_scalar(lambda c: float(Q.ql(rv, c * h).mean()), bounds=(0.1, 10.0), method="bounded", options=dict(xatol=1e-13))
    assert abs(res.x - float((rv / h).mean())) <= 1e-8


def test_k9_equals_argmin_of_pool_qlike_on_real_path():
    tab, S = panel()
    d = tab.index[D_POS]
    pool = tab[tab["t_plus"] <= d]
    res = minimize_scalar(lambda c: float(Q.ql(pool["rv_next"], c * pool["h"]).mean()), bounds=(0.1, 10.0), method="bounded", options=dict(xatol=1e-13))
    assert abs(res.x - m_of(tab, S, d)["m_glob"]) <= 1e-8


def test_terciles_thresholds_from_pool_h():
    hp = np.arange(1.0, 10.0)                                      # 1…9: пороги 3,67 и 6,33
    assert [AN.tercile(hp, x) for x in (1.0, 4.0, 9.0, 50.0, -3.0)] == [1, 2, 3, 3, 1]
