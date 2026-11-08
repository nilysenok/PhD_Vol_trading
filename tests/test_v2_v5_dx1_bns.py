"""V5, лист 1b, G17: статистика BNS, значимые скачки, правило п. 6.5. Синтетика без скачков (уровень 0,1 %) и со вставленным скачком; пустая лента, нулевой объём (нулевые доходности), перепутанный знак."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_dx1_bns as B, v5_dx1_jump as JM


def sim(n_days=20000, n=48, seed=0, jump_at=None, jump=0.0):
    rng = np.random.default_rng(seed)
    z = np.empty(n_days)
    for i in range(n_days):
        r = 0.001 * rng.normal(size=n)
        if jump_at is not None and i % jump_at == 0:
            r[n // 2] += jump
        z[i] = B.day_stat(r)[4]
    return z


def test_z_crit_is_0_1_percent():
    assert abs(B.Z_CRIT - 3.0902) < 1e-3


def test_no_jump_synthetic_size_is_small():
    z = sim()
    rate = float(np.mean(z > B.Z_CRIT))
    boot = np.random.default_rng(5).choice(z > B.Z_CRIT, size=(300, len(z))).mean(axis=1)
    lo, hi = np.quantile(boot, [0.005, 0.995])
    assert rate < 0.01 and hi < 0.01 and lo <= rate <= hi      # потолок: без поправки n/(n−1) статистика слегка завышена; доля значимых — порядка десятых процента


def test_inserted_jump_is_detected_and_sign_irrelevant():
    n = 48
    rng = np.random.default_rng(1)
    base = 0.001 * rng.normal(size=n)
    for j in (0.02, -0.02):
        r = base.copy()
        r[20] += j
        assert B.day_stat(r)[4] > B.Z_CRIT
    assert B.day_stat(base)[4] < B.Z_CRIT


def test_jump_synthetic_detection_rate():
    z = sim(n_days=2000, jump_at=2, jump=0.02, seed=2)
    assert np.nanmean(z[0::2] > B.Z_CRIT) > 0.95 and np.nanmean(z[1::2] > B.Z_CRIT) < 0.01


def test_scale_invariance_and_degenerate_days():
    r = 0.001 * np.random.default_rng(3).normal(size=30)
    assert np.isclose(B.day_stat(r)[4], B.day_stat(100 * r)[4], rtol=1e-9)
    assert np.isnan(B.day_stat(np.zeros(30))[4]) and np.isnan(B.day_stat(np.array([]))[4]) and np.isnan(B.day_stat(r[:3])[4])


def test_ticker_days_matches_project_rv_and_bv():
    from models_v2 import daily as D
    rng = np.random.default_rng(4)
    ix = pd.date_range("2020-03-02 10:00", periods=48, freq="10min")
    px = 100 * np.exp(np.cumsum(0.001 * rng.normal(size=48)))
    bars = pd.DataFrame({"begin": ix, "open": np.r_[100.0, px[:-1]], "close": px, "high": px, "low": px, "volume": 1.0})
    t = B.ticker_days(bars)
    d, _ = D.build_daily(bars, index_days=None)
    assert np.isclose(t["rv"].iloc[0], d["rv"].iloc[0], rtol=1e-12) and np.isclose(t["bv"].iloc[0], d["bv"].iloc[0], rtol=1e-12)


def _conc(share, ql_k8, ql_m, ev_min):
    g = pd.DataFrame([dict(section="conc", base=b, window="2017_2021", share_rows=share, share_qlike=q, share_under_in=0.5, share_under_all=0.3) for b, q in (("H", ql_k8), ("M", ql_m), ("K8", ql_k8))])
    ev = pd.DataFrame([dict(section="events", window="2017_2021", ticker="A", n_sig=ev_min), dict(section="events", window="2017_2021", ticker="B", n_sig=ev_min + 50)])
    return g, ev


def test_rule_all_three_conditions_needed():
    assert JM.decision(*_conc(0.05, 0.12, 0.11, 30))["verdict"] == "считать"
    assert JM.decision(*_conc(0.11, 0.30, 0.30, 99))["verdict"] == "закрыть"        # доля дней > 10 %
    assert JM.decision(*_conc(0.05, 0.12, 0.09, 99))["verdict"] == "закрыть"        # M ниже 2× доли
    assert JM.decision(*_conc(0.05, 0.12, 0.12, 29))["verdict"] == "закрыть"        # < 30 событий


def test_conc_known_answer_and_empty_flag():
    d = pd.date_range("2018-01-02", periods=10, freq="B")
    k = pd.DataFrame({"ticker": "A", "date": d - pd.offsets.BDay(1), "t_plus": d, "group": "M16", "rv_tplus": 1.0, "k1": 1.0, "k2": 1.0, "k3": 1.0, "d2": 1.0, "M": 1.0, "K8": 1.0})
    k.loc[[3, 4], "rv_tplus"] = 4.0
    sig = pd.DataFrame({"ticker": "A", "date": d, "sig": [False] * 3 + [True] * 2 + [False] * 5})
    c = JM.conc_rows(k.assign(t_plus=d), sig)
    r = c[(c["base"] == "K8") & (c["window"] == "2017_2021")].iloc[0]
    assert np.isclose(r["share_rows"], 0.2) and np.isclose(r["share_qlike"], 1.0) and np.isclose(r["share_under_in"], 1.0)
