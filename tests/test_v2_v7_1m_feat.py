"""V7 лист 1a-M, G18 (признаки): Z (9) по закрытым парам — известные ответы и PIT. EWMA₂₀ и CUSUM берут только закрытые пары (t⁺ ≤ d); мутант «текущая пара» ловится. Синтетика."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_cld_brk as BK
from models_v2.s14 import v7_1m_feat as FE, v7_1m_syn as SY

N, J = 330, 150


@pytest.fixture(scope="module")
def s():
    return SY.make(N, 5)


def test_columns_and_closed_pair_known_answers(s):
    tab, Z = s["tab"], FE.build_Z(s["tab"])
    assert list(Z.columns) == FE.ZCOLS and Z.shape[1] == 9
    H, K2, K3, rv = (tab[c].to_numpy() for c in ("H", "K2", "K3", "rv_next"))
    M = np.sqrt(K2 * K3)
    i = 200
    assert Z["ln_H"].iloc[i] == pytest.approx(np.log(H[i])) and Z["ell"].iloc[i] == pytest.approx(np.log(M[i] / H[i])) and Z["ln_K2_K3"].iloc[i] == pytest.approx(np.log(K2[i] / K3[i]))
    assert Z["last_err_H"].iloc[i] == pytest.approx(np.log(rv[i - 1] / H[i - 1]))                 # последняя закрытая пара на d_i — пара строки i − 1 (её t⁺ = d_i)
    assert Z["last_err_M"].iloc[i] == pytest.approx(np.log(rv[i - 1] / M[i - 1]))
    assert Z["jshare"].iloc[i] == tab["jshare"].iloc[i]
    q = lambda f: rv / f - np.log(rv / f) - 1
    ew = pd.Series((q(H) - q(M))[:i]).ewm(halflife=20, adjust=False).mean().iloc[-1]
    assert Z["ewma_dq"].iloc[i] == pytest.approx(ew, rel=1e-12)
    past = np.log(tab["rv_d"].to_numpy()[:i])
    assert Z["pct_lnRV"].iloc[i] == pytest.approx(np.mean(past[np.isfinite(past)] < np.log(tab["rv_d"].iloc[i])))
    cu = BK.cusum(pd.Series(np.log(rv / H)[:i], index=pd.DatetimeIndex(tab["t_plus"].to_numpy()[:i])))[0].iloc[-1]
    assert Z["cusum_errH"].iloc[i] == pytest.approx(cu, rel=1e-12)


def test_first_rows_without_closed_pairs_are_nan_not_zero(s):
    Z = FE.build_Z(s["tab"])
    assert Z.iloc[0][["ewma_dq", "last_err_H", "last_err_M", "cusum_errH"]].isna().all() and Z["pct_lnRV"].iloc[:FE.MIN_PCT].isna().all()


def test_future_targets_do_not_change_past_features(s):
    s2 = SY.perturb_future(s, J)
    Za, Zb = s["Zg"], s2["Zg"]
    assert np.array_equal(Za[:J + 1], Zb[:J + 1], equal_nan=True)                              # строки ≤ J не зависят от пары J и позже
    assert not np.array_equal(Za[J + 1:], Zb[J + 1:], equal_nan=True)                          # возмущение видно позже (тест не пустой)


def test_mutation_current_pair_in_ewma_and_cusum_caught(monkeypatch):
    monkeypatch.setattr(FE, "last_closed", lambda tp, d: np.arange(len(tp)))                   # мутант: «текущая» пара вместо последней закрытой
    a = SY.make(N, 5)
    b = SY.perturb_future(a, J)
    assert not np.array_equal(a["Zg"][:J + 1], b["Zg"][:J + 1], equal_nan=True)


def test_mutation_percentile_with_current_day_changes_value(s, monkeypatch):
    """Перцентиль ln RV_d строго по прошлому: с днём d в окне значение другое."""
    x = np.log(s["tab"]["rv_d"].to_numpy())
    good = FE.pct_past(x)
    bad = np.array([np.mean(x[:i + 1][np.isfinite(x[:i + 1])] < x[i]) if i > 25 else np.nan for i in range(len(x))])
    assert not np.allclose(good[30:], bad[30:], equal_nan=True)


def test_last_closed_handles_gaps_in_calendar():
    d = pd.DatetimeIndex(["2018-01-02", "2018-01-03", "2018-01-09", "2018-01-10"])
    tp = pd.DatetimeIndex(["2018-01-03", "2018-01-09", "2018-01-10", "2018-01-11"])
    assert list(FE.last_closed(tp.to_numpy("datetime64[ns]"), d.to_numpy("datetime64[ns]"))) == [-1, 0, 1, 2]
