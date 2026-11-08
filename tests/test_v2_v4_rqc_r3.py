"""V4-RQC (G35): RQC3 — известный ответ (рамп шага 0,02, заморозка после скачка), мутации: без заморозки, шаг > 0,02, процентили обрезки по будущему. Через боевой путь `v4_rqc_run.forecast`."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd

from models_v2.s11 import v4_rqc_r3 as R3, v4_rqc_run as R
from tests.test_v2_v4_rqc_run import synth


def _flat_z(P):
    return pd.DataFrame({"ticker": P["ticker"], "date": P["date"], "jshare": 0.1, "dlrv": 0.0})


def _la(out, t="AAA"):
    return np.log(out[out["ticker"] == t].sort_values("date")["a"].to_numpy())


def _jump_panel(r0=70):
    """x = 1,6 (a растёт до ln 1,6 шагом 0,02 с дня 60); у AAA в день r0 — скачок доли J (остальные дни константа)."""
    P, _ = synth(n=200, ratio=1.6, sig=0.0)
    Z = _flat_z(P)
    Z.loc[(Z["ticker"] == "AAA") & (Z["date"] == sorted(P["date"].unique())[r0]), "jshare"] = 0.9
    return P, Z


def test_ramp_known_answer_step_002_until_target():
    P, _ = synth(n=200, ratio=1.3, sig=0.0)
    la = _la(R.forecast("RQC3", P, _flat_z(P))[0])
    want = np.minimum(0.02 * np.maximum(np.arange(200) - 59, 0), np.log(1.3))        # к дню r закрыто r пар бумаги; активна с r = 60
    assert np.allclose(la, want, atol=1e-12)


def test_freeze_five_days_after_jump_and_resume():
    P, Z = _jump_panel(70)
    out = R.forecast("RQC3", P, Z)[0]
    la = _la(out)
    assert np.allclose(la[69:75], 0.02 * 10) and np.isclose(la[75], 0.02 * 11)          # дни 70…74 заморожены, на 75-й — снова шаг
    fz = out[out["ticker"] == "AAA"].sort_values("date")["frozen"].to_numpy()
    assert fz[70:75].all() and not fz[69] and not fz[75]
    assert np.allclose(_la(out, "BBB")[69:76], 0.02 * np.arange(10, 17))               # у других бумаг заморозки нет


def test_mutation_no_freeze_is_caught():
    P, Z = _jump_panel(70)
    with mock.patch.object(R3, "FREEZE", 0):
        la = _la(R.forecast("RQC3", P, Z)[0])
    assert not np.allclose(la[69:75], 0.2)


def test_mutation_step_larger_than_002_is_caught():
    P, Z = _jump_panel(70)
    with mock.patch.object(R3, "STEP", 0.05):
        la = _la(R.forecast("RQC3", P, Z)[0])
    assert np.abs(np.diff(la)).max() > 0.02 + 1e-9
    assert np.abs(np.diff(_la(R.forecast("RQC3", P, Z)[0]))).max() <= 0.02 + 1e-12


def test_jump_by_abs_dlrv_also_freezes():
    P, _ = synth(n=200, ratio=1.6, sig=0.0)
    Z = _flat_z(P)
    Z.loc[(Z["ticker"] == "AAA") & (Z["date"] == sorted(P["date"].unique())[70]), "dlrv"] = 2.0
    la = _la(R.forecast("RQC3", P, Z)[0])
    assert np.allclose(la[69:75], 0.2)


def test_winsorize_bounds_from_past_only_mutation_future_percentiles_is_caught():
    P, Z = synth(n=320)
    day = P["date"].iloc[200]
    assert not R.pit_violation("RQC3", P, Z, day)
    with mock.patch.object(R3, "window", lambda hist, full: full):
        assert R.pit_violation("RQC3", P, Z, day)


def test_winsorized_value_is_clipped_to_past_quantiles():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 0.1, 200)
    y[150] = 5.0
    yw, _ = R3.prep_ticker(y, np.full(200, 0.1), np.zeros(200))
    hi = np.quantile(y[:150], R3.Q_HI)
    assert yw[150] == hi and np.array_equal(yw[:60], y[:60])
