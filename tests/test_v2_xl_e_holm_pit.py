"""Волна E Ж-206, замечания проверяющего: (1) st_e3 применяет Холм с N = 182 уникальных trials (75 + 72 + 35), мутация N = число строк ловится;
(2) offset lb остатка B1q/B2q = ln PIT-прогноза HAR-J(б) (s2_harj) до 1e-12, подгонка на всей выборке (full=True) ловится."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import run_fix
from models_v2.s3 import resid as s3resid
from models_v2.s6.e import pe, resid, st_e3

N_TRIALS = 182          # 75 (xgboost) + 72 (lightgbm) + 35 (xgb_lgbm_ensemble)
NP = 20


def frame():
    rows = []
    for i in range(NP):
        rows.append({"candidate": "C%d" % i, "reference": "HAR", "period": "вся история", "p_raw": 0.001})
        rows.append({"candidate": "C%d" % i, "reference": "HAR", "period": "2022", "p_raw": 0.001})
    return pd.DataFrame(rows)


def test_n_trials_is_182_unique_trials_of_three_registries():
    assert pe.n_trials() == N_TRIALS


def test_holm_uses_n_trials_not_row_count():
    t = st_e3.holm_primary(frame())
    prim = t["period"] == "вся история"
    assert (t["n_hypotheses_holm"] == N_TRIALS).all()
    assert np.isclose(t.loc[prim, "p_holm"].min(), 0.001 * N_TRIALS)       # 0,182: при N = 20 было бы 0,02
    assert not t.loc[prim, "significant_holm_0.05"].any()
    assert t.loc[~prim, "p_holm"].isna().all()


def test_mutation_holm_n_equals_row_count_is_caught(monkeypatch):
    monkeypatch.setattr(pe, "n_trials", lambda: 0)       # поведение мутации N = len(prim): N = 20
    t = st_e3.holm_primary(frame())
    assert (t["n_hypotheses_holm"] == NP).all()
    with pytest.raises(AssertionError):
        assert (t["n_hypotheses_holm"] == N_TRIALS).all()
    assert t.loc[t["period"] == "вся история", "significant_holm_0.05"].all()      # именно это даёт завышенную значимость


@pytest.fixture(scope="module")
def world():
    return resid.build_world()


def lb_diff(W):
    lb = W["dz"].off[W["grid"]]
    return float(np.max(np.abs(lb - np.log(W["ctrl"]["s2_harj"].to_numpy(float)))))


def test_resid_offset_is_pit_harj_b(world):
    assert lb_diff(world) < 1e-12


def test_mutation_harj_full_sample_is_caught(monkeypatch):
    orig = s3resid.harj_ab
    monkeypatch.setattr(run_fix.resid, "harj_ab", lambda p, *a, **k: orig(p, full=True))
    assert lb_diff(resid.build_world()) > 1e-6
