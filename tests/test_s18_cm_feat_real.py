"""S-C-IMPL-1 (тестировщик), реальные данные C: sha входа, состав фактора (без SBER), нагрузки только по прошлому (пересчёт строк на реальных сессиях при cut и мутации LAG = 0 / SBER в пуле),
сверка с D_M и тождество c + p + u = RV. Только чтение; реальные выходы охраняет autouse."""
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import c_data, c_decomp as DC, c_factor as CFa, cm_feat as CF
from s18_cm_guard import guard_real_files  # noqa: F401  (autouse)

CUT = "2018-12-31"
FIELDS = ["rv_common", "rv_specific", "rv_uncovered"]


@pytest.fixture(scope="module")
def tab():
    return CF.load_c()


@pytest.fixture(scope="module")
def inp():
    return c_data.build_inputs(CUT)


def recompute(inp, n=12):
    """k = 1 строки последних n сессий окна `inp` -> массив (n, 3)."""
    s = range(len(inp.days) - n, len(inp.days))
    return np.array([[r[f] for f in FIELDS] for i in s for r in DC.session_rows(inp, i) if r["k_factors"] == 1])


def stored(tab, inp, n=12):
    return tab.loc[inp.days[len(inp.days) - n:], FIELDS].to_numpy(float)


def test_load_c_checks_sha_and_refuses_changed_input(tab, tmp_path, monkeypatch):
    assert CF.SHA_C == "c71410531eec3275" and len(tab) == 3124 and set(tab["k_factors"]) == {1}
    bad = tmp_path / "C_sessions.parquet"
    shutil.copy(CF.C_PARQ, bad)
    pd.read_parquet(CF.C_PARQ).assign(rv=lambda t: t["rv"] * 1.0000001).to_parquet(bad)
    with pytest.raises(SystemExit):
        CF.load_c(bad)


def test_factor_excludes_sber_and_loadings_use_strict_past():
    assert "SBER" not in c_data.POOL and len(c_data.POOL) == 16 and CFa.LAG == 1 and CFa.W_SESS == 120


def test_recomputed_rows_equal_stored_and_mutations_lag0_and_sber_in_pool_caught(tab, inp, monkeypatch):
    base = recompute(inp)
    assert np.abs(base - stored(tab, inp)).max() / base.max() <= 1e-9                 # честный пересчёт = записанное (cut не меняет прошлое)
    monkeypatch.setattr(CFa, "LAG", 0)
    mut = recompute(inp)
    assert np.abs(mut - stored(tab, inp)).max() / base.max() > 1e-6                   # LAG = 0: окно нагрузок включает s — расхождение
    monkeypatch.setattr(CFa, "LAG", 1)
    inp_s = c_data.build_inputs(CUT, pool=c_data.POOL + ("SBER",))
    assert np.abs(recompute(inp_s) - stored(tab, inp_s)).max() / base.max() > 1e-6    # SBER в факторе — расхождение


def test_identity_dm_grid_and_positive_components(tab):
    k = CF.components(tab)
    cov = np.isfinite(k["c"])
    assert abs(k["c"][cov] + k["p"][cov] + k["u"][cov] - k["rv"][cov]).max() / k["rv"][cov].max() <= 1e-13
    assert str(tab.index[np.argmax(cov)].date()) == "2014-08-27" and np.isnan(k["c"][:np.argmax(cov)]).all()
    dm = CF.dm_dates()
    assert len(dm) == 2198 and dm.isin(tab.index).all()
    j = np.flatnonzero(tab.index.isin(dm))
    assert (k["c"][j] > 0).all() and (k["p"][j] > 0).all() and (k["u"][j] >= 0).all()
