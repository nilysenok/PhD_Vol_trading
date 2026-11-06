"""Ж-212: таблица ERROR_STRUCTURE_DAILY.parquet и контрольные значения на реальных замороженных компонентах (без сети)."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s6.e import comp
from models_v2.s8 import hs_ctrl, hs_paths as P

TAB = P.OUT / "ERROR_STRUCTURE_DAILY.parquet"
pytestmark = pytest.mark.skipif(not TAB.exists() or not P.COMP.exists(), reason="нет данных программы Ж-212")


@pytest.fixture(scope="module")
def o():
    return pd.read_parquet(TAB)


def test_control_values_match_declared_four_decimals():
    t = hs_ctrl.check().set_index("model")
    assert t["match_4dp"].all() and len(t) == 4 and (t["n"] == 2271).all()


def test_control_mutation_wrong_declared_is_caught(monkeypatch):
    monkeypatch.setitem(hs_ctrl.DECLARED, "XJ", 0.2114)
    assert not hs_ctrl.check().set_index("model").loc["XJ", "match_4dp"]


def test_table_columns_equal_frozen_components_and_required_fields(o):
    c = comp.load()
    for k in ("rv", "HAR", "XJ", "LF"):
        assert np.array_equal(o[k].to_numpy(), c[k].to_numpy())
    need = ("e_HAR", "q_HAR", "q_XJ", "q_LF", "dq_HAR_XJ", "ln_rv_d", "ln_rv_w", "ln_rv_m", "ln1p_J", "rs_plus", "rs_minus", "mkt_rv", "mkt_rv_w", "ter_rv_w", "rvi_regime", "st_g4",
            "F_f1_lag1", "F_f6_har_bias22", "past_eH_last", "past_dqX_m22")
    assert all(k in o.columns for k in need) and len(o) == 2271


def test_table_qlike_uses_project_definition(o):
    assert np.allclose(o["q_XJ"], metrics.qlike(o["rv"], o["XJ"]))
    assert o["q_HAR"].mean() == pytest.approx(0.2337, abs=5e-5)


def test_table_past_errors_are_lagged_one_row_on_consecutive_grid(o):
    """target_date_j <= d_{j+1} (в паузе 2022 строго меньше) и target_date_k > d_k: на d_k закрыты ровно строки 0..k−1, значит past_eH_last[k] = e_HAR[k−1]; строка k НЕ входит."""
    assert (o["target_date"].iloc[:-1].to_numpy() <= o["d"].iloc[1:].to_numpy()).all() and (o["target_date"] > o["d"]).all()
    assert np.array_equal(o["past_eH_last"].to_numpy()[1:], o["e_HAR"].to_numpy()[:-1])
    assert np.isnan(o["past_eH_last"].iloc[0])


def test_table_market_rv_is_value_on_d_not_next_day(o):
    from models_v2 import daily
    idly, _ = daily.build_daily(pd.read_parquet(P.IMOEX), index_days=None)
    k = 500
    assert o["mkt_rv"].iloc[k] == idly["rv"].loc[o["d"].iloc[k]]
    assert o["mkt_rv"].iloc[k] != idly["rv"].loc[o["target_date"].iloc[k]]


def test_table_no_nan_in_core_and_positive(o):
    core = o[["rv", "HAR", "XJ", "LF", "q_HAR", "q_XJ", "q_LF", "ln_rv_d", "ln_rv_w", "ln_rv_m", "mkt_rv"]]
    assert core.notna().all().all() and (o[["rv", "HAR", "XJ", "LF"]] > 0).all().all()
