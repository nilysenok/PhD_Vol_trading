"""V4-E, G23/G24: боевой путь `v4_e_run.forecast` на реальных данных (3 бумаги, 2017–2018): обрезка RV, мутация «закрыто + 1», тождества b ≡ 0, ключ кэша (Д5)."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_data as D, v4_e_cache as EC, v4_e_calib as CAL, v4_e_go as GO, v4_e_run as R

DAY = pd.Timestamp("2018-09-03")


@pytest.fixture(scope="module")
def data():
    P = D.panel(["SBER", "LKOH", "AFLT"])
    P = P[P["date"] <= "2018-12-31"].reset_index(drop=True)
    G = pd.read_parquet(EC.OUT / "B_B2.parquet")
    return P, G[G["ticker"].isin(["SBER", "LKOH", "AFLT"])]


@pytest.mark.parametrize("cfg", R.CFGS)
def test_truncation_real_path(cfg, data):
    assert not R.pit_violation(cfg, *data, DAY)


@pytest.mark.parametrize("cfg", ["E1", "E3", "E5", "E6"])
def test_mutation_closed_plus_one_caught_real_path(cfg, data):
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):
        assert R.pit_violation(cfg, *data, DAY)


def test_identities_b_zero(data):
    P, G = data
    Pm = P.merge(G[["ticker", "date", "F"]], on=["ticker", "date"]).sort_values(["date", "ticker"])
    assert np.array_equal(R.forecast("E1", P, G, zero_b=True)[0]["F"].to_numpy(), Pm["h"].to_numpy())
    assert np.array_equal(R.forecast("E2", P, G, zero_b=True)[0]["F"].to_numpy(), Pm["F"].to_numpy())


def test_loto_panel_excludes_held_out_real_path(data):
    P, G = data
    Q = P.copy()
    Q.loc[Q["ticker"] == "LKOH", "rv"] *= 5.0
    a, b = R.forecast("E3", P, G, exclude="LKOH")[0], R.forecast("E3", Q, G, exclude="LKOH")[0]
    assert np.array_equal(a["c_panel"].to_numpy(), b["c_panel"].to_numpy())
    assert R.forecast("E3", P, G, exclude="LKOH")[1]["panel"] == ["AFLT", "SBER"]


def test_cache_key_tracks_inputs_and_code(tmp_path):
    f1, f2, c = tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "v4_e_x.py"
    f1.write_bytes(b"1")
    f2.write_bytes(b"2")
    c.write_text("x=1")
    k = EC.keys(extra_in=[f1], extra_code=[c])["key"]
    assert EC.keys(extra_in=[f2], extra_code=[c])["key"] != k
    c.write_text("x=2")
    assert EC.keys(extra_in=[f1], extra_code=[c])["key"] != k


def test_go_rules_trim_drops_extreme_rows_and_known_answer():
    n = 1000
    k = pd.DataFrame({"ticker": ["A"] * n, "date": pd.date_range("2017-01-02", periods=n), "t_plus": pd.date_range("2017-01-03", periods=n), "group": "M16", "rv_tplus": np.ones(n),
                      "k2": 1.0, "k3": 1.0, "d2": 1.0, "E": 1.0})
    k.loc[300:309, "E"] = 100.0                               # 10 строк из 1000 = 1 %: единственные крайние
    t = GO.crit_trim(k, ["k2", "E"])
    assert t["k2"] < 1e-9 and t["E"] < 1e-9                   # без крайних 1 % ряд E совпадает с лучшим из K2/K3 (QLIKE = 0)
    assert GO.crit_trim(k, ["k2", "E"], frac=0.0)["E"] > 1e-3
