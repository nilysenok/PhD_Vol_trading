"""V6-1a, G10: прогрев H той же функцией `har.forecast_at`, что K1; пересчёт = K1 побитно на всех датах K1 до 2017-03-31; SHA файла K1 прежний; 377 дат прогрева.
Мутации: прогноз другой функцией модели (harj_a вместо harj_b) и окно с будущей парой (t⁺ = d + 1) ломают побитное совпадение — ловятся через боевую `check_h`."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2 import har as H
from models_v2.s13 import v6_0_io as IO, v6_1_h as HH


@pytest.fixture(scope="module")
def real():
    p, k1 = H.make_panel(HH.daily_sber()), HH.k1_stored()
    return p, k1, HH.h_table(p, k1)


def test_warmup_dates_and_bitwise_k1(real):
    p, k1, tab = real
    c = HH.check_h(tab, k1)
    assert c["n_warm"] == HH.N_WARM == 377 and c["first_warm"] == "2015-07-07" and c["last_warm"] == "2016-12-29"
    assert c["n_early"] >= 60 and c["max_diff_early"] == 0.0 and c["max_diff_all"] == 0.0


def test_k1_file_sha_unchanged():
    prior = IO.read_csv("CONTROLS_SBER.csv").set_index("item").loc["SBER:K1", "sha_v5"]
    assert IO.sha_file(HH.IO4.path_of("H", "SBER")) == prior


def test_warm_h_uses_forecast_at_and_k1_for_later_dates(real, monkeypatch):
    p, k1, tab = real
    assert (tab.loc[tab.index < HH.K1_FIRST, "src"] == "warm").all() and (tab.loc[tab.index >= HH.K1_FIRST, "src"] == "K1").all()
    assert np.array_equal(tab.loc[tab.index >= HH.K1_FIRST, "h"].to_numpy(), k1.reindex(tab.index[tab.index >= HH.K1_FIRST]).to_numpy())
    calls = []
    real_fa = H.forecast_at
    monkeypatch.setattr(H, "forecast_at", lambda pp, i: calls.append(i) or real_fa(pp, i))
    HH.forecast_rows(p, tab["i"].to_numpy()[:5])
    assert calls == [int(i) for i in tab["i"].to_numpy()[:5]]


def test_mutation_other_model_function_caught(real, monkeypatch):
    p, k1, _ = real
    monkeypatch.setattr(HH, "forecast_rows", lambda pp, rows: np.array([H.forecast_at(pp, int(i))["harj_a"] for i in rows]))
    bad = HH.h_table(p, k1)
    assert HH.check_h(bad, k1)["max_diff_early"] > 0.0


def test_mutation_future_pair_in_window_caught(real, monkeypatch):
    p, k1, _ = real
    real_pairs = H.pairs
    monkeypatch.setattr(H, "pairs", lambda pp, i, cols, upto=None: real_pairs(pp, i, cols, upto=i + 1))        # пара (d, t⁺ = d + 1) уже в окне
    assert HH.check_h(HH.h_table(p, k1), k1)["max_diff_early"] > 0.0
