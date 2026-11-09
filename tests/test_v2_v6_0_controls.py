"""V6-0, G1: контроли SBER восстановлены побитно; мутации — «прогноз сдвинут на +1 день» не пропускается, слепая проверка сдвига ловится, K8 другой формулы не равен сохранённому."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_audit as AU
from models_v2.s13 import v6_0_ctrl as CT, v6_0_load as LD


@pytest.fixture(scope="module")
def real():
    w, meta = LD.load()
    return w, meta, CT.controls_table(w)


def test_g1_real_controls_bitwise(real):
    w, _, (tab, res) = real
    assert res["files_ok"] and res["series_ok"] and res["k8_bitwise"] and res["shift_ok"] and res["ref_ok"]
    assert len(w) == 2271 and str(w.index.min().date()) == "2016-12-30" and str(w.index.max().date()) == "2026-02-26"
    for s, e in CT.EXPECT4.items():
        assert round(res["qlike"][s], 4) == e


def test_all_loaded_series_same_dates_target_and_complete(real):
    w, meta, _ = real
    assert (meta["tplus_mismatch"] == 0).all() and (meta["rv_maxdiff"].dropna() == 0).all()
    assert np.isfinite(w.drop(columns=["t_plus"]).to_numpy(float)).all()
    assert not any(s.startswith("J212Cg") for s in meta["series"])           # Cg хранит поправки g, не прогнозы


def test_mutation_shift_plus_one_is_caught_for_every_control(real):
    w = real[0]
    t = CT.shift_rows(w)
    assert t["real_ok"].all() and not t["mutation_plus1_passes"].any()


def test_mutation_blind_shift_check_is_caught(real, monkeypatch):
    """Проверка сдвига, пропускающая всё, обязана быть поймана: резюме shift_ok становится ложным."""
    monkeypatch.setattr(AU, "shift_ok", lambda *a, **k: True)
    t = CT.shift_rows(real[0])
    assert t["mutation_plus1_passes"].all() and not t["ok"].any()


def test_mutation_shifted_forecast_plus_one_day_not_passed_by_real_data(real):
    w = real[0]
    rv = pd.Series(w["rv"].to_numpy(float))
    f = pd.Series(w["K2"].to_numpy(float))
    assert AU.shift_ok(rv, f) and not AU.shift_ok(rv.iloc[:-1], f.shift(-1).iloc[:-1])


def test_mutation_k8_other_formula_not_equal_to_saved(real):
    w = real[0]
    arith = (w["K1"] + 0.5 * (w["K2"] + w["K3"])) / 2                      # другая формула вместо √(H·√(K2·K3))
    v4 = w["V4:K8"].to_numpy(float)
    assert np.array_equal(w["K8"].to_numpy(float), v4) and not np.array_equal(arith.to_numpy(float), v4)


def test_file_sha_mutation_caught(real, monkeypatch):
    monkeypatch.setattr(CT.IO, "sha_file", lambda p: "0" * 64)
    assert not CT.file_rows()["ok"].any()
