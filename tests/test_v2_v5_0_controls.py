"""V5-0, G1: контроли восстановлены побитно; мутации — «прогноз сдвинут на +1 день» не пропускается, K8 другой формулы не равен сохранённому."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s11 import v4_audit as AU, v4_rqc_calib as RQ
from models_v2.s12 import v5_0_ctrl as CT, v5_0_io as IO


@pytest.fixture(scope="module")
def real():
    k = IO.frame()
    return k, CT.controls_table(k)


def test_g1_real_controls_bitwise(real):
    _, (tab, res) = real
    assert res["n_tickers"] == 17 and res["n_files"] == 68 and res["files_ok"]
    assert res["k8_bitwise"] and res["audit_ok"] and res["shift_ok"] and res["sber_ok"]
    assert res["crit_ok"] and res["crit_v4_diff"] <= 1e-12
    for s, e in CT.EXPECT.items():
        assert round(float(res["crit"][s]), 6 if s == "d2" else 5) == e


def test_sber_reference_2271_dates(real):
    _, (_, res) = real
    assert res["sber"]["n_common"] == 2271 and abs(res["sber"]["repro_diff"]) < 1e-4


def test_mutation_shift_plus_one_is_caught_per_ticker():
    t = CT.shift_rows()
    assert t["real_ok"].all() and not t["mutation_plus1_passes"].any()


def test_mutation_blind_shift_check_is_caught(monkeypatch):
    """Проверка сдвига, пропускающая всё, обязана быть поймана: мутация +1 «проходит» -> shift_ok резюме ложно."""
    monkeypatch.setattr(AU, "shift_ok", lambda *a, **k: True)
    t = CT.shift_rows()
    assert t["mutation_plus1_passes"].all()      # слепая проверка: мутация не поймана -> резюме G1 станет False


def test_mutation_k8_other_formula_not_bitwise(real, monkeypatch):
    k, _ = real
    assert CT.k8_rows(k)[0] is True
    monkeypatch.setattr(RQ, "k8", lambda h, k2, k3: np.sqrt(np.asarray(h) * np.asarray(k2)))      # K8 без K3
    assert CT.k8_rows(k)[0] is False
