"""Ж-044: ворота train (zh043_itog.gates). Синтетика. Ворота Ж-044: PBO < 0,2 · плато ≥ 3 · сделок ≥ 30 в год · s9; DSR не ворота."""
import inspect
import sys

import pytest

sys.path[:0] = [".", "scripts"]
import zh043_itog as I                                  # noqa: E402


def cand(**kw):
    c = dict(obj=0.5, msr_2018=0.5, msr_2019=0.6, sosedey_pos=5, sosedey=6, aktivnost_ok=1)
    c.update(kw)
    return c


def ref(**kw):
    r = [dict(sdelok_god=40.0, sverka_s9="sovpalo", max_dpnl=0.0, bar_poz_razl=0) for _ in range(2)]
    for k, v in kw.items():
        r[0][k] = v
    return r


def test_baseline_point_passes_trade_plato_s9():
    g = I.gates(cand(), ref(), 0.95)
    assert g["G2"] == g["G3"] == g["G4"] == 1


def test_gate_trades_below_30_in_one_year_cuts():
    r = ref()
    r[1]["sdelok_god"] = 29.9
    g = I.gates(cand(), r, 0.95)
    assert g["G2"] == 0 and not g["ok"]
    r[1]["sdelok_god"] = 30.0
    assert I.gates(cand(), r, 0.95)["G2"] == 1


def test_gate_plato_fewer_than_3_neighbours_cuts():
    g = I.gates(cand(sosedey_pos=2), ref(), 0.95)
    assert g["G3"] == 0 and not g["ok"]
    assert I.gates(cand(sosedey_pos=3), ref(), 0.95)["G3"] == 1


@pytest.mark.parametrize("kw", [dict(sverka_s9="razoshlos"), dict(max_dpnl=1e-3), dict(bar_poz_razl=1)])
def test_gate_s9_mismatch_cuts(kw):
    g = I.gates(cand(), ref(**kw), 0.95)
    assert g["G4"] == 0 and not g["ok"]


def test_gate_trades_mutation_threshold_caught(monkeypatch):
    monkeypatch.setattr(I, "MIN_TRADES", 10)
    r = ref()
    r[0]["sdelok_god"] = 20.0
    assert I.gates(cand(), r, 0.95)["G2"] == 1          # мутация проходит; настоящий порог 30 — см. тест выше
    assert I.MIN_TRADES != 30


def test_gate_plato_mutation_threshold_caught(monkeypatch):
    monkeypatch.setattr(I, "MIN_NB", 1)
    assert I.gates(cand(sosedey_pos=2), ref(), 0.95)["G3"] == 1
    monkeypatch.undo()
    assert I.MIN_NB == 3 and I.gates(cand(sosedey_pos=2), ref(), 0.95)["G3"] == 0


def test_dsr_001_does_not_cut_when_others_pass():
    assert I.gates(cand(), ref(), 0.01)["ok"] is True


def test_negative_min_msr_does_not_cut():
    assert I.gates(cand(obj=-0.2, msr_2018=-0.2), ref(), 0.01)["ok"] is True


def test_gate_pbo_at_least_02_cuts():
    assert "pbo" in inspect.signature(I.gates).parameters
    assert I.gates(cand(), ref(), 0.95, pbo=0.2)["ok"] is False
    assert I.gates(cand(), ref(), 0.95, pbo=0.19)["ok"] is True
    assert I.gates(cand(), ref(), 0.95, pbo=float("nan"))["ok"] is False


def test_gate_pbo_mutation_threshold_caught(monkeypatch):
    monkeypatch.setattr(I, "PBO_MAX", 0.5)
    assert I.gates(cand(), ref(), 0.95, pbo=0.3)["ok"] is True
    monkeypatch.undo()
    assert I.gates(cand(), ref(), 0.95, pbo=0.3)["ok"] is False
