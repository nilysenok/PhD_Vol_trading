"""R-д: --period доходит до progon_sobol.PERIOD на время расчёта и не оставляет глобального следа (mutations.md № 43). Моки, данные не читаются."""
import contextlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/revers"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import r1_d  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402


class Stop(Exception):
    pass


def _podmena(monkeypatch, seen: dict) -> None:
    def fabrika(sid):
        seen["period"], seen["kub"] = ps.PERIOD, ps.KUB
        raise Stop
    monkeypatch.setattr(ps, "fabrika_real", fabrika)
    monkeypatch.setattr(r1_d, "vybor", lambda ledger, sid: (pd.DataFrame(), {}))


def test_fabrika_poluchaet_test1_a_posle_vozvrat(monkeypatch):
    seen, old = {}, (ps.PERIOD, ps.KUB)
    _podmena(monkeypatch, seen)
    with pytest.raises(Stop):
        r1_d.schet("S1", "test1")
    assert seen["period"] == "test1"
    assert seen["kub"] == r1_d.KUB_DIR / "kub_x.yaml"
    assert (ps.PERIOD, ps.KUB) == old                          # finally вернул прежнее, даже при падении


def test_mutaciya_period_ne_vystavlen_lovitsya(monkeypatch):
    seen = {}
    _podmena(monkeypatch, seen)
    monkeypatch.setattr(r1_d, "modul_period", lambda ps_, period: contextlib.nullcontext())
    with pytest.raises(Stop):
        r1_d.schet("S1", "test1")
    assert seen["period"] != "test1"                           # без выставления фабрика видит train4 — основной тест на этом упал бы
