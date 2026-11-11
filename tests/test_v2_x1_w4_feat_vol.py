"""W4-B1, перечень аномалий объёма (B.G3) и «--check только читает» (B.G2/B.G3): известный ответ на синтетике (скачок ×10 вверх и ×0,1 вниз), реальный перечень = файлу;
`check_all` не меняет хэши выходов, мутация «check пишет» ловится. Подмены — monkeypatch; реальные файлы только читаются."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w4_feat as F

ROOT = Path(F.__file__).resolve().parents[2]
REAL = [F.FEAT / "S_ext.parquet", F.FEAT / "SIGMA_ext.parquet", F.VOLCSV, ROOT / "docs/otchety/harj_x1/ALL_TRIALS.csv"]


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


class FakeSrc:
    """Синтетический источник: 60 рабочих дней 2019, 52 бара 10:00…18:40, объём 1000; день 30 — ×10, день 45 — ×0,1."""

    def __init__(self, mult: dict | None = None, n_days: int = 60):
        rng = np.random.default_rng(7)
        days = pd.bdate_range("2019-01-09", periods=n_days)
        rows = []
        for i, day in enumerate(days):
            px = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, 53)))
            for k in range(52):
                rows.append((day + pd.Timedelta(hours=10, minutes=10 * k), px[k], px[k + 1], 1000.0 * (mult or {}).get(i, 1.0)))
        self.b = pd.DataFrame(rows, columns=["begin", "open", "close", "volume"])
        self.days = days

    def bars(self, inst, cut):
        return self.b

    def index_days(self, cut):
        return set(self.days)


def test_synthetic_volume_jump_x10_up_and_x01_down_known_answer() -> None:
    src = FakeSrc({30: 10.0, 45: 0.1})
    tab = F.volume_table(src, cut="2019-12-31")
    days = FakeSrc().days
    assert list(tab["d"]) == [days[30], days[45]]
    assert list(tab["direction"]) == ["up", "down"] and list(tab["period"]) == ["archive", "archive"]
    np.testing.assert_allclose(tab["ratio"].to_numpy(), [10.0, 0.1], rtol=1e-12)
    np.testing.assert_allclose(tab["prev20_median"].to_numpy(), [1000.0, 1000.0])
    np.testing.assert_allclose(tab["bar_vol_median"].to_numpy(), [10000.0, 100.0])


def test_synthetic_no_anomaly_gives_empty_list_and_boundary_is_ge_10() -> None:
    assert len(F.volume_table(FakeSrc(), cut="2019-12-31")) == 0
    assert len(F.volume_table(FakeSrc({30: 9.99, 45: 0.101}), cut="2019-12-31")) == 0      # ниже порога ×10 / ×0,1 — не аномалия
    assert len(F.volume_table(FakeSrc({30: 10.0}), cut="2019-12-31")) == 1                 # ровно ×10 — аномалия (>=)


def test_synthetic_first_20_sessions_are_not_judged() -> None:
    """В первые 20 сессий окна нет — скачок на дне 10 не перечисляется (не выдумываем медиану)."""
    assert len(F.volume_table(FakeSrc({10: 10.0}), cut="2019-12-31")) == 0


def test_synthetic_volume_uses_only_past_sessions_not_the_future() -> None:
    """Медиана — по 20 ПРОШЛЫМ сессиям: скачок дня 30 не размывает базу дня 31 (день 31 не отмечен), а день после падения не помечается заново."""
    tab = F.volume_table(FakeSrc({30: 10.0}), cut="2019-12-31")
    days = FakeSrc().days
    assert list(tab["d"]) == [days[30]] and days[31] not in set(tab["d"])


def test_real_volume_list_equals_file_and_read_only_check() -> None:
    before = real_state()
    res = F.check_volume(False)
    assert res["ok"] and res["file_matches"] and res["n_days"] == len(pd.read_csv(F.VOLCSV))
    assert real_state() == before                                                         # check_volume(False) ничего не пишет
    assert set(pd.read_csv(F.VOLCSV)["direction"]) <= {"up", "down"}


def _stub_gates(monkeypatch, tmp: Path, mutant_writer=None) -> list:
    outs = [tmp / "S_ext.parquet", tmp / "SIGMA_ext.parquet", tmp / "W4_VOLUME.csv"]
    for p in outs:
        p.write_bytes(b"x" + p.name.encode())
    monkeypatch.setattr(F, "outputs", lambda: outs)
    monkeypatch.setattr(F, "check_archive", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(F, "check_pit", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(F, "check_volume", lambda write, *a, **k: (mutant_writer(outs) if mutant_writer else None, {"ok": True})[1])
    return outs


def test_check_all_keeps_hashes_and_mutation_check_that_writes_is_caught(tmp_path, monkeypatch) -> None:
    _stub_gates(monkeypatch, tmp_path)
    res = F.check_all()
    assert res["hash_unchanged"] and res["ok"]
    t2 = tmp_path / "w"
    t2.mkdir()
    _stub_gates(monkeypatch, t2, mutant_writer=lambda outs: outs[2].write_bytes(b"changed by check"))   # мутация: --check пишет выход
    bad = F.check_all()
    assert not bad["hash_unchanged"] and not bad["ok"]


def test_check_all_fails_when_an_output_is_missing(tmp_path, monkeypatch) -> None:
    outs = _stub_gates(monkeypatch, tmp_path)
    outs[0].unlink()
    assert not F.check_all()["ok"]                                                        # нет файла выхода — не «ок»
