"""S-Z-IMPL-0: `write` отказывает до регистрации S-Z и не перезаписывает отличающийся файл; работает и до, и после регистрации реестра (подмена `registered` и `XR.TRIALS`). Боевой `Z/` и реестр не трогаются."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_reg as XR
from models_v2.s18 import z_feat as F

DM = pd.bdate_range("2024-01-01", periods=5)


def _maps(v=0.0):
    return {y: {s: np.full((5, 4), v + y + s) for s in F.SEEDS} for y in F.YEARS}


def _trials(tmp_path, monkeypatch, ids):
    p = tmp_path / "ALL_TRIALS.csv"
    pd.DataFrame([[i, "n", "f", "p", "registered", "2026-11-12"] for i in ids], columns=XR.COLS).to_csv(p, index=False)
    monkeypatch.setattr(XR, "TRIALS", p)


def _prep(monkeypatch, reg: bool):
    monkeypatch.setattr(F, "registered", lambda: reg)
    monkeypatch.setattr(F, "dm_dates", lambda: DM)
    monkeypatch.setattr(F, "zgrids", lambda *a, **k: None)


def test_registered_reads_trials_by_id(tmp_path, monkeypatch):
    _trials(tmp_path, monkeypatch, ["S-RZ", "S-OTHER"])
    assert F.registered() is False
    _trials(tmp_path, monkeypatch, ["S-RZ", "S-RCKZ"])
    assert F.registered() is True
    monkeypatch.setattr(XR, "TRIALS", tmp_path / "net_takogo.csv")
    assert F.registered() is False


def test_write_refuses_before_registration(tmp_path, monkeypatch):
    _prep(monkeypatch, False)
    p = tmp_path / "Z" / "zmaps.npz"
    with pytest.raises(SystemExit, match="S-RCKZ"):
        F.write(p, _maps(), _maps())
    assert not p.exists() and not p.parent.exists()


def test_write_refuses_with_real_registered_when_trials_lack_id(tmp_path, monkeypatch):
    """Путь без подмены `registered`: реестр без S-RCKZ → отказ."""
    _trials(tmp_path, monkeypatch, ["S-RZ"])
    monkeypatch.setattr(F, "dm_dates", lambda: DM)
    with pytest.raises(SystemExit):
        F.write(tmp_path / "zmaps.npz", _maps(), _maps())
    assert not (tmp_path / "zmaps.npz").exists()


def test_write_after_registration_is_atomic_and_idempotent(tmp_path, monkeypatch):
    _prep(monkeypatch, True)
    p = tmp_path / "Z" / "zmaps.npz"
    assert F.write(p, _maps(), _maps(1.0)) == p and p.exists() and not (p.parent / "tmp_zmaps.npz").exists()
    zm, zs, d = F.unpack(p)
    assert np.array_equal(zm[2025][3], _maps()[2025][3]) and np.array_equal(zs[2025][3], _maps(1.0)[2025][3]) and (d == DM).all()
    before = p.read_bytes()
    assert F.write(p, _maps(), _maps(1.0)) == p and p.read_bytes() == before         # тот же состав — без перезаписи
    with pytest.raises(SystemExit, match="перезапись отказана"):
        F.write(p, _maps(5.0), _maps(1.0))
    assert p.read_bytes() == before
