"""wfП-1 (Д-a): `daily_1850 --wf` пишет маркер .sha256; fastbt.data при импорте не затирает готовый ряд wf.
Синтетика, в сеть не ходит; корень — временный (FASTBT_MAIN), основная копия не пишется."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from fastbt import daily_1850 as D

REPO = Path(__file__).resolve().parent.parent


def wf_series() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["X", "X"], "date": pd.to_datetime(["2026-02-26", "2026-02-27"]),
                         "open": [1.0, 2.0], "high": [2.0, 3.0], "low": [1.0, 2.0], "close": [2.0, 3.0],
                         "volume": [5, 6]})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def import_data(root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "FASTBT_MAIN": str(root)}
    env.pop("FASTBT_DATA_ROOT", None)
    script = root / "imp.py"          # не `-m fastbt.data`: как __main__ модуль импортирует сам себя и ждёт собственный flock
    script.write_text(f"import sys\nsys.path.insert(0, {str(REPO)!r})\nimport fastbt.data\n")
    return subprocess.run([sys.executable, str(script)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120)


def test_g1_wf_pishet_marker_raven_sha(tmp_path):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(wf_series(), p)
    assert D.marker(p).read_text().strip() == sha(p)
    assert D.is_ready(p)


def test_g1_mutaciya_marker_ne_pishetsya_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_atomic_write", lambda path, data: None)      # мутация: маркер не пишется
    p = tmp_path / "ohlcv_daily_1850.parquet"
    D.write_wf(wf_series(), p)
    assert not D.is_ready(p)                                              # тест G1 на этом упал бы


def test_g2_import_ne_zatiraet_gotovyj_wf(tmp_path):
    p = tmp_path / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet"
    p.parent.mkdir(parents=True)
    D.write_wf(wf_series(), p)
    before = sha(p)
    r = import_data(tmp_path)
    assert r.returncode == 0, r.stderr[-500:]
    assert sha(p) == before and D.marker(p).read_text().strip() == before


def test_g4_mutaciya_ensure_ignoriruet_marker_lovitsya(tmp_path):
    p = tmp_path / "data" / "fastbt" / "daily" / "ohlcv_daily_1850.parquet"
    p.parent.mkdir(parents=True)
    D.write_wf(wf_series(), p)
    D.marker(p).unlink()                          # ряд wf без маркера: ensure обязан тронуть его (пересобрать/упасть)
    before = sha(p)
    r = import_data(tmp_path)
    assert r.returncode != 0 or sha(p) != before  # значит, G2 держится именно на маркере
