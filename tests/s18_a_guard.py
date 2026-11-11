"""Общая обвязка тестов S-A (не тест): autouse-охрана боевых файлов и синтетика. Только monkeypatch (№ 43)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "docs/otchety/harj_x1/ALL_TRIALS.csv", ROOT / "docs/otchety/harj_x1/CHECKPOINT.json"]
A_DIR = ROOT / "data/regen/harj_struct/A"


def snapshot() -> dict:
    """sha256 содержимого ALL_TRIALS.csv, CHECKPOINT.json и готовых parquet в A (без `_part` и `.tmp`: их пишет боевой прогон кодера параллельно)."""
    out = {}
    for p in GUARDED + (sorted(A_DIR.glob("*.parquet")) if A_DIR.exists() else []):
        out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


@pytest.fixture(autouse=True)
def guard_real_files():
    """Тесты не меняют реальные ALL_TRIALS.csv, CHECKPOINT.json и готовые parquet в data/regen/harj_struct/A (сравнение содержимого; новые файлы — не ловятся)."""
    before = snapshot()
    yield
    after = snapshot()
    changed = [k for k, v in before.items() if after.get(k) != v and not (k.endswith(".parquet") and after.get(k) is None)]
    assert not changed, "тест изменил боевой файл: %s" % changed


def synth_r2(T=120, seed=0, miss=0.05, nan_days=()):
    """r² баров (T, 53) в долях, U-профиль, часть баров пропущена (NaN); минимум 30 баров в дне."""
    g = np.random.default_rng(seed)
    u = 1.0 + 1.5 * (np.arange(53) / 52 - 0.5) ** 2
    r2 = (g.normal(size=(T, 53)) ** 2) * u * 1e-6 * np.exp(0.3 * g.normal(size=(T, 1)))
    r2[g.random((T, 53)) < miss] = np.nan
    return r2


def two_seg_dates(n1=60, n2=70):
    """Даты: n1 дней до остановки 2022 и n2 после (для `daily.add_windows`)."""
    a = pd.bdate_range(end="2022-02-25", periods=n1)
    b = pd.bdate_range(start="2022-03-25", periods=n2)
    return a.append(b)
