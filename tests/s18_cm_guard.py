"""Общая обвязка тестов S-C-IMPL (не тест): autouse-охрана боевых файлов и синтетика компонентов. Только monkeypatch (№ 43)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "docs/otchety/harj_x1/ALL_TRIALS.csv", ROOT / "docs/otchety/harj_x1/CHECKPOINT.json"]
CM_DIR = ROOT / "data/regen/harj_struct/CM"


def snapshot() -> dict:
    """sha256 содержимого ALL_TRIALS.csv, CHECKPOINT.json и готовых parquet в CM (без `_part` и `.tmp`)."""
    out = {}
    for p in GUARDED + (sorted(CM_DIR.glob("*.parquet")) if CM_DIR.exists() else []):
        out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return out


@pytest.fixture(autouse=True)
def guard_real_files():
    """Тесты не меняют реальные ALL_TRIALS.csv, CHECKPOINT.json и готовые parquet в data/regen/harj_struct/CM (сравнение содержимого)."""
    before = snapshot()
    yield
    after = snapshot()
    changed = [k for k, v in before.items() if after.get(k) != v and not (k.endswith(".parquet") and after.get(k) is None)]
    assert not changed, "тест изменил боевой файл: %s" % changed


def synth_components(T=400, seed=0, with_u=True):
    """Синтетика сессий: c, p > 0, u >= 0 (u = 0 при with_u=False), RV = c + p + u (тождество точно, строки без NaN)."""
    g = np.random.default_rng(seed)
    c = np.exp(-9.0 + 0.5 * g.normal(size=T))
    p = np.exp(-9.2 + 0.5 * g.normal(size=T))
    u = np.exp(-11.0 + 0.3 * g.normal(size=T)) if with_u else np.zeros(T)
    return c, p, u, c + p + u
