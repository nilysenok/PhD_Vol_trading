"""Р-7: гонка двух процессов на одном ключе дискового кэша (bars5m и cache.save). Синтетика, без сети и счёта.

Причина (воспроизведена): tmp-имя зависело только от ключа, поэтому второй писатель того же ключа затирал/забирал
tmp первого, и `os.replace` первого падал FileNotFoundError (оценщик ловит это как status=oshibka -> «быстрый ≠ эталон»).
Чередование «A пишет tmp -> B пишет и подменяет целиком -> A подменяет» задаётся детерминированно.
Мутация: вернуть общее имя tmp — оба теста падают.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastbt import bars5m, cache  # noqa: E402


def _csv(src: Path) -> None:
    src.mkdir()
    rows = [("2019-03-12 10:00", 10, 12, 9, 11, 5, 50), ("2019-03-12 10:02", 11, 15, 8, 14, 7, 70),
            ("2019-03-12 10:10", 20, 21, 19, 20.5, 1, 20)]
    pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume", "value"]).to_csv(src / "AAA_2019.csv", index=False)


def _interleave(monkeypatch, cls, name: str, other) -> None:
    """После записи tmp первым писателем (один раз) целиком отрабатывает второй писатель того же ключа."""
    orig, state = getattr(cls, name), {"in": False}

    def wrapped(*a, **kw):
        r = orig(*a, **kw)
        if not state["in"]:
            state["in"] = True
            other()
        return r
    monkeypatch.setattr(cls, name, wrapped)


def test_bars5m_dva_pisatelya_odnogo_klyucha(tmp_path, monkeypatch):
    src, cch = tmp_path / "src", tmp_path / "c"
    _csv(src)
    monkeypatch.delenv("FASTBT_NO_CACHE", raising=False)
    _interleave(monkeypatch, pd.DataFrame, "to_parquet", lambda: bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cch))
    a = bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cch)
    monkeypatch.undo()
    b = bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cch)      # чтение готового файла
    pd.testing.assert_frame_equal(a, b)
    assert [p.name for p in cch.iterdir() if p.suffix != ".parquet"] == []              # мусорных tmp не осталось


def test_cache_save_dva_pisatelya_odnogo_klyucha(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE", tmp_path)
    monkeypatch.delenv("FASTBT_NO_CACHE", raising=False)
    arr = {"x": np.arange(5.0)}
    _interleave(monkeypatch, np, "savez_compressed", lambda: cache.save("k", arr))
    cache.save("k", arr)
    monkeypatch.undo()
    z = np.load(tmp_path / "k.npz")
    assert z["x"].tolist() == arr["x"].tolist()
    assert [p.name for p in tmp_path.iterdir()] == ["k.npz"]
