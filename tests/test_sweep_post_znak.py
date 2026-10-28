"""Ж-038: знак позиции после слоя исполнения в `sweep_post.positions` совпадает со знаком сырой позиции.
Параметры слоя по одному (≠ 0): exec_cooldown, exec_min_hold, consec_entry, min_strength. Мутация «двойной знак» ловится.
Синтетика, без данных и сети."""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import sweep_post

N = 14
EX0 = dict(consec_entry=1, min_strength=0.0, q_sily=0.0, exec_cooldown=0, exec_min_hold=0, tf_min_hold=0, tf_reentry=0)
PARAMS = [("exec_cooldown", 1), ("exec_min_hold", 2), ("consec_entry", 2), ("min_strength", 0.5)]
# шорт на барах 2..6, лонг на барах 8..11
TRADES = [(2, 7, -1), (8, 12, 1)]


def _run(monkeypatch, ex):
    te = np.array([t[0] for t in TRADES], np.int64)
    tx = np.array([t[1] for t in TRADES], np.int64)
    td = np.array([t[2] for t in TRADES], np.int8)
    monkeypatch.setattr(sweep_post, "trades_one", lambda *a, **k: (te, tx, td))
    v = SimpleNamespace(rmP=[None], trend=None, strength=[np.ones(N)], n=N)
    raw = sweep_post.positions(v, {**EX0}, N)
    return raw, sweep_post.positions(v, ex, N)


def _check(raw, out):
    assert (raw < 0).any() and (raw > 0).any()          # синтетика содержит и шорт, и лонг
    assert not (out > 0)[raw < 0].any()                  # шорт не становится лонгом
    assert not (out < 0)[raw > 0].any()                  # лонг не становится шортом
    assert (out < 0).any()                               # шорт не потерян целиком


@pytest.mark.parametrize("name,val", PARAMS)
def test_znak_posle_sloya_ispolneniya(monkeypatch, name, val):
    raw, out = _run(monkeypatch, {**EX0, name: val})
    _check(raw, out)


@pytest.mark.parametrize("name,val", PARAMS)
def test_mutaciya_dvoynoy_znak_lovitsya(monkeypatch, name, val):
    """Подложенная ошибка: результат exec_layer снова умножен на знак сырой позиции -> тест обязан упасть."""
    orig = sweep_post.exec_layer
    monkeypatch.setattr(sweep_post, "exec_layer", lambda pos, *a, **k: orig(pos, *a, **k) * np.where(pos < 0, -1.0, 1.0))
    raw, out = _run(monkeypatch, {**EX0, name: val})
    with pytest.raises(AssertionError):
        _check(raw, out)
