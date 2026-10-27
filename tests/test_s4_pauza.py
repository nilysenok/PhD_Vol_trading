"""Расследование Ж-038: пауза исполнения (exec_cooldown) 0 против ≥ 1 у S4 (отчёт docs/otchety/2026-10-27_rassl_s4_pauza.md).

Механизм: `fastbt/sweep_post.py::positions` после `exec_layer` (он уже возвращает {−1,0,+1} со знаком сырой позиции)
умножает результат на знак сырой позиции второй раз -> все шорты становятся лонгами при любой ненулевой
паузе / min_hold / consec_entry / min_strength. Ядро (`execlayer.exec_layer`) знак сохраняет — зелёные тесты.
Тест на дефект — xfail(strict=True): после правки оценщика он обязан стать зелёным, и xfail нужно снять.
Синтетика, без данных и сети.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import sweep_post
from fastbt.execlayer import exec_layer

N = 12
EX0 = dict(consec_entry=1, min_strength=0.0, q_sily=0.0, exec_cooldown=0, exec_min_hold=0, tf_min_hold=0, tf_reentry=0)


def _positions(monkeypatch, trades, cd):
    """positions() на синтетических сделках [(вход, выход, знак)]; ядро сделок подменено."""
    te = np.array([t[0] for t in trades], np.int64)
    tx = np.array([t[1] for t in trades], np.int64)
    td = np.array([t[2] for t in trades], np.int8)
    monkeypatch.setattr(sweep_post, "trades_one", lambda *a, **k: (te, tx, td))
    v = SimpleNamespace(rmP=[None], trend=None, strength=[np.ones(N)], n=N)
    return sweep_post.positions(v, {**EX0, "exec_cooldown": cd}, N)


def test_yadro_exec_layer_sohranyaet_znak_shorta():
    raw = np.array([0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], float)
    out = exec_layer(raw, np.ones(N), 1, 0, 1, 0.0, N)
    assert list(out[:5]) == [0, -1, -1, -1, 0]


def test_pauza_ne_menyaet_lonh(monkeypatch):
    a = _positions(monkeypatch, [(2, 6, 1)], 0)
    b = _positions(monkeypatch, [(2, 6, 1)], 1)
    assert list(a) == list(b) == [0, 0, 1, 1, 1, 1] + [0] * 6


@pytest.mark.xfail(strict=True, reason="Ж-038: sweep_post.positions умножает результат exec_layer на знак повторно: "
                                       "шорт при паузе ≥ 1 становится лонгом (sweep_post.py:162)")
def test_pauza_ne_menyaet_znak_shorta(monkeypatch):
    a = _positions(monkeypatch, [(2, 6, -1)], 0)
    b = _positions(monkeypatch, [(2, 6, -1)], 1)
    assert list(a) == [0, 0, -1, -1, -1, -1] + [0] * 6
    assert list(b) == list(a)


def test_stop_i_vhod_v_odnom_bare_pauza_ne_srabatyvaet(monkeypatch):
    """Выход (стоп) на баре 6 и вход (тот же знак) на баре 6: позиция непрерывна, пауза 0 и 1 дают одно и то же."""
    tr = [(2, 6, 1), (6, 9, 1)]
    a = _positions(monkeypatch, tr, 0)
    b = _positions(monkeypatch, tr, 1)
    assert list(a) == list(b) == [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_pauza_podavlyaet_vhod_cherez_bar_posle_vyhoda(monkeypatch):
    """Выход на баре 6, вход на баре 7: пауза 0 — оба; пауза 1 — вход на баре 7 пропущен (повторный вход после зазора)."""
    tr = [(2, 6, 1), (7, 10, 1)]
    a = _positions(monkeypatch, tr, 0)
    b = _positions(monkeypatch, tr, 1)
    assert list(a) == [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    assert list(b) == [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
