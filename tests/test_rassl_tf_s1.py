"""Расследование П5b (#596): прибыль S1 на 1h создаёт Chandelier выше цены входа в баре исполнения лимита.

Причина (docs/otchety/2026-10-29_rassl_tf_s1.md): Run._try_order (fastbt/exec_e_run.py) после исполнения лимита
проверяет стоп в том же баре и выходит ровно по уровню S (`s._exit(i, j, b, S, ...)`). Chandelier (Л5) берёт экстремум
баров ДО входа, поэтому у лонга S может быть выше цены входа и выше high бара: выход по цене, которой в баре не было.
Тесты-воспроизведения помечены xfail(strict): после починки они проходят и xfail нужно снять.
Синтетика, сеть и data/ не читаются.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fastbt import tf_ispolnenie_v5 as X

EPS = 1e-9
# бары 0..5 (каждый — «день» автомата): рост до 110, спад до 100, сигнал лонг на close бара 3, лимит исполняется в баре 4
O = np.array([110.0, 108.0, 104.0, 101.0, 100.0, 100.0])
H = np.array([110.0, 108.0, 104.0, 101.0, 100.5, 100.0])
L = np.array([108.0, 104.0, 101.0, 100.0, 99.5, 100.0])
C = np.array([108.0, 104.0, 101.0, 100.0, 100.0, 100.0])
SOST = np.array([0, 0, 0, 1, 1, 1], np.int8)
SIGMA = np.full(6, 0.01)
ATR = np.full(6, 1.0)


def _vne_bara(trades, h, l):
    """Сделки, вышедшие по цене вне [low; high] бара выхода."""
    return [t for t in trades if not (l[t.exit_day] - EPS <= t.exit_px <= h[t.exit_day] + EPS)]


def test_kontrol_bez_chandelier_vyhod_v_predelah_bara():
    """Контроль: без Chandelier (стоп 2σ ниже входа) выход внутри бара — путь автомата корректен."""
    p = dataclasses.replace(X.PARAMS, chand=None)
    x = X.sessiya("SYN", O, H, L, C, SOST, SIGMA, p, ATR)
    assert len(x.trades) == 1 and not _vne_bara(x.trades, H, L)


@pytest.mark.xfail(strict=True, reason="БЛОКЕР #596: стоп Chandelier в баре входа исполняется по S выше high бара")
def test_chandelier_v_bare_vhoda_ne_vyhodit_vyshe_high():
    """Лимит 99,75 исполнен в баре 4 (low 99,5); Chandelier = 110 − 3·1 = 107 > high 100,5. Цена выхода обязана быть в баре."""
    x = X.sessiya("SYN", O, H, L, C, SOST, SIGMA, X.PARAMS, ATR)
    assert not _vne_bara(x.trades, H, L), [(t.entry_px, t.exit_px, t.reason) for t in x.trades]


@pytest.mark.xfail(strict=True, reason="БЛОКЕР #596: стоп выше входа в баре входа даёт лонгу прибыль из ничего")
def test_stop_v_bare_vhoda_ne_daet_pribyli():
    """Стоп (любого вида), сработавший в баре входа, не может закрыть лонг дороже цены входа: брутто ≤ 0."""
    x = X.sessiya("SYN", O, H, L, C, SOST, SIGMA, X.PARAMS, ATR)
    assert x.gross <= EPS, x.gross


def _rw_sessii(seed: int, n_ses: int = 300, n_bar: int = 9, sub: int = 6):
    """Случайное блуждание: бар ТФ из sub шагов, σ шага 0,2 %; сторона сигнала случайная (−1/0/1), держится 3 бара."""
    rng = np.random.default_rng(seed)
    for _ in range(n_ses):
        x = np.log(100.0) + np.cumsum(rng.normal(0, 0.002, n_bar * sub + 1))
        o = np.exp(x[:-1:sub][:n_bar])
        seg = [x[k * sub:(k + 1) * sub + 1] for k in range(n_bar)]
        h = np.exp([s.max() for s in seg])
        lo = np.exp([s.min() for s in seg])
        c = np.exp([s[-1] for s in seg])
        st = np.repeat(rng.integers(-1, 2, n_bar // 3 + 1), 3)[:n_bar].astype(np.int8)
        st[0] = 0
        yield o, h, lo, c, st


@pytest.mark.xfail(strict=True, reason="БЛОКЕР #596: на случайном блуждании есть выходы вне [low; high] бара")
def test_sluchaynoe_bluzhdanie_vyhody_v_predelah_bara():
    """Случайное блуждание (знания нет): ни одна сделка не выходит по цене, которой не было в баре выхода."""
    plohie, vse = 0, 0
    for o, h, lo, c, st in _rw_sessii(7):
        sg = np.full(len(c), 0.002 * np.sqrt(6))
        atr = np.full(len(c), float(np.mean(h - lo)))
        x = X.sessiya("RW", o, h, lo, c, st, sg, X.PARAMS, atr)
        plohie += len(_vne_bara(x.trades, h, lo))
        vse += len(x.trades)
    assert vse > 100 and plohie == 0, (plohie, vse)


def test_sluchaynoe_bluzhdanie_bez_chandelier_brutto_okolo_nulya():
    """Контроль: без Chandelier на блуждании выходов вне бара нет, прибыли из ничего нет (брутто на сделку ≤ +0,05 %).
    Факт на seed 7: −0,11 % на сделку (304 сделки) — сдвиг в худшую сторону, порядок в баре консервативный."""
    p = dataclasses.replace(X.PARAMS, chand=None)
    plohie, g, n = 0, 0.0, 0
    for o, h, lo, c, st in _rw_sessii(7):
        sg = np.full(len(c), 0.002 * np.sqrt(6))
        x = X.sessiya("RW", o, h, lo, c, st, sg, p, None)
        plohie += len(_vne_bara(x.trades, h, lo))
        g, n = g + x.gross, n + len(x.trades)
    assert n > 100 and plohie == 0 and g / n < 5e-4, (plohie, g / n)
