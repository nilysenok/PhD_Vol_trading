"""Ж-047, помощники тестов 2.1–2.18 (контракт docs/otchety/2026-10-27_zh047_kontrakt.md §2). Тестов нет.

Интерфейс К2 (PR #349): Params(b=k_s, R=, h=, k_tr=, be=, tick=, side=±1, check="bary"|"zakrytie", stop_touch=,
reentry=); exec_e.run(..., close=); состояние — exec_e_run.Run; правила — exec_e_rules; Result.fill_rate.
Числа двоично-точные: σ = 1/16, b = 1, R = 2, h = 1: S = 93.75, T = 112.5, тик 0.25.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run

SG, TICK = 0.0625, 0.25
S_LONG, T_LONG, L_LIM = 93.75, 112.5, 93.75
WAIT = (TypeError, ImportError, AttributeError, KeyError)


def _absent(k):
    """Кода К<k> ещё нет в дереве: К1 — sigma_real; К3 — rezhim_e1; К4 — etap1 / port_rows / run_e1."""
    if k == 2:
        return "stop_touch" not in exec_e.Params.__dataclass_fields__
    mod = {1: "sigma_real", 3: "rezhim_e1", 4: "etap1"}[k]
    return importlib.util.find_spec(f"fastbt.{mod}") is None


def wait(k):
    """xfail(strict) ровно пока кода К<k> нет: иная ошибка = красный; после слияния тест идёт обычным."""
    return pytest.mark.xfail(_absent(k), strict=True, raises=WAIT, reason=f"ждёт К{k}")


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return exec_e.Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False)


def F(x, n=1):
    return [(x, x, x, x)] * n


def P(**kw):
    base = dict(b=1.0, R=2.0, h=1, tick=TICK, comm=0.0004, stop_touch=True)
    base.update(kw)
    return exec_e.Params(**base)


def go(day1, sig=(1, 1, 0), p=None, sigma=SG):
    """День 0: вход по close 100 (k=None). День 1 — исследуемые бары, день 2 — плоский (сигнал 0 → выход)."""
    days = [D(F(100.0)), D(day1), D(F(100.0, 2))]
    return exec_e.run("T", days, list(sig), np.full(3, sigma), p or P())


def first(r):
    return r.trades[0]


def patch_rule(monkeypatch, name, wrap):
    """Мутация правила: функция `exec_e_run.<name>` (из exec_e_rules) заменяется обёрткой wrap(orig)."""
    monkeypatch.setattr(exec_e_run, name, wrap(getattr(exec_e_run, name)))


def scan_mut(flag):
    """Копия Run._scan с мутацией: take_first — тейк проверен раньше стопа; same_bar — экстремум трейла и
    безубыток обновляются по high/low того же бара (до проверки уровней)."""
    def scan(s, i, j, d, b0):
        rl = exec_e_run
        sd, dyn = s.tr.side, s.p.k_tr is not None or s.p.be
        S, T = s._levels(j)
        for b in range(b0, len(d.c)):
            o, h, l = d.o[b], d.h[b], d.l[b]
            if flag == "same_bar" and dyn:
                s._after_bar(j, h, l)
                S, T = s._levels(j)
            chk = [("S", S), ("T", T)] if flag != "take_first" else [("T", T), ("S", S)]
            for nm, lv in chk:
                if lv is None:
                    continue
                hit, px = (rl.stop_hit(sd, o, h, l, lv, s.p.stop_touch) if nm == "S"
                           else rl.take_hit(sd, o, h, l, lv, s.p.tick))
                if hit:
                    return s._exit(i, j, b, px, s.sr if nm == "S" else "take")
            if dyn and flag != "same_bar":
                s._after_bar(j, h, l)
                S, T = s._levels(j)
    return scan


def mutate(monkeypatch, flag):
    monkeypatch.setattr(exec_e_run.Run, "_scan", scan_mut(flag))
