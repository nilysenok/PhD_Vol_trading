"""Ж-047, помощники тестов 2.1–2.18 (контракт docs/otchety/2026-10-27_zh047_kontrakt.md §2). Тестов нет.

Допущения об интерфейсе К2 (ветка zh047-k2 ещё не слита; имена из контракта §1.1, сверить при слиянии):
Params(b=k_s, R=, h=, ktr=, bu=, tick=, side=±1, rezhim="zakrytie"|"bary"); Trade.reason ∈
stop|take|trail|bu|vremya|signal. Пока поля нет — TypeError → xfail(strict) «ждёт К2»; иная ошибка = красный.
Числа двоично-точные: σ = 1/16, b = 1, R = 2, h = 1: S = 93.75, T = 112.5, тик 0.25.
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from fastbt import exec_e

SG, TICK = 0.0625, 0.25
S_LONG, T_LONG, L_LIM = 93.75, 112.5, 93.75
WAIT = (TypeError, ImportError, AttributeError, KeyError)


def _absent(k):
    """Кода К<k> ещё нет в дереве: К1 — sigma_real; К2 — поля Params; К3 — rezhim_e1; К4 — etap1."""
    if k == 2:
        return "tick" not in exec_e.Params.__dataclass_fields__
    mod = {1: "sigma_real", 3: "rezhim_e1", 4: "etap1"}[k]
    return importlib.util.find_spec(f"fastbt.{mod}") is None


def wait(k):
    """xfail(strict) ровно пока кода К<k> нет: тест обязан падать (иная ошибка = красный);
    после слияния ветки условие ложно, тест идёт как обычный — снимать маркеры вручную не нужно."""
    return pytest.mark.xfail(_absent(k), strict=True, raises=WAIT, reason=f"ждёт К{k}")


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return exec_e.Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False)


def F(x, n=1):
    return [(x, x, x, x)] * n


def P(**kw):
    base = dict(b=1.0, R=2.0, h=1, tick=TICK, comm=0.0004)
    base.update(kw)
    return exec_e.Params(**base)


def go(day1, sig=(1, 1, 0), p=None, sigma=SG):
    """День 0: вход по close 100 (k=None). День 1 — исследуемые бары, день 2 — плоский (сигнал 0 → выход)."""
    days = [D(F(100.0)), D(day1), D(F(100.0, 2))]
    return exec_e.run("T", days, list(sig), np.full(3, sigma), p or P())


def first(r):
    return r.trades[0]


def scan_ref(self, i, j, d, b0, mut=""):
    """Эталонный проход по барам (замена `_Run._scan` в мутациях): худший сценарий Ж-047 п. 5.
    mut: take_touch | stop_strict | gap_at_level | take_first | trail_same_bar | be_same_bar."""
    p, tr = self.p, self.tr
    px, sg, tick = tr.entry_px, self.sg, p.tick
    st = getattr(self, "_zs", None)
    if st is None or st["tr"] is not tr:
        st = self._zs = dict(tr=tr, M=px, be=False)
    S = px * (1 - p.b * sg) if p.b is not None else None
    T = None
    if getattr(p, "R", None) is not None and p.b is not None:
        T = math.ceil(px * (1 + p.R * p.b * sg * math.sqrt(p.h)) / tick - 1e-9) * tick
    for b in range(b0, len(d.c)):
        o, h, l = d.o[b], d.h[b], d.l[b]
        if mut == "trail_same_bar":
            st["M"] = max(st["M"], h)
        if mut == "be_same_bar" and getattr(p, "bu", False) and h >= px * (1 + sg):
            st["be"] = True
        lvl = S
        if getattr(p, "ktr", None) is not None:
            lvl = max(lvl if lvl is not None else 0.0, st["M"] * (1 - p.ktr * sg))
        if st["be"]:
            lvl = max(lvl if lvl is not None else 0.0, px)
        hit_s = lvl is not None and (l < lvl if mut == "stop_strict" else l <= lvl)
        hit_t = T is not None and (h >= T if mut == "take_touch" else h >= T + tick)
        sx = None if not hit_s else (lvl if mut == "gap_at_level" else min(o, lvl))
        if mut == "take_first" and hit_t:
            return self._exit(i, j, b, o if o > T else T, "take")
        if hit_s:
            return self._exit(i, j, b, sx, "stop")
        if hit_t:
            return self._exit(i, j, b, o if o > T else T, "take")
        if mut != "trail_same_bar":
            st["M"] = max(st["M"], h)
        if getattr(p, "bu", False) and h >= px * (1 + sg):
            st["be"] = True


def mutate(monkeypatch, mut):
    monkeypatch.setattr(exec_e._Run, "_scan", lambda s, i, j, d, b0: scan_ref(s, i, j, d, b0, mut))
