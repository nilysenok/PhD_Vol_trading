"""Дымовой прогон Э1-а: OAT по одному параметру S1 на реальном 2018 (настройка) через sweep_drivers. Печатает только OK и время.

Запуск: FASTBT_MAIN=<своя копия> FASTBT_CACHE=<папка> /usr/bin/python3 -m fastbt.sweep_smoke
Validation не открывается: период setup.
"""
from __future__ import annotations

import time

from fastbt import sweep_drivers as sd
from fastbt.sweep import Sweep, make_evaluator
from fastbt.sweep_spec import Spec

SPEC = dict(
    obj=dict(kind="Sk", sid="S1"),
    base=dict(ma_window=20, z_entry=2.0, z_exit=0.5, sl_mult=1.5, tp_mult=3.0, max_hold=20,
              breakeven_trigger=None, cooldown_bars=0, partial_exit=False),
    comm_pct=[0.035, 0.04])
GRID = dict(ma_window=[10, 15, 20, 30, 45])


def main():
    t0 = time.time()
    spec = Spec.from_yaml(SPEC)
    sw = Sweep(spec, make_evaluator(spec))
    res = sd.oat_sweep(lambda p, per: sw.evaluate(p, per), spec.base, GRID, "setup")
    sh = [sw.evaluate({**spec.base, "ma_window": w}, "setup")["sharpe"] for w in GRID["ma_window"]]
    hi = [sw.evaluate({**spec.base, "ma_window": w}, "setup", 0.04)["sharpe"] for w in GRID["ma_window"]]
    assert res and all(s == s for s in sh) and len(set(sh)) > 1, "параметр не меняет результат"
    assert all(b <= a + 1e-12 for a, b in zip(sh, hi)), "комиссия 0,04 должна давать Шарп не выше 0,035"
    row = sw.evaluate(spec.base, "setup")
    assert all(k in row for k in ("sharpe", "by_year", "msr", "sortino", "mdd", "T", "skew", "kurt", "N"))
    print(f"OK: {len(sh)} точек, {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
