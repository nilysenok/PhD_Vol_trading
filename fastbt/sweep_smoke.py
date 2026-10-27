"""Дымовой прогон Э1-а: OAT по одному параметру S1 на реальном train 2018 (настройка). Печатает только OK и время.

Запуск: FASTBT_MAIN=<своя копия> FASTBT_CACHE=<папка> /usr/bin/python3 -m fastbt.sweep_smoke
Validation не открывается: период setup, режим oat.
"""
from __future__ import annotations

import time

from fastbt.sweep import Spec, Sweep, make_evaluator

SPEC = dict(
    obj=dict(kind="Sk", sid="S1"),
    base=dict(ma_window=20, z_entry=2.0, z_exit=0.5, sl_mult=1.5, tp_mult=3.0, max_hold=20,
              breakeven_trigger=None, cooldown_bars=0, partial_exit=False),
    grids=dict(ma_window=[10, 15, 20, 30, 45]),
    period="setup", mode="oat", comm_pct=[0.035, 0.04])


def main():
    t0 = time.time()
    spec = Spec.from_yaml(SPEC)
    res = Sweep(spec, make_evaluator(spec)).run()
    t = res["table"]
    assert len(t) == 2 * len(SPEC["grids"]["ma_window"]) and t["sharpe"].notna().all()
    assert t["sharpe"].nunique() > 1, "параметр не меняет результат"
    assert (t[t.comm_pct == 0.04]["sharpe"].values <= t[t.comm_pct == 0.035]["sharpe"].values + 1e-12).all()
    print(f"OK: {len(res['table'])} строк, {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
