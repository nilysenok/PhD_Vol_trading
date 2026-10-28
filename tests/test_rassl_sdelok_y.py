"""Расслоение sdelok_y: поле ledger (cnt/2, ступени лестницы усреднены) против счёта закрытых сделок разбора.
Причина расхождения ×2 у лестничных строк: sum_rows склеивает trades обеих ступеней, cnt — усредняет. Синтетика, без сети.
Мутация: счёт без деления на число ступеней (как в разборе до правки) обязан не совпасть с полем ledger."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import prohod1_rezhimy as R  # noqa: E402
from fastbt import exec_e  # noqa: E402
from fastbt.etap1_run import sum_rows  # noqa: E402
from fastbt.exec_e_types import Day, Params  # noqa: E402


def _ryad(seed: int, n: int = 60):
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    sig = rng.choice([-1.0, 0.0, 1.0], n, p=[0.25, 0.35, 0.4])
    sig[-1] = 0.0
    return [Day(np.array([c]), np.array([c]), np.array([c]), np.array([c]), False) for c in px], sig


def _run(days, sig):
    return exec_e.run_e1("T", days, sig, np.ones(len(days)), Params(check="bary"), w=np.full(len(days), 0.5))


def test_lestnica_ledger_raven_zakrytym_s_vesom_stupeni():
    for seed in range(20):
        days, sig = _ryad(seed)
        rs = [_run(days, sig), _run(days, sig)]
        lad = sum_rows(rs)
        pole = float(lad.cnt.sum()) / 2                                       # как etap1_eval.evaluate: sdelok_y = cnt/2
        _, nd, no = R.zakrytye(lad.trades, 2)
        assert no == 0 and nd == pole, f"seed {seed}: разбор {nd} ≠ поле ledger {pole}"
        _, nd1, _ = R.zakrytye(lad.trades, 1)                                 # мутация (старый счёт): вдвое больше
        assert nd1 == 2 * pole or pole == 0


def test_bez_lestnicy_ledger_raven_zakrytym():
    for seed in range(20):
        days, sig = _ryad(seed)
        r = _run(days, sig)
        _, nd, no = R.zakrytye(r.trades, 1)
        assert nd == float(r.cnt.sum()) / 2 and no == 0


def test_otkrytye_ne_schitayutsya_i_vesyat_po_stupeni():
    class T:
        def __init__(self, reason):
            self.reason = reason
    dol, nd, no = R.zakrytye([T("signal"), T("take"), T(None), T("signal"), T("take"), T(None)], 2)
    assert (nd, no) == (2.0, 1.0) and dol["signal"] == 0.5 and dol["take"] == 0.5
