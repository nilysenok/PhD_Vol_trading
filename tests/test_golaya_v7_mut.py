"""Тестировщик #860: мутации, не пойманные тестами автора (p = k/n; терциль с днём d)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis")]
import golaya_v7 as gv  # noqa: E402
import golaya_v7_tablicy as gt  # noqa: E402
from test_golaya_v7 import TK, bary  # noqa: E402


def test_p_po_perestanovkam_k_plyus_1_cherez_n_plyus_1(tmp_path):
    """p = (k+1)/(n+1), k — число контролей не хуже центра; мутация p = k/n (даёт 0 при k = 0) падает."""
    raw, im = bary()
    gt.tablicy(gv.iz_barov(raw, im, "train4", TK), "train4", tmp_path, n=3)
    for nm in ("k2", "placebo"):
        t = pd.read_csv(tmp_path / f"{nm}_train4.csv", comment="#")
        k = np.rint(t["dolya_ne_huzhe_centra"] * t["n"])
        assert (t["n"] == 3).all()
        assert np.allclose(t["p_perm"], (k + 1) / (t["n"] + 1)), nm
        assert (t["p_perm"] > 0).all()


def test_tercil_okno_do_dnya_resheniya_myutaciya_s_dnem_d():
    """Окно из 29 дней до d меньше минимума 30 → пусто; мутация (окно включает день d, 30 значений) вернула бы терциль."""
    sl = np.full((100, 1), 0.7)               # один тикер: окно = число дней
    assert gt.tercil(sl, 29, 0.9) == ""
    assert gt.tercil(sl, 30, 0.9) in (1, 2, 3)
