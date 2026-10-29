"""R-ж: p по k2 — (1+#{≥})/(n+1) по конечным; nan не «≥»; nan-факт или пустой k2 -> nan."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import r1_zh  # noqa: E402

nan = float("nan")


def test_minimalnyj_p_odin_k_n_plus_odin():
    assert r1_zh.p_k2_znachenie(10.0, [1.0, 2.0, 3.0, 4.0]) == 1 / 5


def test_nan_v_k2_ne_schitayutsya_i_ne_vhodyat_v_n():
    v = [1.0, nan, 2.0, nan, 20.0]
    assert r1_zh.p_k2_znachenie(10.0, v) == 2 / 4          # n = 3 конечных, одно ≥


def test_nan_nablyudenie_daet_nan():
    assert np.isnan(r1_zh.p_k2_znachenie(nan, [1.0, 2.0]))


def test_ves_k2_nan_daet_nan_a_ne_nol():
    assert np.isnan(r1_zh.p_k2_znachenie(1.0, [nan, nan, nan]))


def test_mutaciya_mean_ge_s_nan_kak_false_lovitsya():
    def stara(fakt, v):
        return float((np.asarray(v, float) >= fakt).mean())
    assert stara(1.0, [nan, nan]) == 0.0                   # старая формула: ложный p = 0
    assert np.isnan(r1_zh.p_k2_znachenie(1.0, [nan, nan]))
    assert stara(10.0, [1.0, 2.0]) == 0.0 and r1_zh.p_k2_znachenie(10.0, [1.0, 2.0]) == 1 / 3


def test_metka_stroki_ne_H2():
    assert r1_zh.NAMES["hyb"][1] == "диагностика (не H2)"
