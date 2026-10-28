"""R-ж: синтетика для затвора по терцилям σ̂ и параметров периода (без данных 2022+)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import cd_s3_sigma as S  # noqa: E402
import r1_zh  # noqa: E402


def test_zatvor_zakryvaet_verhnij_tercil_i_ne_smotrit_vperyod():
    a = np.arange(300, dtype=float)                       # растущая σ: каждый день — максимум окна
    z = S.zatvor(a)
    assert z[:125].all() and not z[126:].any()          # ранг не определён -> открыт; потом закрыт
    b = a.copy()
    b[200:] = -1e9                                        # будущее не влияет на решение в день 199
    assert S.zatvor(b)[199] == z[199]


def test_periody_test1_ne_zahodyat_za_2024():
    p = r1_zh.PERIODS["test1"]
    assert p["last"] == "2024-12-31" and max(p["gody"]) == 2024 and p["n_k2"] == 20
    assert r1_zh.PERIODS["train4"]["n_k2"] == 100
