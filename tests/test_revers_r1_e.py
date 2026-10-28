"""Реверс раунд 1, лист R-е: результат по фазам на синтетике с известным ответом."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("r1_e", ROOT / "scripts/revers/r1_e.py")
r1e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1e)


def test_po_fazam_synthetic_known_answer():
    n = 300
    idx = pd.bdate_range("2016-01-04", periods=n)
    faza = pd.Series(["рост_спокойно"] * 100 + ["падение_стресс"] * 100 + [np.nan] * 100, index=idx, dtype=object)
    x = np.where(np.arange(n) % 2 == 0, 0.011, 0.009)       # среднее 0,01, ненулевая σ
    x[100:200] *= -1
    baza = np.ones(n)
    rows = pd.DataFrame(r1e.po_fazam("A", x, baza, faza, 0.0)).set_index("faza")
    assert rows.dni["все"] == n and rows.dni["рост_спокойно"] == 100 and rows.dni["nan"] == 100
    assert rows.dni["падение_спокойно"] == 0 and np.isnan(rows.sharpe["падение_спокойно"])   # пустая фаза: NaN, не 0
    assert abs(rows.sredn_den["рост_спокойно"] - 0.01) < 1e-12 and abs(rows.sredn_den["падение_стресс"] + 0.01) < 1e-12
    assert rows.sharpe["рост_спокойно"] > 0 > rows.sharpe["падение_стресс"]                  # знак не перепутан
    pl = pd.DataFrame(r1e.po_fazam("A", x, baza, faza, 0.05)).set_index("faza")
    assert abs(pl.sredn_den["рост_спокойно"] - (0.01 - 0.05)) < 1e-12                       # плата за шорт вычтена


def test_train4_doli_match_478_table():
    csv = Path(__import__("os").environ.get("FASTBT_MAIN", ROOT)) / "data/iss_mix_2015_2019/IMOEX.csv"
    if not csv.exists():
        import pytest
        pytest.skip("нет IMOEX 2015–2019 в этой копии")
    s, e, gody, _ = r1e.PERIODY["train4"]
    im = r1e.zagruzit_imoex(csv, s, e)
    assert im.index.max() < pd.Timestamp("2020-01-01")                                      # граница чтения
    mine = r1e.doli_po_godam(r1e.fazy_dnej(im, im.index), gody)
    ref = pd.read_csv(ROOT / "docs/tables/msp/fazy_doli.csv", keep_default_na=False)
    ref["dni"] = ref.dni.astype(int)
    pd.testing.assert_frame_equal(mine.reset_index(drop=True), ref)
