"""R-е: колонка status таблицы долей: test1 — реверсный маркер, train4 — прежнее значение."""
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

STARYJ = "разработка 2016–2019, до аудита"
REVERS = "реверс раунд 1, 2022–2024, справочно, без выбора"


def _tab(period):
    idx = pd.bdate_range("2015-06-01" if period == "train4" else "2021-06-01", periods=1100)
    imoex = pd.Series(100 + np.cumsum(np.sin(np.arange(1100) / 7)), index=idx)
    x = np.full(len(idx), 0.001)
    return r1e.tablicy(period, imoex, {"A": (idx, x, x)})["fazy_doli"]


def test_status_test1_reversnyj_vo_vseh_strokah():
    assert set(_tab("test1")["status"]) == {REVERS}


def test_status_train4_prezhnij_pobajtno():
    assert set(_tab("train4")["status"]) == {STARYJ}
