"""R-д (реверс раунд 1): пересчёт на длину периода и сверка с образцом #467. Синтетика, в сеть и в данные ≥ 2022 не ходит."""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/revers"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import r1_d  # noqa: E402


def test_na_god_tri_goda_vmesto_chetyreh():
    m = dict(sdelok=120, ir_sdelki=0.1, sdelok_v_god=30.0, sharpe_rasch=0.1 * math.sqrt(30))
    o = r1_d.na_god(m, 3)
    assert o["sdelok_v_god"] == 40.0
    assert abs(o["sharpe_rasch"] - 0.1 * math.sqrt(40)) < 1e-12
    assert r1_d.na_god(m, 4) == m and r1_d.na_god({}, 3) == {}


def test_test1_pishet_v_itog_train4_v_sverku():
    assert r1_d.PERIODS["test1"]["n_placebo"] == 20 and r1_d.PERIODS["test1"]["godov"] == 3
    assert r1_d.PERIODS["train4"]["out"].endswith("_sverka_train4")      # train4 не затирает итоговые таблицы


def test_sverka_s_obrazcom_467():
    n = 0
    for kind in ("tochki", "kontrol", "terciles"):
        for k in range(1, 7):
            f = ROOT / f"docs/tables/revers_r1/_sverka_train4/diag_{kind}_S{k}.csv"
            if f.exists():
                a = pd.read_csv(f, comment="#")
                b = pd.read_csv(ROOT / f"docs/tables/progon_e1/diag_{kind}_S{k}.csv", comment="#")
                pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-12)
                n += 1
    if n == 0:
        pytest.skip("сверочных таблиц нет: запустить r1_d.py schet S<k> --period train4")
