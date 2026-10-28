# -*- coding: utf-8 -*-
"""Диагностика прохода 1: ранг σ без заглядывания (Ж-091 п. 3), терцили, метрики сделок. Синтетика, сеть и данные не нужны."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from prohod1_diag_metriki import metriki, rang_sigma, tercil, terciles  # noqa: E402


def test_rang_ne_zaglyadyvaet_vpered():
    rng = np.random.default_rng(1)
    sg = rng.random(600)
    e = 400
    r = rang_sigma(sg, e)
    fut = sg.copy()
    fut[e:] = rng.random(len(sg) - e) * 100          # будущее ≥ e (день входа и позже) меняется — ранг обязан остаться
    assert r == rang_sigma(fut, e)
    leak = sg.copy()
    leak[e - 1] = 1e9                                 # последний известный день изменился — ранг обязан измениться (тест видит ввод)
    assert rang_sigma(leak, e) > r


def test_rang_okno_252_i_minimum():
    sg = np.arange(1000, dtype=float)
    assert rang_sigma(sg, 600) > 0.99                 # значение на e−1 больше всех в окне
    assert math.isnan(rang_sigma(sg, 100))            # меньше 126 значений в истории
    assert math.isnan(rang_sigma(np.full(600, np.nan), 400))


def test_tercil_granicy():
    assert [tercil(x) for x in (0.0, 0.33, 0.34, 0.66, 0.67, 1.0)] == [0, 0, 1, 1, 2, 2]
    assert tercil(float("nan")) == -1


def test_metriki_pustaya_lenta_i_znak():
    pusto = pd.DataFrame({"ret": [], "rang": []})
    m = metriki({}, pusto)
    assert m["sdelok"] == 0 and math.isnan(m["ir_sdelki"])
    df = pd.DataFrame({"ret": [0.02, -0.01, 0.03, 0.01], "rang": [0.1, 0.5, 0.9, 0.2]})
    m = metriki({"sharpe": 1.0}, df)
    assert math.isclose(m["ir_sdelki"], df.ret.mean() / df.ret.std(ddof=1))
    assert math.isclose(m["sharpe_rasch"], m["ir_sdelki"] * math.sqrt(4 / 4))
    neg = metriki({}, df.assign(ret=-df.ret))
    assert neg["ir_sdelki"] < 0 < m["ir_sdelki"]      # перепутанный знак ловится


def test_terciles_schitayut_po_gruppam():
    df = pd.DataFrame({"ret": [0.01, 0.02, -0.01, 0.03], "rang": [0.1, 0.2, 0.9, 0.5]})
    t = terciles(df).set_index("terc")
    assert t.loc[0, "sdelok"] == 2 and math.isclose(t.loc[0, "sr_sdelki"], 0.015) and t.loc[0, "dolya_pribyl"] == 1.0
    assert t.loc[2, "dolya_pribyl"] == 0.0 and t.loc[1, "sdelok"] == 1
