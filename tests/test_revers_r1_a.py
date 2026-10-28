"""R-а: строка метрик на синтетике с известным ответом (без сети, без данных проекта)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_a  # noqa: E402


def test_stroka_hvosty_i_plata():
    x = np.tile([0.001, -0.001], 50) + np.linspace(0, 1e-6, 100)
    delo, baza = np.ones(100, bool), np.zeros(100)
    s = r1_a.D.stroka(x, delo, "hud", 0.0, baza)
    assert s["dni"] == 90 and s["dni_v_dele"] == 90
    full = r1_a.D.stroka(x, delo, None, 0.0, baza)
    assert full["dni"] == 100
    assert s["sharpe"] > full["sharpe"]                      # вычеркнули худшие дни
    paid = r1_a.D.stroka(x, delo, None, 0.05, np.full(100, 1e-3))
    assert paid["sharpe"] < full["sharpe"]                   # перепутанный знак платы поймается


def test_sverka_ловит_расхождение():
    import pandas as pd
    a = pd.DataFrame({"v": [1.0, 2.0]})
    assert r1_a.sverka(a, a) == 0.0
    assert r1_a.sverka(a + 1e-3, a) > 1e-9
