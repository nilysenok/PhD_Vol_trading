"""Шаг 3′ D/B: экран «доля закрытых дней» снят (Ъ 27.10, 15:10). Точки с долей 0,05 и 0,8 остаются в сетке;
мутация «экран вернули» роняет тест. python3 -m pytest tests/test_s9_shag_db_ekran.py -q
"""
import inspect
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scripts" / "analysis")]

import s9_shag_db as DB  # noqa: E402

from fastbt import s9  # noqa: E402

N = 700


def _eng(cc):
    rng = np.random.RandomState(5)
    S, RV = np.exp(rng.normal(-4, 0.4, (2, N))), 18 + np.abs(rng.normal(0, 5, (2, N)))
    return types.SimpleNamespace(sigma=lambda s: S, rvi=lambda lag: np.c_[np.full((2, lag), np.nan), RV[:, :N - lag]],
                                 n_t=np.array([N, N - 50]), calib_closed=lambda G: cc)


def _screened_points():
    """Мутация: экран возвращён — точки с долей вне [0,10; 0,50] выбрасываются (кроме масштаба)."""
    src = textwrap.dedent(inspect.getsource(DB.points))
    old = "        out.append((f, lab, G, scale, eng.calib_closed(G)))"
    assert src.count(old) == 1
    new = "        cc = eng.calib_closed(G)\n        if scale or 0.10 <= cc <= 0.50:\n            out.append((f, lab, G, scale, cc))"
    ns = dict(vars(DB))
    exec(src.replace(old, new), ns)
    return ns["points"]


@pytest.mark.parametrize("cc", [0.05, 0.3, 0.8])
@pytest.mark.parametrize("podhod", ["D", "B"])
def test_vse_tochki_v_s8(podhod, cc):
    n = len(DB.points(podhod, _eng(0.3), "x", s9))
    pts = DB.points(podhod, _eng(cc), "x", s9)
    assert len(pts) == n and n > 0 and all(p[4] == cc for p in pts)
    assert len(pts[0]) == 5                           # флага «отобрана» нет


def test_mutatsiya_ekran_vernuli_ronyaet():
    n = len(DB.points("D", _eng(0.3), "x", s9))
    bad = _screened_points()("D", _eng(0.8), "x", s9)
    assert len(bad) < n                               # мутант теряет точки; боевой test_vse_tochki_v_s8[0.8-D] — нет


def test_B_ekran_ne_primenyalsya():
    """B: масштаб (scale=True) — экран и раньше не действовал; мутант с экраном число B-точек не меняет."""
    n = len(DB.points("B", _eng(0.8), "x", s9))
    assert len(_screened_points()("B", _eng(0.8), "x", s9)) == n
