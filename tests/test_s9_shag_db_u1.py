"""Условие 1 С8 в шаге 3′ (D, B) — строгое: все 4 года лучше A (Ъ 27.10, 13:05); «3 из 4» — справочный столбец.
python3 -m pytest tests/test_s9_shag_db_u1.py -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scripts" / "analysis"), str(HERE)]

import yaml  # noqa: E402
from test_s9_shag0 import _world  # noqa: E402
from test_s9_shag_db import _inf_gate  # noqa: E402

import s9_shag_common as C  # noqa: E402

from fastbt import fast_kontrol as fk  # noqa: E402
from fastbt import s9  # noqa: E402

COMM = C.COMM_MAIN


def _point(uslovie1="strogo"):
    """Точка, у которой условия 2 и 3 выполнены, а база A обыгрывает её ровно в одном из 4 лет (3 из 4)."""
    ns = _world()
    g = _inf_gate(ns)
    m = s9.metrics(ns.port(ns.A_row * g, comm=COMM))
    mA = dict(m, sharpe=m["sharpe"] - 1, worst=m["worst"] - 1)
    mA.update({str(y): m[str(y)] - 1 for y in s9.YEARS})
    mA[str(s9.YEARS[0])] = m[str(s9.YEARS[0])] + 1
    base = C.make_base(ns, COMM, 50, uslovie1)
    base.update(mA=mA, m3=dict(sharpe=m["sharpe"] - 1))
    return C.eval_point(ns, fk.Kontrol(ns, COMM, n_shift=100), base, ns.A_row * g, g)


def test_three_of_four_fails_strict_selection_but_flagged_in_reference_column():
    r = _point()
    assert r["usl2"] is True and r["usl3"] is True            # условия 2–3 не виноваты
    assert r["u1_strogo"] is False and r["usl1"] is False and r["passed_v2"] is False
    assert r["u1_odno_isklyuchenie"] is True                  # справочно: 3/4 — True
    assert C.select(__import__("pandas").DataFrame([r])) is None


def test_mutation_select_by_odno_isklyuchenie_is_caught():
    r = _point("odno_isklyuchenie")                           # мутация: в отбор попало «3 из 4»
    assert r["passed_v2"] is True                             # …и точка ложно прошла — проверка выше упала бы


def test_yaml_both_grids_strict():
    for x in "DB":
        assert C.uslovie1_yaml(x) == "strogo"
        assert yaml.safe_load((C.DESIGNS / f"setka_{x}_v3.yaml").read_text(encoding="utf-8"))["uslovie1"] == "strogo"
