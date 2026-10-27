"""Шаги 1–3 главы 4 (v3): условия 1, 2, 3 С8 v2 расходятся на синтетической сетке; источник σ; рантайм-проверка рядов.
Мутации в памяти (str.replace с assert на одну замену): условие 2 → «Шарп > A»; условие 1 / условие 3 снято из итога;
strogo ↔ odno_isklyuchenie; harj читает гибрид. Данные не читаются."""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from test_s9_shag_c import _mutated, mini_grid, world  # noqa: E402

from fastbt import s9  # noqa: E402
from fastbt import shag_c as C  # noqa: E402

YRS = [str(y) for y in s9.YEARS]


def met(years, worst, sharpe):
    return dict(zip(YRS, years), worst=worst, sharpe=sharpe)


MA = met([0.5] * 4, -1.0, 1.0)
MK = met([0.5] * 4, -1.0, 1.2)
GOOD, LOW_P = 0.01, 0.20
# точка: (метрики, p к2) -> ожидаемые (u1_strogo, u1_odno, usl2, usl3)
POINTS = {
    "1_не_2": (met([0.9] * 4, -0.5, 1.1), GOOD, (True, True, False, True)),        # Шарп 1,1 > A, но < к3 1,2
    "1_2_не_3": (met([0.9] * 4, -0.5, 1.5), LOW_P, (True, True, True, False)),
    "3_из_4": (met([0.9, 0.9, 0.9, 0.4], -0.5, 1.5), GOOD, (False, True, True, True)),
    "не_1_но_2_3": (met([0.9, 0.9, 0.4, 0.4], -0.5, 1.5), GOOD, (False, False, True, True)),
    "всё": (met([0.9] * 4, -0.5, 1.5), GOOD, (True, True, True, True)),
}


def run_all(f=None, u1="strogo"):
    f = f or C.usloviya
    return {k: f(m, MA, MK, p, u1) for k, (m, p, _) in POINTS.items()}


def chk_conditions(f=None):
    """Для обоих режимов: u1_*, usl2, usl3 и итог совпадают с ожидаемыми на всех 5 точках."""
    for u1 in C.USLOVIYA1:
        for k, r in run_all(f, u1).items():
            s, o, c2, c3 = POINTS[k][2]
            assert (r["u1_strogo"], r["u1_odno_isklyuchenie"], r["usl2_k3"], r["usl3_k2"]) == (s, o, c2, c3), k
            assert r["usl1_c8"] == (s if u1 == "strogo" else o), (k, u1)
            assert r["proshla_c8v2"] == (r["usl1_c8"] and c2 and c3), (k, u1)


def test_conditions_diverge():
    rs = run_all()
    assert rs["1_не_2"]["usl1_c8"] and not rs["1_не_2"]["usl2_k3"]
    assert rs["1_2_не_3"]["usl1_c8"] and rs["1_2_не_3"]["usl2_k3"] and not rs["1_2_не_3"]["usl3_k2"]
    t = run_all(u1="strogo")["3_из_4"], run_all(u1="odno_isklyuchenie")["3_из_4"]
    assert not t[0]["proshla_c8v2"] and t[1]["proshla_c8v2"]
    assert [k for k, r in rs.items() if r["proshla_c8v2"]] == ["всё"]
    chk_conditions()


def test_unknown_uslovie1_raises():
    with pytest.raises(ValueError):
        C.usloviya(MA, MA, MK, 0.0, "stogo")
    assert C.USLOVIE1 == "strogo" and C.load_grid()["uslovie1"] == "strogo"


@pytest.mark.parametrize("old,new", [
    ('bool(m["sharpe"] > mK["sharpe"])', 'bool(m["sharpe"] > mA["sharpe"])'),             # условие 2 → Шарп > A
    ("proshla_c8v2=c1 and c2 and c3", "proshla_c8v2=c2 and c3"),                          # условие 1 снято
    ("proshla_c8v2=c1 and c2 and c3", "proshla_c8v2=c1 and c2"),                          # условие 3 снято
    ('{"strogo": all_years and rest, "odno_isklyuchenie": bool(r["passed"])}',
     '{"odno_isklyuchenie": all_years and rest, "strogo": bool(r["passed"])}'),           # режимы перепутаны
])
def test_mutations_caught(old, new):
    f = _mutated(C.usloviya, old, new, C.__dict__)
    with pytest.raises(AssertionError):
        chk_conditions(f)


def test_evaluate_outputs_both_flags_and_selection_table():
    import s9_shag_c
    res = s9_shag_c.tables(world(seed=5), "hyb", mini_grid(), 200, n_shift=60, with_sverka=False)
    t = C.tag(0.0004)
    df = res["setka"]
    assert {f"u1_strogo_{t}", f"u1_odno_isklyuchenie_{t}"} <= set(df.columns)
    assert (df[f"u1_strogo_{t}"].astype(bool) <= df[f"u1_odno_isklyuchenie_{t}"].astype(bool)).all()
    assert (df[f"usl1_c8_{t}"] == df[f"u1_strogo_{t}"]).all()                              # умолчание strogo
    assert {"u1_strogo", "u1_odno_isklyuchenie"} <= set(res["otbor"].columns)


# ---------- источник σ ----------
def _expected():
    """Файлы и столбцы рядов σ — из ch4_dev_sigma.py (имена выходов), а не из s9.SOURCES_V2."""
    txt = (ROOT / "scripts" / "analysis" / "ch4_dev_sigma.py").read_text(encoding="utf-8")
    files = set(re.findall(r"sigma_\w+_dev\.parquet", txt))
    return files, {"naive": {1: "sigma_naive_dev.parquet"},
                   "harj": {h: f"sigma_harj_b_h{h}_dev.parquet" for h in (1, 5)},
                   "hyb": {h: f"sigma_hyb_h{h}_dev.parquet" for h in (1, 5)}}


def chk_src():
    files, exp = _expected()
    assert set(C.SRC) == set(exp)
    for sg, hs in exp.items():
        assert set(C.SRC[sg]) == set(hs), sg
        for h, fn in hs.items():
            assert fn in files, fn
            path, col, hz = s9.SOURCES_V2[C.SRC[sg][h]]
            assert Path(path).name == fn and col == f"sigma_pred_h{h}" and hz == f"h{h}", (sg, h)
    grid = C.load_grid()
    for sg in C.SHAG:
        assert {p["src"] for p in C.plan(sg, grid)} == set(C.SRC[sg].values())


def test_src_reads_own_series():
    chk_src()


def test_mutation_harj_reads_hybrid(monkeypatch):
    monkeypatch.setitem(C.SRC, "harj", {1: "hyb_h1", 5: "hyb_h5"})
    with pytest.raises(AssertionError):
        chk_src()


class _Eng:
    def __init__(self, d):
        self.d = d

    def sigma(self, lab):
        return self.d[lab]


def test_runtime_identical_series_fail():
    a, b = np.random.RandomState(0).rand(2, 30), np.random.RandomState(1).rand(2, 30)
    ok = _Eng({"naive_h1": a, "naive_h5": a.copy(), "harj_h1": b, "harj_h5": b * 2, "hyb_h1": b + 1, "hyb_h5": b + 2})
    C.check_sigma_distinct(ok, list(ok.d))                           # naive_h1 = naive_h5 — по построению, допустимо
    bad = _Eng(dict(ok.d, harj_h1=ok.d["hyb_h1"].copy()))
    with pytest.raises(ValueError, match="совпадают побайтно"):
        C.check_sigma_distinct(bad, list(bad.d))
    nan = np.full((2, 5), np.nan)                                    # одинаковые NaN-ряды — тоже совпадение
    with pytest.raises(ValueError):
        C.check_sigma_distinct(_Eng({"harj_h1": nan, "hyb_h1": nan.copy()}), ["harj_h1", "hyb_h1"])
