"""Одно определение Шарпа в таблицах гл. 4 (владелец 27.10): Δ = Шарп(ddof=1) точки − Шарп(ddof=1) базы;
ЛВ даёт только p. Синтетика. Мутации в памяти (Δ из ЛВ, ddof=0, знак года не по годовым Шарпам) роняют тесты."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts" / "analysis"))
from test_s9_shag_c import mini_grid, world  # noqa: E402

from fastbt import shag0  # noqa: E402
from fastbt import shag_c as C  # noqa: E402
from fastbt import shag_c_tab as T  # noqa: E402
from fastbt.msp import sharpe  # noqa: E402
from fastbt.s9_stats import lw_sharpe_diff  # noqa: E402

IDX = pd.date_range("2018-01-01", "2021-12-31", freq="B")


def _p(x):
    return pd.DataFrame(dict(excess=x, total=x, exposure=1.0, turnover=0.0), index=IDX)


def _pair():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0002, 0.01, len(IDX))
    sg = np.where(IDX.year <= 2019, 0.0015, -0.0015)
    return _p(base + sg + rng.normal(0, 0.002, len(IDX))), _p(base)


def test_delta_is_difference_of_table_sharpes():
    p1, p0 = _pair()
    d = shag0.delta(p1, p0, 50)
    want = sharpe(p1["excess"].values) - sharpe(p0["excess"].values)
    assert abs(d["d_vs_A"] - want) < 1e-12
    assert abs(C.lw(p1, p0, 50)[0] - want) < 1e-12
    assert abs(shag0.d_sharpe(p1, p0) - want) < 1e-12
    # ЛВ (ddof=0) отличается — тест различает определения
    assert abs(lw_sharpe_diff(p1["excess"].values, p0["excess"].values, 50, 5)[0] - want) > 1e-6
    # p — по-прежнему ЛВ
    assert d["p_vs_A"] == lw_sharpe_diff(p1["excess"].values, p0["excess"].values, 50, 5)[2]


def test_year_signs_from_yearly_table_sharpes():
    p1, p0 = _pair()
    d = shag0.delta(p1, p0, 20)
    for y in shag0.YEARS:
        m = IDX.year == y
        want = sharpe(p1["excess"].values[m]) - sharpe(p0["excess"].values[m])
        assert d[f"znak_{y}"] == ("+" if want > 0 else "−")
    assert [d[f"znak_{y}"] for y in shag0.YEARS] == ["+", "+", "−", "−"]
    assert C.znaki(p1, p0) == {k: v for k, v in d.items() if k.startswith("znak_")}


def test_decomposition_exact_and_mutations(monkeypatch):
    eng, grid = world(seed=5), mini_grid()
    df, gs, bases = C.run_setka(eng, "hyb", grid, 100, n_shift=60, log=lambda *_: None)
    top = df.loc[df[f"sharpe_{C.tag(C.COMM_SEL)}"].idxmax()]
    g = eng.g_row(gs[(top["h"], top["family"], top["label"])])
    for c in T.COMM_MAIN:
        t = C.tag(c)
        assert abs(top[f"d_vs_A_{t}"] - (top[f"sharpe_{t}"] - top[f"sharpe_A_{t}"])) < 1e-12
        assert abs(top[f"d_vs_k3_{t}"] - (top[f"sharpe_{t}"] - top[f"sharpe_k3_{t}"])) < 1e-12
    raz = T.razlozhenie(eng, bases, g, 100)
    T.proverka_razlozheniya(raz, top)
    for r in (x for x in raz if x["rezhim"] == "разложение"):
        t = C.tag(r["comm_pct"] / 100)
        assert abs(r["chast1_k3_minus_A"] + r["chast2_C_minus_k3"] - (top[f"sharpe_{t}"] - top[f"sharpe_A_{t}"])) < 1e-12
        assert abs(r["summa_chastey_minus_d"]) < 1e-12
    # мутация: Δ из ЛВ — разложение перестаёт совпадать с табличными Шарпами
    lw_d = lambda p1, p0, reps: (C.fast_boot.lw_sharpe_diff(p1["excess"].values, p0["excess"].values, reps, 5)[0],
                                 0.5)  # noqa: E731
    monkeypatch.setattr(T, "lw", lw_d)
    bad = T.razlozhenie(eng, bases, g, 100)
    with pytest.raises(AssertionError):
        T.proverka_razlozheniya(bad, top)


def test_mutation_lw_delta_caught(monkeypatch):
    p1, p0 = _pair()
    want = sharpe(p1["excess"].values) - sharpe(p0["excess"].values)
    monkeypatch.setattr(shag0, "d_sharpe",
                        lambda a, b: lw_sharpe_diff(a["excess"].values, b["excess"].values, 20, 5)[0])
    assert abs(shag0.delta(p1, p0, 20)["d_vs_A"] - want) > 1e-6          # тест выше упал бы


def test_mutation_ddof0_in_main_column_caught(monkeypatch):
    p1, p0 = _pair()
    s0 = lambda x: float(np.mean(x) * np.sqrt(252) / np.std(x, ddof=0))  # noqa: E731
    assert abs(s0(p1["excess"].values) - sharpe(p1["excess"].values)) > 1e-6
    monkeypatch.setattr(shag0, "sharpe", s0)
    want = sharpe(p1["excess"].values) - sharpe(p0["excess"].values)
    assert abs(shag0.stats(p1, np.ones(len(p1), bool))["sharpe"] - sharpe(p1["excess"].values)) > 1e-6
    assert abs(shag0.delta(p1, p0, 20)["d_vs_A"] - want) > 1e-6


def test_mutation_year_sign_not_from_yearly_sharpes_caught(monkeypatch):
    """Знак года из общего Δ (а не из годового Шарпа) даёт «+» во всех годах — тест знаков падает.
    Знак из ЛВ-оценки года (ddof=0) совпал бы с ddof=1: оба множителя одинаковы — различить их нельзя, и не нужно."""
    p1, p0 = _pair()
    monkeypatch.setattr(shag0, "year_mask", lambda p, y: np.ones(len(p), bool))
    d = shag0.delta(p1, p0, 20)
    assert [d[f"znak_{y}"] for y in shag0.YEARS] != ["+", "+", "−", "−"]
