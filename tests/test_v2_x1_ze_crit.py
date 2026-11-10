"""Лист ZE (Ж-247 п. 4), оценка и критерии: Холм N = 751 / N_wide = 3207 (не число строк; мутация № 103); K1–K6 на таблицах известного ответа — каждое условие отдельно закрывает; окна «без 2022» и 2023–2025; K6 — по состоянию E за 2023–2025 против ZB."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ze_crit as CR


def tables(**kw):
    """Таблицы, где все K1–K6 выполнены; kw[K] = переопределение для отказа одного условия."""
    pr = pd.DataFrame([dict(y="ZE", base="X1", window="all", delta_pct=-0.5, delta_q=-0.001), dict(y="ZE", base="X1", window="2018_2021", delta_pct=0.05, delta_q=0.0001),
                       dict(y="ZE", base="ZEp", window="all", delta_pct=-0.3, delta_q=-0.0007)])
    st = pd.DataFrame([dict(y="ZE", state="N", window="all", base="X1", delta_pct=0.0, delta_q=0.0), dict(y="ZE", state="E", window="2023_2025", base="ZB", delta_pct=-1.0, delta_q=-0.01)])
    yr = pd.DataFrame([dict(y="ZE", base="X1", year=y, delta_q=-0.01 if y < 2023 else 0.0) for y in range(2018, 2026)])
    if "K1" in kw:
        pr.loc[0, ["delta_pct", "delta_q"]] = [0.2, 0.001]
    if "K2" in kw:
        st.loc[0, "delta_pct"] = 0.2
    if "K3" in kw:
        yr["delta_q"] = [0.01] * 4 + [-0.01] * 4                                   # только 4 года ≤ 0
    if "K4" in kw:
        pr.loc[2, ["delta_pct", "delta_q"]] = [0.1, 0.0003]
    if "K5" in kw:
        pr.loc[1, "delta_pct"] = 0.5
    if "K6" in kw:
        st.loc[1, ["delta_pct", "delta_q"]] = [2.0, 0.02]
    return pr, st, yr


def test_all_six_pass_continue() -> None:
    c = CR.crit_table(*tables())
    assert list(c["crit"]) == ["K1", "K2", "K3", "K4", "K5", "K6", "итог"] and c["passed"].all() and c.iloc[-1]["name"] == "продолжить" and c.iloc[-1]["value"] == 6.0


@pytest.mark.parametrize("k", ["K1", "K2", "K3", "K4", "K5", "K6"])
def test_each_condition_alone_closes(k) -> None:
    c = CR.crit_table(*tables(**{k: 1})).set_index("crit")
    assert not c.at[k, "passed"] and c.drop(index=[k, "итог"])["passed"].all() and c.at["итог", "name"] == "закрыть" and c.at["итог", "value"] == 5.0


def test_k5_k2_tolerance_is_inclusive_010_and_k6_zero_passes() -> None:
    pr, st, yr = tables()
    pr.loc[1, "delta_pct"], st.loc[0, "delta_pct"], st.loc[1, ["delta_pct", "delta_q"]] = 0.10, 0.10, [0.0, 0.0]
    assert CR.crit_table(pr, st, yr).iloc[-1]["name"] == "продолжить"


def test_holm_uses_program_n_751_not_row_count_and_mutation_is_caught(monkeypatch) -> None:
    t = pd.DataFrame(dict(window="all", p_one=[0.0001, 0.002, 0.04, 0.2, 0.5]))
    h = CR.holm(t, ("all",))
    assert CR.N_HOLM == 751 and CR.N_WIDE == 3207 and h["p_holm_N"].min() == pytest.approx(min(1.0, 0.0001 * 751)) and h["p_holm_Nwide"].min() == pytest.approx(min(1.0, 0.0001 * 3207))
    monkeypatch.setattr(CR, "N_HOLM", len(t))                       # мутация: N = числу строк таблицы
    assert CR.holm(t, ("all",))["p_holm_N"].min() == pytest.approx(0.0005) and CR.holm(t, ("all",))["p_holm_N"].min() < h["p_holm_N"].min()


def test_pair_list_has_two_configs_and_no_y5_y6() -> None:
    assert set(y for y, _ in CR.PAIRS) == {"ZE", "ZEp"} and ("ZE", "ZEp") in CR.PAIRS and ("ZE", "ZB") in CR.PAIRS and not any("Y5" in p or "Y6" in p for p in CR.PAIRS)


def test_diag_windows_ex2022_and_2023_2025_masks() -> None:
    tp = pd.DatetimeIndex(["2021-06-01", "2022-03-01", "2022-12-30", "2023-01-02", "2025-12-30", "2026-02-02"])
    assert list(CR.wmask(tp, None, "ex2022")) == [True, False, False, True, True, True] and list(CR.wmask(tp, None, "2023_2025")) == [False, False, False, True, True, False]


def test_diag_table_known_answer_delta_signs() -> None:
    n = 900
    tp = pd.bdate_range("2021-01-04", periods=n)
    d = tp - pd.offsets.BDay(1)
    rv, x1 = np.exp(np.random.default_rng(0).normal(0, 0.3, n)), np.ones(n)
    s = dict(X1=x1, ZB=x1, ZE=rv, ZEp=x1)                           # ZE = идеальный прогноз: Δ к X1 < 0
    t = CR.diag_table(rv, s, tp, d)
    assert set(t["window"]) == {"ex2022", "2023_2025"} and (t[(t["y"] == "ZE") & (t["base"] == "X1")]["delta_q"] < 0).all()
    assert (t[(t["y"] == "ZB") & (t["base"] == "X1")]["delta_q"] == 0).all() and (t[(t["y"] == "ZE") & (t["base"] == "ZB")]["delta_q"] < 0).all()
