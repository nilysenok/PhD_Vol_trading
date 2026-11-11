"""S-T (Ж-256): мутации (mutations.md № 125, 160) — подложенная ошибка в таблицах обязана ловиться `t_chk.problems`: знак Δ, база Δ, среднее годовых вместо дневных (№ 160), обрезка в основной колонке, потеря вкладов, неверный знаменатель среза."""
from __future__ import annotations

import copy

import pytest

from models_v2.s18 import r_eval_tab as RT, t_chk as C, t_slices as TS, t_table as TT


@pytest.fixture(scope="module")
def b():
    return TT.build()


def _mut(b):
    return copy.deepcopy(b["t"])


def test_honest_passes(b):
    assert C.problems(b["t"], b["inp"]) == []


def test_mutation_sign_of_delta(b):
    t = _mut(b)
    t["table"]["d_W5-G_pct"] = -t["table"]["d_W5-G_pct"]
    assert any("знак или база" in e for e in C.problems(t, b["inp"]))


def test_mutation_base_of_delta_is_candidate_not_control(b):
    t = _mut(b)
    tb = t["table"]
    ok = tb["status"] == "рассчитан"
    tb.loc[ok, "d_X1_pct"] = 100.0 * (tb.loc[ok, "qlike"] - tb.loc[tb["series"] == "X1", "qlike"].iloc[0]) / tb.loc[ok, "qlike"]
    assert any("знак или база" in e for e in C.problems(t, b["inp"]))


def test_mutation_mean_of_yearly_instead_of_daily_160(b):
    t = _mut(b)
    y = t["years"]
    per = y[(y["base"] == "-") & (y["kind"] == "period")]
    simple = per.groupby("series")["q_series"].mean()
    ok = t["table"]["status"] == "рассчитан"
    assert (abs(simple - t["table"].set_index("series")["qlike"].reindex(simple.index)) > 1e-6).any()   # подмена отличима
    t["table"].loc[ok, "qlike"] = t["table"].loc[ok, "series"].map(simple)
    e = C.problems(t, b["inp"])
    assert any("среднему дневных потерь" in x for x in e)


def test_mutation_trim_in_main_column(b):
    t = _mut(b)
    s = t["slices"]
    for h, bb in TS.PAIRS:
        i = s.index[(s["candidate"] == h) & (s["base"] == bb) & (s["kind"] == "основной")][0]
        s.at[i, "delta_pct"] = RT.trim_delta(b["inp"]["ls"][h], b["inp"]["ls"][bb])
    e = C.problems(t, b["inp"])
    assert any("основная строка" in x for x in e)


def test_mutation_denominator_is_candidate_or_all_dates(b):
    t = _mut(b)
    s = t["slices"]
    k = s["kind"] == "диагностика"
    s.loc[k, "q_base"] = s.loc[s["kind"] == "основной", "q_base"].iloc[0]
    assert any("знаменатель" in x for x in C.problems(t, b["inp"]))


def test_mutation_years_contribution_lost(b):
    t = _mut(b)
    y = t["years"]
    i = y.index[(y["series"] == "S-RCK") & (y["base"] == "W5-G") & (y["period"] == "2022")][0]
    y.at[i, "contrib_pct"] += 1e-6
    assert any("Σ вкладов" in x for x in C.problems(t, b["inp"]))


def test_mutation_trim_wrong_side_or_k(b):
    t = _mut(b)
    s = t["slices"]
    i = s.index[(s["candidate"] == "S-RCK") & (s["base"] == "W5-G") & (s["slice"].str.startswith("симметричная"))][0]
    s.at[i, "delta_pct"] += 0.01
    assert any("обрезка ≠ trim_delta" in x or "S_R_CONC" in x for x in C.problems(t, b["inp"]))


def test_mutation_trimsets_jaccard_and_count(b):
    t = _mut(b)
    ts = t["trimsets"]
    ts.loc[ts["kind"] == "пара сравнений", "jaccard"] = 1.0
    assert any("Жаккар" in x for x in C.problems(t, b["inp"]))
