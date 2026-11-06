"""C5 (Ж-206, волна C, поток XGB): J только по Y 2017–2021, оси с эффектом (порог, cap 4), годовой PIT-выбор (фолды y < Y, ничья в пользу И2).
Мутации: J по всем годам, ничья не в пользу И2, фолды y >= Y в pen_Y, нет cap. Известный ответ на синтетических фолдах."""
import pandas as pd
import pytest

from models_v2.s6.xgb import c_sel, critx


def folds(f) -> pd.DataFrame:
    return pd.DataFrame([{"Y": Y, "y": y, "q": f(Y, y)} for Y in range(2017, 2027) for y in range(2015, Y)])


BASE = folds(lambda Y, y: 0.2 + 0.001 * (y - 2015))


def pen(Y, q):
    v = [q(Y, y) for y in range(2015, Y)]
    m = sum(v) / len(v)
    return m + 0.25 * (max(v) - m)


def test_j_guide_by_hand_and_only_2017_2021():
    q = lambda Y, y: 0.2 + 0.001 * (y - 2015) + 0.01 * (Y == 2019)  # noqa: E731
    f = folds(q)
    assert c_sel.j_guide(f) == pytest.approx(sum(pen(Y, q) for Y in range(2017, 2022)) / 5, abs=1e-15)


def test_j_guide_ignores_later_years_and_mutation_all_years_would_not():
    worse = BASE.copy()
    worse.loc[worse["Y"] >= 2022, "q"] += 5.0
    assert c_sel.j_guide(worse) == pytest.approx(c_sel.j_guide(BASE), abs=1e-15)
    all_years = lambda f: critx.crit(f)["crit_pen"]  # noqa: E731 — мутант: J по всем годам
    assert all_years(worse) != pytest.approx(all_years(BASE))


def test_effect_axes_threshold_order_and_cap():
    d = {"a": -0.01, "b": -1e-13, "c": 0.0, "d": 0.02, "e": -0.03, "f": -0.002, "g": -0.005, "h": -0.004, "i": -0.02}
    assert c_sel.effect_axes(d) == ["e", "i", "a", "g"]                       # b (|Δ| <= 1e-12), c, d исключены; cap = 4 (мутант без cap вернул бы 6)
    assert len(c_sel.effect_axes(d, cap=99)) == 6 and c_sel.effect_axes({"a": 0.1, "b": 0.0}) == []


def better_at(Y_good, delta=-0.02):
    return folds(lambda Y, y: 0.2 + 0.001 * (y - 2015) + (delta if Y == Y_good else 0.0))


def test_annual_select_picks_candidate_only_where_better():
    choice, comp = c_sel.annual_select(BASE, {"A": better_at(2020)})
    assert choice[2020] == "A" and all(v is None for Y, v in choice.items() if Y != 2020)
    assert len(comp) == len(BASE) and (comp[comp["Y"] == 2020]["q"] < BASE[BASE["Y"] == 2020]["q"]).all()
    assert comp[comp["Y"] != 2020].reset_index(drop=True).equals(BASE[BASE["Y"] != 2020].reset_index(drop=True))


def test_tie_goes_to_i2_and_mutation_would_not():
    tie = BASE.assign(q=BASE["q"] - 1e-13)                                     # лучше на 1e-13 <= 1e-12 -> ничья
    choice, _ = c_sel.annual_select(BASE, {"A": tie})
    assert all(v is None for v in choice.values())
    strict = BASE.assign(q=BASE["q"] - 1e-6)
    assert all(v == "A" for v in c_sel.annual_select(BASE, {"A": strict})[0].values())


def test_choice_uses_only_past_folds_pit():
    cand = better_at(2020)
    spoiled = cand.copy()
    spoiled.loc[spoiled["y"] >= spoiled["Y"], "q"] = 0.0                       # y >= Y нет по построению; добавим такие фолды явно
    extra = pd.DataFrame([{"Y": Y, "y": y, "q": 0.0001} for Y in range(2017, 2027) for y in (Y, Y + 1)])
    leaky = pd.concat([cand, extra], ignore_index=True)
    assert c_sel.annual_select(BASE, {"A": leaky})[0] == c_sel.annual_select(BASE, {"A": cand})[0]
    a = pd.concat([BASE, extra], ignore_index=True)                            # и база с будущими фолдами не влияет
    assert c_sel.annual_select(a, {"A": cand})[0] == c_sel.annual_select(BASE, {"A": cand})[0]


def test_two_candidates_first_wins_ties_between_them_and_best_by_year():
    choice, _ = c_sel.annual_select(BASE, {"A": better_at(2020, -0.02), "B": better_at(2020, -0.05), "C": better_at(2023, -0.01)})
    assert choice[2020] == "B" and choice[2023] == "C" and choice[2017] is None


def test_choice_id_deterministic_and_sensitive():
    ch = {2017: None, 2018: "A"}
    assert c_sel.choice_id("l2", ch) == c_sel.choice_id("l2", dict(ch)) != c_sel.choice_id("l2", {2017: None, 2018: "B"}) != c_sel.choice_id("gamma", ch)
