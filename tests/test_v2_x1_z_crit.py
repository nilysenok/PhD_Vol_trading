"""Этап Z (Ж-246), критерии K1–K5 — известный ответ на синтетике (каждое условие по отдельности роняет итог; ZB0 и ZCp критериев не получают); Холм N = 749 (мутация N = 1 ловится); состояния E/N;
таблицы на синтетике имеют ожидаемую форму; отчёт — генератор из таблиц (подмена числа в таблице меняет отчёт)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_crit as CR, x1_z_doc as DOC, x1_z_eval as EZ, x1_z_synth as SY, x1_z_walk as W


def frames(**over):
    """Таблицы, на которых проходят все K1–K5 у ZA, ZB, ZC; `over` — подмена одной величины."""
    pr, st, yr = [], [], []
    for y in CR.HYP:
        pr += [dict(y=y, base="X1", window="all", delta_pct=-0.5, delta_q=-0.001), dict(y=y, base="X1", window="2018_2021", delta_pct=-0.2, delta_q=-0.0005), dict(y=y, base=CR.CTRL[y], window="all", delta_pct=-0.3, delta_q=-0.0007)]
        st.append(dict(y=y, state="N", window="all", delta_pct=0.05))
        yr += [dict(y=y, base="X1", year=k, delta_q=-0.001 if k < 2023 else 0.001) for k in range(2018, 2027)]            # ≤ 0 в 5 годах из 8 (2018–2022)
    pr, st, yr = pd.DataFrame(pr), pd.DataFrame(st), pd.DataFrame(yr)
    if "k1" in over:
        pr.loc[(pr["y"] == "ZB") & (pr["base"] == "X1") & (pr["window"] == "all"), "delta_pct"] = over["k1"]
    if "k2" in over:
        st.loc[st["y"] == "ZB", "delta_pct"] = over["k2"]
    if "k3" in over:
        yr.loc[(yr["y"] == "ZB") & (yr["year"] == 2022), "delta_q"] = over["k3"]
    if "k4" in over:
        pr.loc[(pr["y"] == "ZB") & (pr["base"] == "ZB0"), ["delta_pct", "delta_q"]] = over["k4"]
    if "k5" in over:
        pr.loc[(pr["y"] == "ZB") & (pr["window"] == "2018_2021"), "delta_pct"] = over["k5"]
    return pr, st, yr


def verdict(**over):
    c = CR.crit_table(*frames(**over))
    b = c[c["y"] == "ZB"].set_index("crit")
    return {k: bool(b.at[k, "passed"]) for k in ("K1", "K2", "K3", "K4", "K5")}, b.at["итог", "name"], c


def test_all_pass_continues_and_controls_get_no_criteria() -> None:
    ok, name, c = verdict()
    assert all(ok.values()) and name == "продолжить" and set(c["y"]) == set(CR.HYP) and not {"ZB0", "ZCp"} & set(c["y"])
    assert (c[c["crit"] == "итог"]["name"] == "продолжить").all()


@pytest.mark.parametrize("k,val,fail", [("k1", 0.0, "K1"), ("k1", 0.3, "K1"), ("k2", 0.11, "K2"), ("k3", 0.001, "K3"), ("k4", 0.0, "K4"), ("k4", 0.2, "K4"), ("k5", 0.11, "K5")])
def test_each_condition_alone_closes_the_hypothesis(k, val, fail) -> None:
    ok, name, c = verdict(**{k: val})
    assert [x for x, v in ok.items() if not v] == [fail] and name == "закрыть"
    other = c[(c["y"] != "ZB") & (c["crit"] == "итог")]["name"]
    assert (other == "продолжить").all()                       # остальные гипотезы не затронуты


def test_boundaries_are_inclusive_for_k2_k5_and_k3_counts_non_positive_years() -> None:
    ok, name, c = verdict(k2=0.10, k5=0.10)
    assert all(ok.values()) and name == "продолжить"
    ok3, _, c3 = verdict(k3=0.0)
    assert ok3["K3"] and c3[(c3["y"] == "ZB") & (c3["crit"] == "K3")]["strict_years_neg"].iloc[0] == 4          # Δ = 0 засчитан (≤ 0), строгих лет 4


def series_and_zs(n=2100, seed=0):
    x1, D = SY.make(n, seed, p_e=0.15)
    zs = {k: W.walk(x1, D, k) for k in W.KINDS}
    rng = np.random.default_rng(1)
    rv, X = x1["rv"].to_numpy(), x1["F"].to_numpy()
    s = {k: X * np.exp(rng.normal(0, 0.1, n)) for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "Y1", "Y2", "Y3")}
    s.update({"X1": X, **{k: f["F"].to_numpy() for k, f in zs.items()}})
    return x1, D, zs, s, rv


@pytest.fixture(scope="module")
def synth_tables():
    x1, D, zs, s, rv = series_and_zs()
    t = CR.main_tables(rv, s, x1["t_plus"], x1["d"], D["e"], zs)
    return t, x1, D, zs, s, rv


def test_main_tables_shape_holm_n_749_and_states(synth_tables) -> None:
    t, x1, D, zs, s, rv = synth_tables
    assert set(t) == {"METRICS", "PAIRS", "RULE62", "YEARS", "STATES", "CELLS", "GO", "MCS", "STRESS", "CRIT", "PARAMS"}
    pr = t["PAIRS"]
    assert len(pr) == len(EZ.PAIRS) * len(EZ.PWIN) and {"ZB|ZB0", "ZC|ZCp", "ZA|ZB"} <= {a + "|" + b for a, b in zip(pr["y"], pr["base"])}
    k = (pr["window"] == "all") & pr["p_one"].notna()
    assert np.isclose(pr.loc[k, "p_holm_N"].min(), min(1.0, pr.loc[k, "p_one"].min() * EZ.N_HOLM)) and (pr.loc[k, "p_holm_Nwide"] >= pr.loc[k, "p_holm_N"]).all()
    st = t["STATES"]
    full = st[(st["window"] == "all") & (st["y"] == "ZA")].set_index("state")
    assert full.at["E", "n"] + full.at["N", "n"] == len(x1) and int(full.at["E", "n"]) == int(D["e"].sum())
    assert set(t["CRIT"]["y"]) == set(CR.HYP) and (t["PARAMS"]["y"].isin(EZ.ZS)).all()


def test_holm_n_equals_row_count_mutation_is_caught(synth_tables, monkeypatch) -> None:
    t, x1, D, zs, s, rv = synth_tables
    with pytest.raises(ValueError):
        EZ.EVX.holm_both(np.array([0.01, 0.02]), {"N": 1, "N_wide": 1})          # N меньше числа тестов — отказ библиотеки
    monkeypatch.setattr(EZ, "N_HOLM", len(EZ.PAIRS))                              # мутация: N = числу строк окна (класс № 103)
    mut = EZ.pairs_table(rv, s, x1["t_plus"], x1["d"])
    a = t["PAIRS"]
    assert not np.allclose(a["p_holm_N"].fillna(-1), mut["p_holm_N"].fillna(-1))


def test_report_is_generated_from_tables_and_substituted_number_changes_it(synth_tables) -> None:
    t, x1, D, zs, s, rv = synth_tables
    num = dict(n_dates=len(x1), first="a", last="b", pairs_at_activation=124, min_pairs=120, n_ext_at_activation=31, min_ext=30, n_inactive={k: 1 for k in W.KINDS}, n_extreme_days=5, n_degenerate_days=0, lams=[0.0], omegas=[0.0],
               v_full=126, v_ext=60, n_holm=EZ.N_HOLM, n_wide=EZ.N_WIDE, registered_at="t", configs_spent=5, notes=["x"])
    txt = DOC.build(t, num)
    assert txt.splitlines()[0].startswith("Статус: разработка, историческая пошаговая проверка, не независимая; ZB, ZC — после просмотра")
    t2 = {**t, "PAIRS": t["PAIRS"].assign(delta_pct=t["PAIRS"]["delta_pct"] + 0.123)}
    assert DOC.build(t2, num) != txt
    t3 = {**t, "CRIT": t["CRIT"].assign(value=t["CRIT"]["value"] + 1)}
    assert DOC.build(t3, num) != txt
    assert ("пройдено" in txt) and ("продолжить" in txt or "закрыть" in txt)
