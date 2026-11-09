"""V7 лист 2b, G28–G29: парные сравнения 1–7 (гипотезы H1, H2, H3), правило 6.2, Δ в % K1, стресс, Холм N, версия 1 рядом — на синтетике с известным ответом.
Мутации: знак Δ; правило 6.2 по окну 2022+; стресс «без 1 %» не отбрасывает дни; N Холма = числу строк; v1 и v2 перепутаны."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q
from models_v2.s14 import v7_1b_ev as EV, v7_1m_stat as ST, v7_2_ev as E2

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)
TP = IDX.to_numpy("datetime64[ns]")


def make(seed=0, nx=0.05, ny=0.4):
    g = np.random.default_rng(seed)
    rv = np.exp(g.normal(0, 0.5, N))
    s = {k: rv * np.exp(g.normal(0, 0.3, N)) for k in E2.CTRL + E2.CFGS}
    s["cfg6"], s["cfg4"] = rv * np.exp(g.normal(0, nx, N)), rv * np.exp(g.normal(0, ny, N))        # H2: 6 заведомо лучше 4
    return rv, s


def row(t, x, y, w):
    return t[(t["x"] == x) & (t["y"] == y) & (t["window"] == w)].iloc[0]


def test_pairs_known_answer_x_better_gives_rule62_yes_and_reverse_no():
    rv, s = make()
    t = E2.pairs_table(rv, s, TP, pairs=(("H2", "cfg6", "cfg4"), ("H2", "cfg4", "cfg6")))
    a, b = row(t, "cfg6", "cfg4", "2018_2021"), row(t, "cfg4", "cfg6", "2018_2021")
    m = ST.wmask(TP, "2018_2021")
    want = 100 * (Q.ql(rv[m], s["cfg6"][m]) - Q.ql(rv[m], s["cfg4"][m])).mean() / Q.ql(rv[m], s["K1"][m]).mean()
    assert abs(a["delta_pct"] - want) < 1e-12 and a["delta_pct"] < 0 and a["lo90"] < a["delta_pct"] < a["hi90"]
    assert a["p_one"] < 0.05 and a["years_neg"] == 4 and a["rule62"] == "да"
    assert b["delta_pct"] > 0 and b["rule62"] == "нет" and abs(a["delta_pct"] + b["delta_pct"]) < 1e-9


def test_mutation_rule62_on_window_2022_plus_is_caught():
    """X лучше Y только в 2022–2025: правило 6.2 на окне разработки — «нет»; в окне 2022+ правило не применяется («—»)."""
    rv, s = make(nx=0.4, ny=0.05)
    late = ST.wmask(TP, "2022_2025")
    s["cfg6"] = np.where(late, rv * np.exp(np.random.default_rng(5).normal(0, 0.02, N)), s["cfg6"])
    t = E2.pairs_table(rv, s, TP, pairs=(("H2", "cfg6", "cfg4"),))
    assert row(t, "cfg6", "cfg4", "2018_2021")["rule62"] == "нет" and row(t, "cfg6", "cfg4", "2022_2025")["rule62"] == "—"
    assert row(t, "cfg6", "cfg4", "2022_2025")["delta_pct"] < 0                       # в 2022+ лучше — но в правило не входит


def test_rule62_function_needs_all_three_conditions():
    ok = dict(delta_q=-1.0, p_one=0.01, years_neg=3)
    assert E2.rule62(ok, "2018_2021") == "да"
    for bad in (dict(ok, delta_q=0.1), dict(ok, p_one=0.2), dict(ok, years_neg=2)):
        assert E2.rule62(bad, "2018_2021") == "нет"
    assert E2.rule62(ok, "2022_2025") == "—"


def test_stress_trims_exactly_one_percent_with_largest_gap_and_changes_delta():
    rv, s = make()
    s["cfg6"] = s["cfg6"].copy()
    m = ST.wmask(TP, "2018_2021")
    bad_days = np.flatnonzero(m)[:10]
    s["cfg6"][bad_days] = rv[bad_days] * 30                                               # 10 дней с огромной разностью к cfg4
    t = E2.pairs_table(rv, s, TP, pairs=(("H2", "cfg6", "cfg4"),))
    main, trim = row(t, "cfg6", "cfg4", "2018_2021"), row(t, "cfg6", "cfg4", "2018_2021 без 1 % крайних")
    assert main["n"] == int(m.sum()) and trim["n"] == main["n"] - int(round(0.01 * main["n"]))
    assert trim["delta_pct"] < main["delta_pct"] - 1e-3                                  # выбросы убраны — Δ меньше
    assert row(t, "cfg6", "cfg4", "2019_2021")["n"] == int(ST.wmask(TP, "2019_2021").sum())


def test_holm_uses_program_n_not_row_count(monkeypatch):
    rv, s = make()
    seen = []
    orig = E2.EVX.holm_both
    monkeypatch.setattr(E2.EVX, "holm_both", lambda p, cp0: seen.append((len(p), dict(cp0))) or orig(p, cp0))
    t = E2.pairs_table(rv, s, TP)
    assert len(seen) == 4 and all(n == len(E2.PAIRS) and cp == {"N": 733, "N_wide": 3166} for n, cp in seen)     # мутация «N = числу строк» передала бы len(p)
    k = t["window"] == "2018_2021"
    p = t.loc[k, "p_one"].to_numpy()
    assert np.allclose(t.loc[k, "p_holm_N"], SD.holm_n(p, ST.N_HOLM)) and np.allclose(t.loc[k, "p_holm_Nwide"], SD.holm_n(p, ST.N_WIDE))
    assert (t["p_holm_N"] >= t["p_one"] - 1e-15).all()


def test_pairs_cover_declared_hypotheses():
    got = {(h, x, y) for h, x, y in E2.PAIRS}
    assert {("H2", "cfg6", "cfg4"), ("H3", "cfg5", "cfg6"), ("H3", "cfg7", "cfg6"), ("H3", "cfg5", "cfg7")} <= got
    assert all(("H1", c, b) in got for c in ("cfg1", "cfg2", "cfg3") for b in ("V6:A0", "V6:A0g"))
    assert all((h, c, b) in got for h in ("4–7 против баз",) for c in ("cfg4", "cfg5", "cfg6", "cfg7") for b in ("V6:A0", "V6:A0g", "cfg1"))


def test_main_tables_shapes_and_go_for_all_seven():
    rv, s = make()
    t = E2.main_tables(rv, s, TP)
    assert len(t["GO"]) == 7 * 7 and set(t["GO"]["config"]) == set(E2.CFGS) and len(t["TESTS"]) == 7 * 5 * len(EV.REFS)
    assert set(t["METRICS"]["series"]) == set(E2.CTRL + E2.CFGS) and len(t["PAIRS"]) == len(E2.PAIRS) * 4


def test_v1_table_known_answer_and_mutation_swapped_versions_caught():
    rv, s = make()
    g = np.random.default_rng(9)
    s1 = {c: rv * np.exp(g.normal(0, 0.5, N)) for c in E2.CFGS}                          # v1 заведомо хуже v2
    t = E2.v1_table(rv, s, s1, TP)
    r = t[(t["config"] == "cfg6") & (t["window"] == "2018_2021")].iloc[0]
    m = ST.wmask(TP, "2018_2021")
    want = 100 * (Q.ql(rv[m], s1["cfg6"][m]) - Q.ql(rv[m], s["K3"][m])).mean() / Q.ql(rv[m], s["K1"][m]).mean()
    assert abs(r["d_k3_v1"] - want) < 1e-12 and abs(r["qlike_v1"] - Q.ql(rv[m], s1["cfg6"][m]).mean()) < 1e-12 and r["qlike_v1"] > r["qlike_v2"]
    sw = E2.v1_table(rv, {**s, **s1}, {c: s[c] for c in E2.CFGS}, TP)                    # мутация: версии поменяны местами
    q1 = sw[(sw["config"] == "cfg6") & (sw["window"] == "2018_2021")].iloc[0]["qlike_v1"]
    assert abs(q1 - r["qlike_v1"]) > 1e-3 and abs(q1 - r["qlike_v2"]) < 1e-12            # колонка v1 действительно берёт аргумент v1: перестановка меняет числа
