"""V7 лист 1b-C, G25: расчёт оценки на синтетике с известным ответом. Мутации: Δ не от уровня K1; знак Δ перевёрнут; даты рядов не одинаковые (сдвиг одного ряда на +1);
N Холма = числу строк; «go»: условия 3, 5, 6 (1 % крайних дат, концентрация выигрыша в одном году, подтверждение 2022–2025)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s14 import v7_1b_ev as EV, v7_1m_stat as ST

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)
NAMES = EV.REFS + EV.CFGS


def make(noise_f=0.1, noise_b=0.15, seed=0):
    """rv — лог-нормальный; база = rv·exp(шум_b); конфигурация = rv·exp(шум_f). Возвращает rv, словарь рядов (REFS + cfg1…3), t⁺."""
    g = np.random.default_rng(seed)
    rv = np.exp(g.normal(0, 0.5, N))
    s = {k: rv * np.exp(g.normal(0, noise_b, N)) for k in EV.REFS}
    s.update({k: rv * np.exp(g.normal(0, noise_f, N)) for k in EV.CFGS})
    return rv, s, IDX.to_numpy("datetime64[ns]")


def row(t, w, b):
    return t[(t["window"] == w) & (t["base"] == b)].iloc[0]


def test_delta_known_answer_is_percent_of_k1_level_and_negative_when_model_better():
    rv, s, tp = make()
    t = EV.tests_table(rv, s["cfg1"], {k: s[k] for k in EV.REFS}, tp)
    m = ST.wmask(tp, "2018_2021")
    q = lambda f: Q.ql(rv[m], f[m])
    want = 100 * (q(s["cfg1"]) - q(s["V6:A0"])).mean() / q(s["K1"]).mean()
    r = row(t, "2018_2021", "V6:A0")
    assert abs(r["delta_pct"] - want) < 1e-12 and r["delta_pct"] < 0 and r["lo90"] < r["delta_pct"] < r["hi90"]
    assert abs(row(t, "2018_2021", "K1")["delta_pct"] - 100 * (q(s["cfg1"]) - q(s["K1"])).mean() / q(s["K1"]).mean()) < 1e-12


def test_mutation_delta_from_base_level_instead_of_k1_is_caught(monkeypatch):
    rv, s, tp = make()
    refs = {k: s[k] for k in EV.REFS}
    honest = EV.tests_table(rv, s["cfg1"], refs, tp)
    orig = EV.EV.delta_ci
    monkeypatch.setattr(EV.EV, "delta_ci", lambda qx, qy, k1, **kw: orig(qx, qy, float(np.mean(qy)), **kw))
    mut = EV.tests_table(rv, s["cfg1"], refs, tp)
    assert abs(row(mut, "2018_2021", "V6:A0")["delta_pct"] - row(honest, "2018_2021", "V6:A0")["delta_pct"]) > 1e-3


def test_mutation_sign_of_delta_flipped_is_caught(monkeypatch):
    rv, s, tp = make()
    refs = {k: s[k] for k in EV.REFS}
    honest = EV.tests_table(rv, s["cfg1"], refs, tp)
    orig = EV.EV.delta_ci
    monkeypatch.setattr(EV.EV, "delta_ci", lambda qx, qy, k1, **kw: orig(qy, qx, k1, **kw))
    mut = EV.tests_table(rv, s["cfg1"], refs, tp)
    h, m = row(honest, "2018_2021", "V6:A0"), row(mut, "2018_2021", "V6:A0")
    assert h["delta_pct"] < 0 < m["delta_pct"] and abs(h["delta_pct"] + m["delta_pct"]) < 1e-12


def test_same_dates_and_mutation_series_shifted_plus_one_day_is_caught():
    rv, s, tp = make()
    frames = {k: pd.DataFrame({"F": v}, index=IDX) for k, v in s.items()}
    assert EV.same_dates(frames)
    frames["cfg1"] = frames["cfg1"].set_index(frames["cfg1"].index + pd.Timedelta(days=1))
    assert not EV.same_dates(frames)
    shifted = dict(s, cfg1=np.roll(s["cfg1"], 1))                                    # тот же ряд, сдвинутый на одну дату: выравнивание нарушено
    honest = EV.tests_table(rv, s["cfg1"], {k: s[k] for k in EV.REFS}, tp)
    mut = EV.tests_table(rv, shifted["cfg1"], {k: s[k] for k in EV.REFS}, tp)
    assert row(mut, "2018_2021", "V6:A0")["delta_pct"] > row(honest, "2018_2021", "V6:A0")["delta_pct"] + 1.0


def test_holm_uses_program_n_not_row_count_and_mutation_is_caught(monkeypatch):
    rv, s, tp = make(noise_f=0.12, noise_b=0.13)
    refs = {k: s[k] for k in EV.REFS}
    t = EV.tests_table(rv, s["cfg1"], refs, tp)
    sub = t[t["window"] == "2018_2021"]
    p = sub["p_one"].to_numpy()
    order = np.argsort(p)
    want = np.empty(len(p))
    want[order] = np.minimum(1.0, np.maximum.accumulate((ST.N_HOLM - np.arange(len(p))) * p[order]))
    assert np.allclose(sub["p_holm_N"].to_numpy(), want, rtol=0, atol=1e-12)
    monkeypatch.setattr(EV, "N_CP", {"N": len(p), "N_wide": len(p)})
    mut = EV.tests_table(rv, s["cfg1"], refs, tp)
    assert (mut[mut["window"] == "2018_2021"]["p_holm_N"].to_numpy() < sub["p_holm_N"].to_numpy() - 1e-9).any()


def go(f, s, tp, rv):
    return EV.go_table(rv, f, {k: s[k] for k in EV.REFS}, tp).set_index("cond")


def test_go_all_six_pass_when_model_is_better_everywhere_and_known_gap():
    rv, s, tp = make(noise_f=0.05, noise_b=0.2)
    g = go(s["cfg1"], s, tp, rv)
    assert bool(g.loc[0, "passed"]) and all(bool(g.loc[i, "passed"]) for i in range(1, 7))
    m = ST.wmask(tp, "2018_2021")
    q = Q.ql(rv[m], s["cfg1"][m])
    best = min(Q.ql(rv[m], s[b][m]).mean() for b in EV.GO_BASES)
    assert abs(g.loc[1, "value"] - (q.mean() - best)) < 1e-12 and g.loc[0, "value"] == 6.0


def test_go_condition6_fails_when_model_is_worse_on_2022_2025():
    rv, s, tp = make(noise_f=0.05, noise_b=0.2)
    f = s["cfg1"].copy()
    bad = ST.wmask(tp, "2022_2025")
    f[bad] = rv[bad] * np.exp(np.random.default_rng(5).normal(0, 0.6, bad.sum()))
    g = go(f, s, tp, rv)
    assert bool(g.loc[1, "passed"]) and not bool(g.loc[6, "passed"]) and not bool(g.loc[0, "passed"])


def test_go_condition5_fails_when_whole_gain_comes_from_one_year():
    rv, s, tp = make(noise_f=0.2, noise_b=0.2)
    f = s["cfg1"].copy()
    y = ST.wmask(tp, "2018")
    f[y] = rv[y] * np.exp(np.random.default_rng(3).normal(0, 0.01, y.sum()))
    f[~y] = s["V6:A2"][~y]                                                              # в остальные годы — не лучше базы
    g = go(f, s, tp, rv)
    assert not bool(g.loc[5, "passed"]) and g.loc[5, "value"] > 0.5 and not bool(g.loc[0, "passed"])


def test_go_condition3_trims_exactly_one_percent_of_dates_and_condition1_does_not():
    rv, s, tp = make(noise_f=0.1, noise_b=0.15)
    m = ST.wmask(tp, "2018_2021")
    k = int(round(Q.TRIM * m.sum()))
    f = s["cfg1"].copy()
    pos = np.flatnonzero(m & ST.wmask(tp, "2019_2021"))[:k]
    f[pos] = rv[pos] * np.exp(3.0)                                                      # k дат с огромной потерей ряда
    g = go(f, s, tp, rv)
    assert not bool(g.loc[1, "passed"]) and bool(g.loc[3, "passed"])
    assert g.loc[3, "value"] < -Q.MARGIN < g.loc[1, "value"]                           # без k дат — выигрыш; с ними — проигрыш


def test_years_table_known_answer_gain_times_n():
    rv, s, tp = make(noise_f=0.05, noise_b=0.2)
    yt = EV.years_table(rv, s["cfg1"], {k: s[k] for k in EV.REFS}, tp)
    r = yt[yt["year"] == 2019].iloc[0]
    m = ST.wmask(tp, "2019")
    best = min(Q.ql(rv[m], s[k][m]).mean() for k in ("K2", "K3"))
    assert r["n"] == m.sum() and abs(r["gain_times_n"] - (best - Q.ql(rv[m], s["cfg1"][m]).mean()) * m.sum()) < 1e-9


def test_metrics_windows_and_mcs_shapes_on_same_dates():
    rv, s, tp = make()
    m = EV.metrics_table(rv, s, tp)
    assert set(m["window"]) == set(EV.METR_WIN) and len(m) == len(s) * len(EV.METR_WIN)
    assert m[(m["series"] == "K1") & (m["window"] == "all")]["n"].iloc[0] == N
    mc = EV.mcs_table(rv, s, tp)
    assert set(mc["window"]) == set(EV.MCS_WIN) and mc["in_mcs"].any()
