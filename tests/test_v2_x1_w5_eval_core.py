"""W5-B, оценка на синтетике, чистые функции `x1_w5_eval_t` (W5_SPEC 5, 6, 11.6; mutations.md № 103, 125, 140, 158): Σ вкладов лет = Δ (1e-12), выброшенная дата ломает тождество; Холм N = 798 / N_wide = 3301
(известный ответ независимой реализации), мутация «N = числу строк»; знак Δ, база процентов, «лучше/хуже»; «подтверждено» только при p Холма < 0,05; пороги У1–У5 равны зарегистрированным 0,415 / 0,50 / 0,26 и
границы (≤) проверяются на выдуманных таблицах, мутация порога «из результата» ловится; чувствительность без пары 2022-02-24 и без самого влиятельного года — известный ответ; p по перестановкам. Только monkeypatch (№ 43)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_q as Q
from models_v2.s13 import v6_1_eval as EVX
from models_v2.s16 import x1_w1_stat as SS
from models_v2.s16 import x1_w5_eval as E
from models_v2.s16 import x1_w5_eval_t as T
from models_v2.s16 import x1_w5_gap as G

REAL = [T.__file__ and E.REP / n for n in E.OUTS] + [E.CKPT, G.REG.TRIALS, G.GAP_PIT]


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
        s["W5"] = sorted((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in G.OUT.rglob("*") if p.is_file()) if G.OUT.exists() else []
        return s
    before = st()
    yield
    assert st() == before


def synth(c: float = 0.01, seed: int = 5, n: int = 2300) -> tuple:
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2018-01-04", periods=n)
    L = np.exp(rng.normal(-8, 0.4, n))
    rv = L * np.exp(rng.normal(0.0, 0.5, n) + 0.125)
    F = {"X1": L, "K1": L * 1.5, "K2": L * 1.9, "K3": L * 2.1, "M": L * 1.8, "ZB": L, "W5-Ga": L, "W5-Gb": L, "W5-GK2": L * 1.9, "W5-G": L * np.exp(c)}
    return rv, tp, F


RV, TP, F = synth()


def ql(rv, f):
    r = rv / f
    return r - np.log(r) - 1.0


def holm(p: np.ndarray, n: int) -> np.ndarray:
    """Независимая реализация: p_(k)·(N − k + 1), накопленный максимум, ≤ 1."""
    o = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[o] * (n - np.arange(len(p)))))
    out = np.empty(len(p))
    out[o] = adj
    return out


def check_stats() -> None:
    s = T.stats(RV, TP, F, [0.2] * 19)
    m = s[s["kind"] == "main"]
    assert m["base"].tolist() == ["X1", "K2", "K3"] and set(m["hybrid"]) == {"W5-G"}
    for r in m.itertuples():                                                          # знак и база: Δ = 100 (Q_W5 / Q_base − 1); минус — W5-G лучше
        assert r.delta_pct == pytest.approx(100.0 * (ql(RV, F["W5-G"]).mean() / ql(RV, F[r.base]).mean() - 1.0), rel=1e-12)
    assert (m["n_holm"] == 798).all() and (m["n_holm_wide"] == 3301).all()
    assert m["p_holm_N"].to_numpy() == pytest.approx(holm(m["p_one"].to_numpy(), 798), rel=1e-9)
    assert m["p_holm_Nwide"].to_numpy() == pytest.approx(holm(m["p_one"].to_numpy(), 3301), rel=1e-9)
    for r in m.itertuples():                                                          # № 158: «подтверждено» только при Δ < 0 и p Холма < 0,05
        assert ("подтверждено (p Холма < 0,05)" in r.verdict) == (r.delta_pct < 0 and r.p_holm_N < 0.05)
        assert ("численно лучше базы" in r.verdict) == (r.delta_pct < 0)              # № 125/140: слово «лучше» ↔ знак Δ
    ref = s[s["kind"] == "reference"]
    assert ref["p_holm_N"].isna().all() and (ref["verdict"] == "справочно, критериев не получает").all()          # GK2, M, Ga, Gb критериев не получают


def test_stats_known_answer_and_precondition_is_not_vacuous() -> None:
    check_stats()
    m = T.stats(RV, TP, F, [0.2] * 19).query("kind == 'main'")
    k2 = m[m["base"] == "K2"].iloc[0]
    assert k2["delta_pct"] < 0 and k2["p_one"] < 0.05 <= k2["p_holm_N"]               # строка, на которой отличаются Холм и сырой p: «подтверждено» тут быть не должно
    assert "не подтверждено (p Холма ≥ 0,05)" in k2["verdict"]
    assert "подтверждено (p Холма < 0,05)" in m[m["base"] == "K3"].iloc[0]["verdict"]


def test_holm_n_equals_row_count_mutation_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(T, "N_HOLM", 3)
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(T, "N_HOLM", 798)
    monkeypatch.setattr(T, "N_WIDE", 12)
    with pytest.raises(AssertionError):
        check_stats()


def test_sign_base_and_wording_mutations_are_caught(monkeypatch) -> None:
    orig_ci, orig_v = EVX.delta_ci, SS.verdict
    monkeypatch.setattr(EVX, "delta_ci", lambda x, y, base, **k: {**orig_ci(x, y, base, **k), "delta_pct": -orig_ci(x, y, base, **k)["delta_pct"]})          # знак Δ
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(EVX, "delta_ci", lambda x, y, base, **k: orig_ci(x, y, float(x.mean()), **k))           # база процентов — QLIKE W5-G, а не базы
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(EVX, "delta_ci", orig_ci)
    monkeypatch.setattr(SS, "verdict", lambda d, p: orig_v(-d, p))                    # слова «лучше/хуже» перевёрнуты
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(SS, "verdict", lambda d, p: orig_v(d, 0.0))                   # «подтверждено» при любом p
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(SS, "verdict", orig_v)
    monkeypatch.setattr(EVX, "holm_both", lambda p, cp0: (np.asarray(p), np.asarray(p)))                       # Холм подменён сырым p
    with pytest.raises(AssertionError):
        check_stats()


def test_years_identity_known_answer_and_dropped_date_mutation(monkeypatch) -> None:
    F2 = dict(F, **{"W5-G": F["X1"] * np.where(TP.year == 2020, np.exp(-0.05), 1.0)})
    y = T.years(RV, TP, F2)
    assert T.identity_gap(y) < 1e-12
    w = y[(y["series"] == "W5-G") & (y["base"] == "X1") & (y["kind"] == "period")].set_index("period")
    lg, lx = ql(RV, F2["W5-G"]), ql(RV, F2["X1"])
    want = 100.0 * (lg - lx)[TP.year == 2020].sum() / (len(RV) * lx.mean())            # вклад года = 100 Σ(l_W5 − l_X1) / (n · Q_X1)
    assert w.at["2020", "contrib_pct"] == pytest.approx(want, rel=1e-9) and abs(want) > 0.1
    assert w.drop(index="2020")["contrib_pct"].abs().max() < 1e-12 or w.drop(index=["2020", "2026_до_27.02", "2026_F"])["contrib_pct"].abs().max() < 1e-12
    assert w["contrib_pct"].sum() == pytest.approx(w["delta_total_pct"].iloc[0], abs=1e-12)
    orig = T.masks

    def dropped(tp):                                                                  # мутация: одна дата выброшена из маски года
        mk = orig(tp)
        k = np.flatnonzero(mk["2019"])[3]
        mk["2019"] = mk["2019"].copy()
        mk["2019"][k] = False
        return mk
    monkeypatch.setattr(T, "masks", dropped)
    F3 = dict(F, **{"W5-G": F["X1"] * np.exp(np.random.default_rng(1).normal(0, 0.05, len(RV)))})
    assert T.identity_gap(T.years(RV, TP, F3)) > 1e-9


def tables(d1=0.0, d2=0.0, d18=0.0, qg=0.20, qa=0.21, qb=0.21, d_all=-0.1, loyo=-0.05) -> tuple:
    tbl = pd.DataFrame(dict(series=["W5-G", "W5-Ga", "W5-Gb"], qlike=[qg, qa, qb]))
    rows = [("2018", "period", d18, 0.1), ("2019", "period", -1.0, -0.3), (T.H1, "half", d1, np.nan), (T.H2, "half", d2, np.nan)]
    yrs = pd.DataFrame([dict(series="W5-G", base="X1", kind=k, period=p, delta_pct=dp, contrib_pct=c, delta_total_pct=d_all) for p, k, dp, c in rows])
    cnc = pd.DataFrame([dict(base="X1", item="loyo", key="2019", value=loyo)])
    return tbl, yrs, cnc


def passed(**kw) -> dict:
    g = T.guard(*tables(**kw), 0.05)
    return dict(zip(g["id"], g["passed"]))


def check_boundaries() -> None:
    assert all(passed(d1=0.415, d2=0.50, d18=0.26).values())                          # на границе — проходит (≤)
    assert passed(d1=0.4151)["У1"] is False and passed(d2=0.5001)["У2"] is False and passed(d18=0.2601)["У3"] is False
    assert passed(d1=0.4149)["У1"] and passed(d2=0.4999)["У2"] and passed(d18=0.2599)["У3"]
    assert passed(d1=0.30)["У1"] and passed(d18=0.30)["У3"] is False                 # 0,30 между 0,26 и 0,415: У3 не проходит, У1 проходит
    assert passed(d_all=0.0)["основной"] is False and passed(d_all=-1e-9)["основной"]  # строго меньше нуля
    assert passed(qg=0.21)["У4"] is False and passed(qa=0.19)["У4"] is False and passed(qb=0.19)["У4"] is False    # строго лучше обоих контролей
    assert passed(loyo=0.0)["У5"] and passed(loyo=1e-9)["У5"] is False                # Δ без лучшего периода ≤ 0


def test_guard_thresholds_are_registered_constants_and_boundaries_known_answer() -> None:
    assert T.THR == {"У1": 0.415, "У2": 0.50, "У3": 0.26}                              # 11.6/5: константы спецификации, не из данных
    assert "0,2525" in T.FORMULA["У1"] and "0,1567" in T.FORMULA["У3"] and "0,50" in T.FORMULA["У2"]
    check_boundaries()


def test_threshold_computed_from_result_mutation_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(T, "THR", {"У1": 0.60, "У2": 0.80, "У3": 0.40})                # мутация: пороги подогнаны под результат
    with pytest.raises(AssertionError):
        check_boundaries()
    assert T.THR != {"У1": 0.415, "У2": 0.50, "У3": 0.26}


def test_decision_is_derived_from_guard_table_and_names_failed_condition() -> None:
    ok = T.guard(*tables(), 0.05)
    assert T.decide(ok).startswith("новая контрольная точка") and "не «подтверждено»" in T.decide(ok)
    bad = T.guard(*tables(d18=0.30), 0.05)
    s = T.decide(bad)
    assert s.startswith("X1 остаётся контрольной точкой") and "У3 (+0.3000)" in s and "У1" not in s
    flipped = bad.copy()
    flipped.loc[flipped["id"] == "У3", "passed"] = True                               # подмена в таблице → решение меняется (--check ловит текст ≠ пересчёту, см. e2e)
    assert T.decide(flipped).startswith("новая контрольная точка")
    assert "справка" in set(ok["id"]) and set(ok["id"]) == {"основной", "У1", "У2", "У3", "У4", "У5", "справка"}     # GK2 и W5-Ga/Gb критериев не получают: условий по ним нет


def test_sensitivity_without_2022_02_24_and_without_most_influential_year_known_answer() -> None:
    gl = F["X1"] * np.where(TP.year == 2020, np.exp(-0.2), 1.0) * np.where(TP == pd.Timestamp("2022-02-24"), np.exp(-1.5), 1.0)
    F2 = dict(F, **{"W5-G": gl})
    s0 = T.sens(RV, TP, F2).set_index("variant")
    s = pd.DataFrame({k: s0[s0.index.str.startswith(k)].iloc[0] for k in ("все даты", "без пары", "без наиболее", "без 1 %")}).T      # варианты — по началу названия
    g, x = ql(RV, gl), ql(RV, F["X1"])
    d = lambda k: 100.0 * (g[k].sum() - x[k].sum()) / x[k].sum()
    allk = np.ones(len(RV), bool)
    no = TP != pd.Timestamp("2022-02-24")
    row = s.loc["без пары"]
    assert row["n"] == len(RV) - 1 and row["delta_pct"] == pytest.approx(d(no), rel=1e-12) and s.loc["все даты", "delta_pct"] == pytest.approx(d(allk), rel=1e-12)
    top = s.loc["без наиболее"]
    cy = {k: abs((g[TP.year == k] - x[TP.year == k]).sum()) for k in np.unique(TP.year)}
    best = max(cy, key=cy.get)
    assert best == 2020 and top["excluded"] == "2020" and top["delta_pct"] == pytest.approx(d(TP.year != 2020), rel=1e-12)
    trim = s.loc["без 1 %"]
    keep = np.ones(len(RV), bool)
    keep[np.argsort(-np.abs(g - x))[: int(np.ceil(0.01 * len(RV)))]] = False
    assert trim["n"] == keep.sum() and trim["delta_pct"] == pytest.approx(d(keep), rel=1e-12)
    assert s.loc["все даты", "n"] == len(RV)                     # основной результат — со всеми датами


def test_perm_p_known_answer_and_nineteen_shifts_use_walk_on_shifted_labels() -> None:
    assert T.perm_p(0.25, [0.1, 0.2, 0.3]) == (1 + 2) / 4 and T.perm_p(0.05, [0.1] * 19) == 1 / 20 and T.perm_p(0.5, [0.1] * 19) == 1.0
    n = 700
    rng = np.random.default_rng(2)
    d, tp = pd.bdate_range("2018-01-03", periods=n), pd.bdate_range("2018-01-04", periods=n)
    g = (rng.random(n) < 0.4).astype(int)
    x1 = np.exp(rng.normal(-8, 0.3, n))
    rv = x1 * np.exp(0.2 * g + rng.normal(0, 0.2, n))
    inp = dict(x=pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=x1)), gap=g, rv=rv)
    q = E.perm_q(inp)
    assert len(q) == 19
    for j in (0, 7, 18):
        w = G.walk(d, tp, rv, x1, np.roll(g, 252 + 100 * j))
        assert q[j] == float(Q.ql(rv, w["F"].to_numpy()).mean())                       # сдвиги 252 + 100·(j − 1), j = 1…19
    assert len(set(np.round(q, 12))) > 1
