"""W8-B, чувствительность, вклады «год/половина/окно × тип перерыва», путь β и MCS на синтетике (W8_SPEC 5, 9; mutations.md № 125, 140, 142): Σ вкладов по типам = Δ периода, Σ по годам = Δ окна (1e-12), выброшенный тип
ломает тождество; чувствительность без пары t⁺ 2022-02-24 и без самого влиятельного года — известный ответ (только показ, β не переоценивается); флаги включения групп: n ≥ 30 ⇒ 1, n = 29 ⇒ 0.
Только monkeypatch (№ 43). Реальные файлы только читаются (autouse-хэши)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_w8_eval as E
from models_v2.s16 import x1_w8_eval_t as T


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (E.REG.TRIALS, E.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.OUT.rglob("*")) if p.is_file()} if E.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


def synth(n: int = 1800) -> tuple:
    rng = np.random.default_rng(11)
    tp = pd.bdate_range("2018-01-04", periods=n)
    L = np.exp(rng.normal(-8, 0.4, n))
    rv = L * np.exp(rng.normal(0.0, 0.5, n) + 0.1)
    gap = (np.arange(n) % 5 == 4).astype(int)
    typ = np.where(gap == 0, "none", np.array(["weekend", "hol1", "long", "worksat", "other"])[np.arange(n) // 5 % 5])
    F = {"X1": L, "K1": L * 1.4, "K2": L * 1.8, "K3": L * 2.0, "W5-G": L * np.exp(0.05 * gap), "W8-G": L * np.exp(0.02 + 0.06 * gap * (np.arange(n) % 3 == 0)), "M": L * 1.7, "ZB": L, "W8-Gb": L}
    x = pd.DataFrame(dict(d=tp - pd.Timedelta(days=1), t_plus=tp, rv=rv, F=L, gap=gap, type=typ, yr=tp.year.astype(str), half=np.where(tp.year <= 2021, "h1", "h2"), u=np.log(rv / L)))
    for k, c in (("W5G", "W5-G"), ("X1", "X1"), ("K2", "K2"), ("K3", "K3")):
        x["l_" + k] = Q.ql(rv, F[c])
    return rv, tp, F, x


RV, TP, F, X = synth()


def ql(f):
    r = RV / f
    return r - np.log(r) - 1.0


def test_sensitivity_without_2022_02_24_and_without_most_influential_year_known_answer() -> None:
    gl = F["X1"] * np.where(TP.year == 2020, np.exp(-0.2), 1.0) * np.where(TP == pd.Timestamp("2022-02-24"), np.exp(-1.5), 1.0)
    s = T.sens(RV, TP, dict(F, **{"W8-G": gl}))
    assert (s["kind"] == "sens").all() and set(s["base"]) == {"W5-G", "X1"} and (s["verdict"] == "показ, не условие").all()
    for b in ("W5-G", "X1"):
        v = s[s["base"] == b].set_index("variant")
        g, x = ql(gl), ql(F[b])
        d = lambda k: 100.0 * (g[k].sum() - x[k].sum()) / x[k].sum()                  # noqa: E731
        allk, y = np.ones(len(RV), bool), TP.year.to_numpy()
        pick = lambda p: v[v.index.str.startswith(p)].iloc[0]                         # noqa: E731
        assert pick("все даты")["delta_pct"] == pytest.approx(d(allk), rel=1e-12) and pick("все даты")["n"] == len(RV)
        assert pick("без пары")["n"] == len(RV) - 1 and pick("без пары")["delta_pct"] == pytest.approx(d(TP != pd.Timestamp("2022-02-24")), rel=1e-12)
        assert pick("h1")["delta_pct"] == pytest.approx(d(y <= 2021), rel=1e-12) and pick("h2 (")["delta_pct"] == pytest.approx(d(y >= 2022), rel=1e-12)
        assert pick("без 2022")["delta_pct"] == pytest.approx(d(y != 2022), rel=1e-12) and pick("h2 без 2022")["delta_pct"] == pytest.approx(d(y >= 2023), rel=1e-12)
        cy = {k: abs((g[y == k] - x[y == k]).sum()) for k in np.unique(y)}
        best = max(cy, key=cy.get)
        assert pick("без наиболее влиятельного")["delta_pct"] == pytest.approx(d(y != best), rel=1e-12) and ("(%d)" % best) in pick("без наиболее влиятельного").name


def test_cells_known_answer_identity_and_dropped_type_mutation(monkeypatch) -> None:
    c, ident = T.cells(X, F["W8-G"])
    assert ident < 1e-12 and set(c["base"]) == {"W5-G", "X1", "K2", "K3"} and set(c["type"]) == set(T.TYPES) | {"all_gap", "all"}
    r = c[(c["base"] == "X1") & (c["period"] == "2020") & (c["type"] == "hol1")].iloc[0]
    l8, lx = Q.ql(RV, F["W8-G"]), ql(F["X1"])
    m = (TP.year == 2020) & (X["type"] == "hol1").to_numpy()
    p = TP.year == 2020
    assert r["n"] == m.sum() and r["Q_W8G"] == pytest.approx(l8[m].mean(), rel=1e-12) and r["Q_base"] == pytest.approx(lx[m].mean(), rel=1e-12)
    assert r["contrib_pp_period"] == pytest.approx(100 * (l8[m] - lx[m]).sum() / (p.sum() * lx[p].mean()), rel=1e-12)      # вклад в п. п. QLIKE базы периода
    assert r["contrib_pp_M"] == pytest.approx(100 * (l8[m] - lx[m]).sum() / (len(RV) * lx.mean()), rel=1e-12)             # и окна M
    a = c[(c["base"] == "K2") & (c["type"] == "all")].set_index("period")
    assert a.at["M", "delta_pct"] == pytest.approx(100 * (l8.mean() / ql(F["K2"]).mean() - 1), rel=1e-12) and a.at["h1", "n"] + a.at["h2", "n"] == len(RV)
    monkeypatch.setattr(T, "TYPES", T.TYPES[:3])                                       # мутация: типы long/worksat/other выброшены из вкладов
    assert T.cells(X, F["W8-G"])[1] > 1e-6                                             # тождество «Σ по типам = Δ периода» нарушено — check это ловит


def test_cells_sign_and_base_mutation_changes_delta_but_not_identity(monkeypatch) -> None:
    good = T.cells(X, F["W8-G"])[0].set_index(["base", "period", "type"])["delta_pct"]
    assert good[("X1", "M", "all")] == pytest.approx(100 * (Q.ql(RV, F["W8-G"]).mean() / ql(F["X1"]).mean() - 1), rel=1e-12)
    swapped = X.rename(columns={"l_X1": "l_W5G", "l_W5G": "l_X1"})                      # мутация: перепутаны базы X1 и W5-G
    bad = T.cells(swapped, F["W8-G"])[0].set_index(["base", "period", "type"])["delta_pct"]
    assert not np.isclose(bad[("X1", "M", "all")], good[("X1", "M", "all")], rtol=1e-9)


def test_beta_table_flags_inclusion_at_thirty_pairs_and_formats() -> None:
    n = pd.Series([29, 30, 31, 0])
    w = pd.DataFrame(dict(refit_date=pd.to_datetime(["2018-07-02", "2018-08-01", "2018-08-01", "2018-09-03"]), beta=[0.1, 0.1, 0.1, 0.2],
                          b_T1=[0.1, 0.2, 0.2, 0.3], b_T2=[0.1] * 4, b_T3=[0.0, 0.5, 0.5, 0.1], n_T1=n, n_T2=[100] * 4, n_T3=[0, 29, 30, 5]))
    t = T.beta_table(w)                                                                # по одной строке на месяц (дубль refit_date отброшен)
    assert t["refit_date"].tolist() == ["2018-07-02", "2018-08-01", "2018-09-03"] and t["beta_W5G"].tolist() == [0.1, 0.1, 0.2]
    assert t["incl_T1"].tolist() == [0, 1, 0] and t["incl_T2"].tolist() == [1, 1, 1] and t["incl_T3"].tolist() == [0, 0, 0]       # строка 2018-08-01 (первая): n_T3 = 29 → не включена
    assert (t.loc[t["n_T1"] == 29, "incl_T1"] == 0).all() and T.MIN_EXT == 30


def test_mcs_has_seven_series_and_worst_is_out() -> None:
    m = T.mcs(RV, F)
    assert set(m["series"]) == {"W8-G", "W5-G", "X1", "ZB", "K2", "K3", "M"} and {"mean_qlike", "p", "in_mcs"} <= set(m.columns)
    q = m.set_index("series")
    assert not bool(q.at["K3", "in_mcs"]) and bool(q["mean_qlike"].idxmin() in set(q.index[q["in_mcs"].astype(bool)]))      # самый худший вне, самый лучший внутри
    assert q.at["K3", "mean_qlike"] == pytest.approx(ql(F["K3"]).mean(), rel=1e-12)
