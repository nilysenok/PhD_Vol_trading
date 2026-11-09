"""V6 лист Z3 (Ж-230): таблицы причин — известный ответ на синтетике, тождества, PIT, мутации (порог терцилей по всей выборке, выброшенная дата, перевёрнутый знак Δ). Сети и обучения нет."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s3 import dm as DM
from models_v2.s13 import v6_0_q as Q, v6_why_a as A, v6_why_b as B, v6_why_check as CK, v6_why_data as DT, v6_why_stage as ST, v6_why_stat as S


def synth(seed=0, last="2026-02-27"):
    dates = pd.bdate_range("2017-01-02", last)
    n, rng = len(dates), np.random.default_rng(seed)
    k1 = np.exp(rng.normal(0, 0.4, n))
    rv = k1 * np.exp(rng.normal(0, 0.5, n))
    f = {"K1": k1}
    for k in DT.SER:
        if k != "K1":
            f[k] = k1 * np.exp(np.clip(rng.normal(0, 0.15, n), -1, 1))
    return dict(dates=dates, tp=dates + pd.offsets.BDay(1), rv=rv, f=f)


def total(t, h="A2", b="K3"):
    y = t["WHY_YEARS"]
    return y[(y.hybrid == h) & (y.base == b) & (y.year == "total")].iloc[0]


def test_znak_i_izvestnyj_otvet_gibrid_ravno_rv():
    d = synth()
    d["f"]["A2"] = d["rv"].copy()                    # QLIKE гибрида = 0 -> Δ = −mean QLIKE(K3)
    t = ST.tables(d)
    assert abs(total(t).delta + Q.ql(d["rv"], d["f"]["K3"]).mean()) < 1e-12
    y = t["WHY_YEARS"]
    assert (y[(y.hybrid == "A2") & (y.base == "K3")].sign == -1).all()


def test_mutatsiya_znak_delta_lovitsya(monkeypatch):
    d = synth()
    d["f"]["A2"] = d["rv"].copy()
    monkeypatch.setattr(S, "dd", lambda rv, fh, fb: -(Q.ql(rv, fh) - Q.ql(rv, fb)))
    t = ST.tables(d)
    assert not abs(total(t).delta + Q.ql(d["rv"], d["f"]["K3"]).mean()) < 1e-12 and total(t).delta > 0     # известный ответ нарушен


def test_tozhdestva_chisto_i_pustoj_god():
    d = synth()
    t = ST.tables(d)
    assert CK.identities(t, len(d["dates"])) == []
    s = synth(last="2020-06-30")                                      # лет 2021+ нет: строки пропущены, тождество цело
    short = {"WHY_YEARS": A.years_table(s), "WHY_TAILS": A.tails_table(s), "WHY_REGIME": A.regime_table(s)}
    assert CK.identities(short, len(s["dates"])) == []


def test_mutatsiya_vybroshennaya_data_v_godah(monkeypatch):
    d = synth()
    orig = S.year_masks

    def drop(tp):
        m = orig(tp)
        m["2018"] = m["2018"].copy()
        m["2018"][np.flatnonzero(m["2018"])[0]] = False
        return m
    monkeypatch.setattr(S, "year_masks", drop)
    assert any("годы" in x for x in CK.identities(ST.tables(d), len(d["dates"])))


def test_mutatsiya_vybroshennaya_data_v_gruppah(monkeypatch):
    d = synth()
    orig = A.tercile_groups

    def drop(f):
        g = orig(f).astype(object)
        g[300] = "?"
        return g
    monkeypatch.setattr(A, "tercile_groups", drop)
    assert any("группы" in x for x in CK.identities(ST.tables(d), len(d["dates"])))


def test_tozhdestvo_topa_lovit_podmenu():
    t = ST.tables(synth())
    t["WHY_TAILS"].loc[0, "contrib_top"] += 1e-6
    assert any("топ-дни" in x for x in CK.identities(t, 1))


def test_pit_terciley_i_mutatsiya_porogi_po_vsej_vyborke(monkeypatch):
    d = synth()
    assert CK.pit(d) == []
    def whole(f, min_hist=A.MIN_HIST):                                 # мутация: порог по ВСЕЙ переданной выборке, не по датам ≤ d
        q = np.quantile(f, [1 / 3, 2 / 3])
        late = np.arange(len(f)) >= min_hist
        return np.where(late, q[0], np.nan), np.where(late, q[1], np.nan)
    monkeypatch.setattr(A, "thresholds", whole)
    assert any("зависят от данных после" in x for x in CK.pit(d))


def test_terciley_izvestnyj_otvet_monotonnyj_ryad():
    up = A.tercile_groups(np.arange(1.0, 701.0))
    assert (up[:250] == "no_group").all() and (up[250:] == "high").all()     # текущий максимум расширяющегося окна
    dn = A.tercile_groups(np.arange(700.0, 0.0, -1.0))
    assert (dn[250:] == "low").all()


def test_kraynie_dni_odin_vybros():
    d = synth()
    k3 = d["f"]["K3"]
    d["f"]["A2"] = k3.copy()
    d["f"]["A2"][500] = k3[500] * 3.0                                  # единственный день с l_гибрид ≠ l_K3
    z = A.tails_table(d)
    r = z[(z.hybrid == "A2") & (z.base == "K3") & (z.frac == 0.01)].iloc[0]
    assert r.n_top == round(0.01 * len(d["rv"])) and abs(r.share_top_pct - 100.0) < 1e-9 and abs(r.delta_without) < 1e-15


def test_hac_postoyannyj_regressor_ravno_nw_variance():
    y = np.random.default_rng(1).normal(size=400)
    _, se = S.hac_ols(y, np.ones((400, 1)))
    assert abs(se[0] ** 2 - DM.nw_variance(y)) < 1e-14
    beta, se2 = S.hac_ols(2 + 3 * y, np.column_stack([np.ones(400), y]))
    assert np.allclose(beta, [2, 3]) and (se2 < 1e-9).all()


def test_mz_izvestnyj_otvet_i_zaxvat_i_klip():
    d = synth()
    d["f"]["K3"] = d["rv"].copy()
    m = B.mz_table(d)
    r = m[(m.series == "K3") & (m.kind == "mz") & (m.window == "2017_2021")].iloc[0]
    assert abs(r.b - 1) < 1e-9 and abs(r.a) < 1e-9 and abs(r["shift"]) < 1e-12
    d = synth()
    k1, k3 = d["f"]["K1"], d["f"]["K3"]
    d["f"]["A2"] = k1 * (k3 / k1) ** 0.5
    d["f"]["E2"] = k1 * np.exp(1.0)
    c = B.capture_table(d)
    a = c[(c.base == "K3") & (c.hybrid == "A2") & (c.window == "all")].iloc[0]
    assert abs(a.slope - 0.5) < 1e-9 and abs(a["corr"] - 1) < 1e-9 and a.same_sign_pct == 100.0
    g = B.gsize_table(d)
    e = g[(g.series == "E2") & (g.window == "all")].iloc[0]
    assert e.at_clip_pct == 100.0 and abs(e.mean_abs - 1.0) < 1e-12


def test_vyrozhdenie_konstantnyj_k1_i_nulevoj_nakloh():
    d = synth()
    d["f"]["K3"] = d["f"]["K1"].copy()                                # x = 0: наклон не определён -> NaN, не ноль и не ошибка
    c = B.capture_table(d)
    assert c[(c.base == "K3")].slope.isna().all() and c[(c.base == "K3")]["corr"].isna().all()


def test_skolzyashchaya_i_raznost_okon():
    x, k = np.full(400, 0.02), np.full(400, 0.2)
    assert np.allclose(B.roll_means(x, k), 10.0) and len(B.roll_means(x, k)) == 151
    e = S.diff_ci(x, 0.2, x, 0.2)
    assert e["diff_pct"] == 0 and e["lo90"] == 0 and e["hi90"] == 0
    assert S.diff_ci(x, 0.2, x * 2, 0.2)["diff_pct"] == pytest.approx(10.0)
