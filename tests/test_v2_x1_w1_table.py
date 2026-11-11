"""W1.2–W1.3: окно M и определение QLIKE (среднее ДНЕВНЫХ потерь; мутация № 160), общие даты; сверка окон F и H с FWD_SERIES / FWD_FAMILY / P0_TABLE (1e-12). Сеть не нужна."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_t0 as T0, x1_w1 as W

NAMES = ("K1", "K2", "K3", "A2", "NTP2", "cfg6", "cfg8", "X1", "ZB", "ZB0")
SHA = ({k: "a" for k in NAMES}, {k: "n" for k in NAMES})


def synth(f1: float = 4.0) -> dict:
    """10 рядов на рабочих днях 2018-01-03…2026-02-26 (t⁺ до 2026-02-27); K1: F = 2·RV в 2018 (по t⁺), далее f1·RV — потери по годам разные при разном n лет."""
    d = pd.bdate_range("2018-01-03", "2026-02-26")
    tp = pd.DatetimeIndex(list(d[1:]) + [pd.Timestamp("2026-02-27")])
    rng = np.random.default_rng(0)
    rv = np.exp(rng.normal(0, 0.3, len(d)))
    fr = {}
    for k in NAMES:
        F = rv * np.where(tp.year == 2018, 2.0, f1) if k == "K1" else rv * np.exp(rng.normal(0.1, 0.2, len(d)))
        fr[k] = pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, F=F, part="archive"))
    return fr


def l_of(r: float) -> float:
    return r - np.log(r) - 1.0


def table(fr: dict) -> pd.DataFrame:
    return W.table({"M": T0.common({k: fr[k] for k in ("K1", "K2", "K3", "X1")})}, SHA)


def want_q1(fr: dict) -> float:
    tp = pd.DatetimeIndex(fr["K1"]["t_plus"])
    n18 = int((tp.year == 2018).sum())
    return (n18 * l_of(0.5) + (len(tp) - n18) * l_of(0.25)) / len(tp)


def test_known_answer_daily_losses_not_mean_of_periods() -> None:
    fr = synth()
    t = table(fr).set_index("series")
    assert t.at["K1", "qlike"] == pytest.approx(want_q1(fr), abs=1e-12)
    assert t.at["K2", "d_k2_pct"] == 0.0 and t.at["K2", "d_k3_pct"] == pytest.approx(100 * (t.at["K2", "qlike"] - t.at["K3", "qlike"]) / t.at["K3", "qlike"], abs=1e-12)
    assert t.at["X1", "d_x1_pct"] == 0.0 and t.at["K1", "d_x1_pct"] > 0                      # Δ в % QLIKE базы, знак: плюс — хуже базы


def test_mutation_mean_of_period_scores_is_caught_no160(monkeypatch) -> None:
    fr = synth()
    assert table(fr).set_index("series").at["K1", "qlike"] == pytest.approx(want_q1(fr), abs=1e-12)

    def mutant(rv, f):                                                                       # среднее показателей периодов (годов)
        tp = pd.DatetimeIndex(fr["K1"]["t_plus"])
        q = pd.Series(np.asarray(rv)[:len(tp)] / np.asarray(f)[:len(tp)]).pipe(lambda r: r - np.log(r) - 1.0)
        return float(q.groupby(tp.year.to_numpy()).mean().mean())
    monkeypatch.setattr(W, "window_qlike", mutant)
    assert abs(table(fr).set_index("series").at["K1", "qlike"] - want_q1(fr)) > 1e-3


def test_windows_common_dates_and_mutation_not_common_dates_is_caught(monkeypatch) -> None:
    fr = synth()
    fr["X1"] = fr["X1"].iloc[100:].reset_index(drop=True)                                      # ряд позже начинается
    fr["K2"] = fr["K2"].drop(index=500).reset_index(drop=True)                                 # дыра
    w = W.windows(fr)["M"]
    want = pd.DatetimeIndex(fr["X1"]["d"]).difference([synth()["K1"]["d"].iloc[500]])
    assert all(pd.DatetimeIndex(v["d"]).equals(want) for v in w.values())
    t = table(fr)
    assert set(t["n"]) == {len(want)} and set(t["first_tplus"]) == {str(w["K1"]["t_plus"].min().date())}
    monkeypatch.setattr(T0, "common", lambda f: {k: v.copy() for k, v in f.items()})           # мутант: каждый ряд на своих датах
    with pytest.raises(SystemExit):
        table(fr)


def test_series_not_covering_core_gets_own_window_and_does_not_cut_main() -> None:
    fr = synth()
    fr["ZB"] = fr["ZB"].iloc[300:].reset_index(drop=True)                                      # ZB начинается позже
    w = W.windows(fr)
    assert "ZB" not in w["M"] and "ZB" in w["M|ZB"] and len(w["M"]["K1"]) == len(fr["K1"])    # главное окно не сокращено ради ZB


@pytest.fixture(scope="module")
def real():
    from models_v2.s16 import x1_w1_join as J
    fr, jn, sha = J.load_joined()
    return W.windows(fr), W.table(W.windows(fr), sha)


def test_real_mcs_equals_library_on_window_m_and_window_2017_is_not_empty(real) -> None:
    """Библиотечный `mcs90(..., "all")` режет окно до 2018–2026: на окне 2017 он пуст (NaN, p = 1) — W1 считает MCS без этого окна и обязан совпасть с библиотекой там, где она определена."""
    wins, t = real
    al = wins["M"]
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"]).to_numpy()
    lib = T0.ST.mcs90(rv, {k: v["F"].to_numpy(float) for k, v in al.items()}, tp, "all").set_index("series")
    mine = t[t["window"] == "M"].set_index("series")
    assert (mine["mcs90"] == lib["in_mcs"]).all() and np.allclose(mine["mcs_p"], lib["p"], atol=1e-12)
    y17 = t[t["window"] == "2017"]
    assert (y17["qlike"] > 0).all() and not y17["mcs_p"].isna().any() and len(set(y17["mcs_p"])) >= 1
    a17 = wins["2017"]
    assert np.isfinite(W.mcs90(a17["K1"]["rv"].to_numpy(float), {k: v["F"].to_numpy(float) for k, v in a17.items()})["mean_qlike"]).all()


def test_real_window_m_bounds_and_n(real) -> None:
    wins, t = real
    m = t[t["window"] == "M"]
    assert set(m["first_tplus"]) == {"2018-01-04"} and set(m["last_tplus"]) == {"2026-11-10"} and set(m["series"]) == {"K1", "K2", "K3", "X1", "ZB", "ZB0", "cfg8"}
    h, f = t[t["window"] == "H"], t[t["window"] == "F"]
    assert set(m["n"]) == {h["n"].iloc[0] + f["n"].iloc[0]} and not any(k.startswith("M|") for k in wins)


def test_real_window_f_equals_fwd_series_and_family_1e12(real) -> None:
    wins, t = real
    fs = pd.read_csv(T0.REP.parent / "harj_fwd/FWD_SERIES.csv").set_index("series")
    f = t[t["window"] == "F"].set_index("series")
    for k in ("K1", "K2", "K3", "X1"):
        assert f.at[k, "qlike"] == pytest.approx(fs.at[k, "qlike"], abs=1e-12) and f.at[k, "n"] == fs.at[k, "n"]
    from models_v2.s16 import x1_w1_stat as SS
    st = SS.stats_table(wins).query("window == 'F'").set_index(["hybrid", "base"])
    fam = pd.read_csv(T0.REP.parent / "harj_fwd/FWD_FAMILY.csv")
    al = wins["F"]
    rv = al["K1"]["rv"].to_numpy(float)
    for r in fam.itertuples():
        w = st.loc[(r.A, r.B)]
        for mine, theirs in ((w.qlike_h, r.qlike_A), (w.qlike_b, r.qlike_B), (w.delta_pct, r.delta_pct), (w.dm_stat, r.dm_stat), (w.p_one, r.p_one)):
            assert mine == pytest.approx(theirs, abs=1e-12)
        qa, qb = Q.ql(rv, al[r.A]["F"].to_numpy(float)), Q.ql(rv, al[r.B]["F"].to_numpy(float))
        ci10 = T0.EVX.delta_ci(qa, qb, float(qb.mean()), b=2000, block=10, seed=0)       # FWD считал интервал с блоком 10; W1 — блок 20 (как T0, задание): интервалы разные по определению
        assert ci10["lo90"] == pytest.approx(r.lo90_pct, abs=1e-12) and ci10["hi90"] == pytest.approx(r.hi90_pct, abs=1e-12)
        assert (w.lo90, w.hi90) != (r.lo90_pct, r.hi90_pct)


def test_real_window_h_equals_p0_table_1e12(real) -> None:
    wins, t = real
    p0 = pd.read_csv(T0.REP / "P0_TABLE.csv", comment="#").set_index("series")
    h = t[t["window"] == "H"].set_index("series")
    alias = {"V6:A2": "A2", "V4:NTP2": "NTP2", "V7:cfg6": "cfg6", "V7:cfg8": "cfg8"}
    for k in p0.index:
        assert h.at[alias.get(k, k), "qlike"] == pytest.approx(p0.at[k, "q_all"], abs=1e-12)
    assert set(h["n"]) == {int(p0["n_all"].iloc[0])}
