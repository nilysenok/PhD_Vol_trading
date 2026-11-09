"""FIN: сводные таблицы — известные ответы на синтетике, отказ на «дырах», сверка с таблицами листов (подмена числа ловится)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de_fin_leaf as LEAF, de_fin_tabs as T

Q = T.Q


def synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(sum((list(pd.bdate_range("%d-01-03" % y, periods=70)) for y in range(2018, 2027)), []))
    tp, n = idx + pd.Timedelta(days=1), len(idx)
    lv = 0.3 * np.sin(np.arange(n) / 15)
    rv = np.exp(lv + rng.normal(0, 0.5, n))
    f = lambda s: np.exp(lv + rng.normal(0, s, n))
    ser = {k: f(0.4) for k in T.CTRL}
    ser["K3"] = ser["K2"] * 3.0                                  # K3 заведомо хуже K2: «лучший бустинг» — K2
    x = {k: f(0.3) for k in ("X1", "X2", "X3", "X4", "ENS")}
    return idx, tp, rv, ser, x, rng.integers(0, 2, n)


@pytest.fixture(scope="module")
def world():
    idx, tp, rv, ser, x, state = synth()
    s = T.assemble(ser, x)
    return dict(idx=idx, tp=tp, rv=rv, ser=ser, x=x, state=state, s=s, t=T.all_tables(rv, s, tp, state))


def test_assemble_refuses_zero_forecast_and_unequal_length(world):
    bad = dict(world["x"], X1=np.zeros(len(world["rv"])))        # нулевой прогноз
    with pytest.raises(SystemExit, match="одинаковых дат"):
        T.assemble(world["ser"], bad)
    with pytest.raises(SystemExit, match="одинаковых дат"):
        T.assemble(world["ser"], dict(world["x"], X2=world["x"]["X2"][:-1]))      # дыра: ряд короче
    nan = world["x"]["X3"].copy()
    nan[5] = np.nan
    with pytest.raises(SystemExit, match="одинаковых дат"):
        T.assemble(world["ser"], dict(world["x"], X3=nan))


def test_tables_cover_series_years_windows_and_configs(world):
    t = world["t"]
    m = t["METRICS"]
    assert set(m["series"]) == set(T.ALL)
    assert {str(y) for y in range(2018, 2026)} | {"2018_2021", "2019_2021", "2022_2025", "2026", "all"} <= set(m["window"])
    assert set(t["TESTS"]["config"]) == set(T.X) and set(t["TESTS"]["window"]) == {"2018_2021", "2019_2021", "2022_2025", "2026", "all"}
    assert set(t["MCS"]["window"]) == {"2018_2021", "2022_2025", "all"} and set(t["GO"]["config"]) == set(T.X)
    assert {"X1 против NTP2 (6.2, польза состояния)", "X4 против ансамбля без затвора (6.2, польза затвора)"} <= set(t["PAIRS"]["hyp"])
    p = t["PAIRS"]
    assert (p["p_holm_N"] >= p["p_one"] - 1e-12).all()           # Холм не уменьшает p


def test_metrics_qlike_equals_direct_known_answer(world):
    m = world["t"]["METRICS"].set_index(["series", "window"])
    mask = np.asarray(world["tp"].year == 2019)
    direct = float(Q.ql(world["rv"][mask], world["s"]["X2"][mask]).mean())
    assert m.loc[("X2", "2019"), "qlike"] == pytest.approx(direct, rel=1e-12)
    assert m.loc[("X2", "2019"), "n"] == mask.sum()


def test_delta_sign_perfect_forecast_negative_and_inflated_positive(world):
    for F, sign in ((world["rv"].copy(), -1), (world["s"]["K1"] * 100.0, 1)):
        s = dict(world["s"], X1=F)
        t = T.main_tables(world["rv"], s, world["tp"])["TESTS"]
        r = t[(t["config"] == "X1") & (t["base"] == "K1") & (t["window"] == "2018_2021")].iloc[0]
        assert np.sign(r["delta_pct"]) == sign                     # перепутанный знак Δ упал бы здесь


def test_stab_known_answers_correction_under_over_and_top1(world):
    s = dict(world["s"])
    s["X1"] = s["K1"] * np.exp(0.1)                                # поправка к HAR-J ровно +0,1
    s["X2"] = s["K2"].copy()
    d23 = np.flatnonzero(np.asarray(world["tp"].year == 2023))[3]
    s["X2"][d23] *= 50.0                                           # один плохой день — вся разница к лучшему бустингу (K2) от него
    st = T.stab_table(world["rv"], s, world["tp"]).set_index(["series", "period"])
    r = st.loc[("X1", "2024")]
    assert r["mean_corr"] == pytest.approx(0.1) and r["share_corr_pos"] == 1.0 and r["share_corr_neg"] == 0.0
    assert r["share_under"] + r["share_over"] == pytest.approx(1.0)
    assert r["qlike_part_under"] + r["qlike_part_over"] == pytest.approx(r["qlike"])
    x2 = st.loc[("X2", "2023")]
    assert x2["best_boost"] == "K2" and x2["share_top1"] == pytest.approx(1.0) and x2["delta_total"] > 0
    assert st.loc[("X2", "2024"), "delta_total"] == 0.0            # вне плохого дня разницы нет


def test_adv_table_is_qlike_k2_minus_series_and_identity(world):
    t = world["t"]
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    a = t["ADV"].set_index("series")
    assert a.loc["X1", "adv_2018_2021"] == pytest.approx(m[("K2", "2018_2021")] - m[("X1", "2018_2021")])
    assert (a["d_adv"] - (a["adv_2022_2025"] - a["adv_2018_2021"])).abs().max() < 1e-12


def test_leaf_diff_catches_substituted_number_and_missing_row():
    a = pd.DataFrame(dict(window=["w1", "w2"], base=["b", "b"], delta_pct=[1.0, 2.0], rule=["да", "нет"]))
    assert LEAF._diff(a, a.copy(), ["window", "base"], "t") == []
    b = a.copy()
    b.loc[1, "delta_pct"] = 2.5
    assert any("delta_pct" in x for x in LEAF._diff(a, b, ["window", "base"], "t"))
    c = pd.concat([a, pd.DataFrame(dict(window=["w3"], base=["b"], delta_pct=[3.0], rule=["да"]))])
    assert any("нет строк" in x for x in LEAF._diff(a, c, ["window", "base"], "t"))
    d = a.copy()
    d.loc[0, "rule"] = "нет"
    assert any("rule" in x for x in LEAF._diff(a, d, ["window", "base"], "t"))
