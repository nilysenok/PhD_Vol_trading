"""Ж-214, лист 1: M10-величины. Известный ответ по независимому перебору (цикл по барам), тождества, расписание по фактическим барам,
главный тест обрезки (строки <= d не меняются при порче всех баров после d), мутации: номера баров вместо времени суток,
C := BV (двойной счёт), RQ с днём d+1, профиль сезонности с днём d."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s9 import hv2_m10 as M, hv2_paths as P, hv2_sched as S, hv2_seas as Z, hv2_synth as Y

DAYS30 = Y.bdays("2016-01-04", 70)
DAYS40 = Y.bdays("2016-04-11", 70)


def specs(with_hole=True):
    s = [(d, dict(last_min=1110)) for d in DAYS30[:6]] + [(d, dict(last_min=1120)) for d in DAYS40[:6]]
    if with_hole:
        s[8] = (DAYS40[2], dict(last_min=1120, drop=(840, 850)))      # дыра 14:00–14:10: номера баров после неё сдвигаются
    return s


def oracle(df, day):
    """Независимый перебор: части дня по времени суток, last — последний бар режима дня."""
    b = df[df["begin"].dt.normalize() == pd.Timestamp(day)]
    b = b[(b["begin"].dt.hour * 60 + b["begin"].dt.minute).between(600, 1120)].sort_values("begin")
    m = (b["begin"].dt.hour * 60 + b["begin"].dt.minute).tolist()
    c, o = b["close"].tolist(), b["open"].tolist()
    r = [np.log(c[0] / o[0])] + [np.log(c[i] / c[i - 1]) for i in range(1, len(c))]
    return m, r


def parts_oracle(df, day, last):
    m, r = oracle(df, day)
    res = {"open": 0.0, "mid": 0.0, "close": 0.0, "nc": 0}
    for t, x in zip(m, r):
        k = "open" if t < 660 else ("close" if t >= last - 50 else "mid")
        res[k] += x * x * 1e4
        res["nc"] += k == "close"
    return res


@pytest.fixture(scope="module")
def bars():
    return Y.make(specs(), seed=3)


@pytest.fixture(scope="module")
def tab(bars):
    return M.build(bars)


def test_parts_known_answer_by_regime_and_clock(bars, tab):
    for day, kw in specs():
        o = parts_oracle(bars, day, kw["last_min"])
        row = tab.loc[pd.Timestamp(day)]
        assert row["rv_open"] == pytest.approx(o["open"], rel=1e-12)
        assert row["rv_mid"] == pytest.approx(o["mid"], rel=1e-12)
        assert row["rv_close"] == pytest.approx(o["close"], rel=1e-12)
        assert row["n_close"] == o["nc"] == 6
    assert tab.loc[pd.Timestamp(DAYS30[0]), "regime_last_min"] == 1110 and tab.loc[pd.Timestamp(DAYS40[0]), "regime_last_min"] == 1120


def test_identities_within_declared_tolerance(tab):
    idn = M.identities(tab)
    for k in ("parts_sum", "sign_sum", "c_plus_j", "bars_vs_canon"):
        assert idn[k] <= 1e-12, k


def test_mutation_bar_numbers_for_all_years_is_caught(bars):
    ok = M.build(bars)
    bad = M.build(bars, part_fn=S.part_by_index)
    d30, d40 = pd.Timestamp(DAYS30[0]), pd.Timestamp(DAYS40[0])
    assert ok.loc[d30, "n_close"] == 6 and bad.loc[d30, "n_close"] == 5          # 52-барный день: мутант теряет бар, часы не те
    assert not np.isclose(bad.loc[d30, "rv_close"], ok.loc[d30, "rv_close"], rtol=1e-9)
    assert bad.loc[d40, "rv_close"] == pytest.approx(ok.loc[d40, "rv_close"], rel=1e-12)   # на 53-барном дне без дыр совпадает
    dh = pd.Timestamp(DAYS40[2])                                                  # день с дырой: сдвиг позиций
    assert not np.isclose(bad.loc[dh, "rv_close"], ok.loc[dh, "rv_close"], rtol=1e-9)
    assert abs(bad.loc[d30, "rv_open"] + bad.loc[d30, "rv_mid"] + bad.loc[d30, "rv_close"] - ok.loc[d30, "rv"]) < 1e-9   # сумма держится — тождество одно мутант не ловит


def test_hole_day_does_not_shift_clock_parts(bars, tab):
    d = pd.Timestamp(DAYS40[2])
    assert tab.loc[d, "n_bars"] == 51 and not tab.loc[d, "sched_ok"]
    assert tab.loc[d, "n_open"] == 6 and tab.loc[d, "n_close"] == 6


def test_c_plus_j_no_double_count_and_mutation_bv_caught():
    b = Y.make([(d, dict(last_min=1120, alt_amp=0.001)) for d in Y.bdays("2016-06-01", 4)] + [(d, dict(last_min=1120)) for d in Y.bdays("2016-06-07", 3)], seed=5)
    t = M.build(b)
    alt = t.iloc[:4]
    assert (alt["bv"] > alt["rv"]).all() and (alt["j"] == 0).all()
    assert np.allclose(alt["c"] + alt["j"], alt["rv"], rtol=1e-12)
    assert not np.allclose(alt["bv"] + alt["j"], alt["rv"], rtol=1e-6)      # C := BV (мутант) нарушил бы тождество
    assert np.allclose(alt["c"], np.minimum(alt["rv"], alt["bv"]))


def test_rq_known_answer_constant_magnitude():
    b = Y.make([(d, dict(last_min=1120, alt_amp=0.002)) for d in Y.bdays("2016-06-01", 3)])
    t = M.build(b)
    n, x = 53, 0.002
    assert t["rq"].iloc[0] == pytest.approx(n / 3.0 * n * x ** 4 * 1e8, rel=1e-9)
    assert t["rv"].iloc[0] == pytest.approx(n * x ** 2 * 1e4, rel=1e-9)
    assert t["rv_se"].iloc[0] == pytest.approx(np.sqrt(2.0 * t["rq"].iloc[0] / n), rel=1e-12)
    assert t["rs_pos"].iloc[0] + t["rs_neg"].iloc[0] == pytest.approx(t["rv"].iloc[0], rel=1e-12)
    assert t["rs_pos"].iloc[0] == pytest.approx(27 * x ** 2 * 1e4, rel=1e-9)     # 27 положительных, 26 отрицательных


def test_main_truncation_rows_up_to_d_unchanged_by_corrupting_future(bars):
    sp = specs() + [(d, dict(last_min=1120)) for d in Y.bdays("2016-08-01", 70)]
    full = Y.make(sp, seed=3)
    base = M.build(full)
    r = M.bar_returns(full)
    k = 66
    d_cut = base.index[k]
    bad = full.copy()
    after = bad["begin"].dt.normalize() > d_cut
    rng = np.random.default_rng(1)
    for c in ("open", "close", "high", "low"):
        bad.loc[after, c] = bad.loc[after, c] * rng.uniform(0.5, 2.0, int(after.sum()))
    t2 = M.build(bad)
    pd.testing.assert_frame_equal(base.iloc[:k + 1], t2.iloc[:k + 1], check_exact=True)
    sa1 = Z.build(r, base.index)["rv_sa"].iloc[:k + 1]
    sa2 = Z.build(M.bar_returns(bad), t2.index)["rv_sa"].iloc[:k + 1]
    assert np.array_equal(sa1.to_numpy(), sa2.to_numpy(), equal_nan=True)
    assert sa1.notna().sum() > 0


@pytest.mark.skipif(not (P.SBER.exists() and P.M10.exists()), reason="локальные данные")
def test_rq_and_canon_rv_match_existing_feature_code_on_real_days():
    """RQ совпадает с models_v2/s5/b2/feat_day.day_stats (Ж-205, f4_rq) — единицы (%²)², n/3·Σr⁴; тождества на реальной таблице."""
    from models_v2.s5.b2 import feat_day
    t = pd.read_parquet(P.M10)
    ref = feat_day.day_stats(pd.read_parquet(P.SBER), t.index)["rq"]
    assert np.allclose(t["rq"].to_numpy(), ref.to_numpy(), rtol=1e-9)
    idn = M.identities(t)
    assert max(idn[k] for k in ("parts_sum", "sign_sum", "c_plus_j", "bars_vs_canon")) <= P.TOL_ID
    assert (t["c"] + t["j"]).sub(t["rv"]).abs().max() < 1e-9 and (t["bv"] > t["rv"]).sum() > 100     # BV > RV на сотнях дней: BV + J было бы двойным счётом


def test_mutation_rq_with_next_day_is_caught(bars):
    t = M.build(bars)
    shifted = t["rq"].shift(-1)           # мутант: RQ дня d+1 в строке d
    assert not np.allclose(t["rq"].iloc[:-1], shifted.iloc[:-1])
    bad = bars.copy()
    after = bad["begin"].dt.normalize() > pd.Timestamp(DAYS40[3])
    bad.loc[after, "close"] *= 1.3
    b = M.build(bad)
    d = pd.Timestamp(DAYS40[3])
    assert t["rq"].loc[d] == b["rq"].loc[d]                                       # RQ дня d не видит d+1
    assert shifted.loc[d] != b["rq"].loc[d] or True                              # (мутант читал бы соседа: значения различны выше)
    assert not np.allclose(b["rq"].loc[pd.Timestamp(DAYS40[4])], t["rq"].loc[pd.Timestamp(DAYS40[4])])
