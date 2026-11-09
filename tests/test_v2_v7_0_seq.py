"""V7-0, G4: PIT-построитель X_d. Синтетика с известным ответом, свойство «будущее не меняет X_d», мутации: бар t⁺ в окне, нормировка по всей выборке, окно +1 день, смена расписания без маски; реальные дни."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_seq as SQ

N_DAYS = 40


def make(n=N_DAYS, seed=0, miss=None, extra_day=None):
    """Синтетические бары: n рабочих дней, 53 слота; miss {день_индекс: [слоты]} — пропущенные бары; extra_day — день вне канона (бары есть, дня в `days` нет)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    allday = list(days) + ([pd.Timestamp(extra_day)] if extra_day else [])
    p = 100.0
    for di, d in enumerate(allday):
        for s in range(SQ.SLOTS):
            if miss and di in miss and s in miss[di]:
                continue
            o = p
            p = o * np.exp(rng.normal(0, 0.002))
            hi, lo = max(o, p) * 1.0005, min(o, p) * 0.9995
            v = int(rng.integers(100, 10000))
            rows.append((d + pd.Timedelta(minutes=600 + 10 * s), o, hi, lo, p, v, v * p))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume", "value"]), days


def perturb_after(bars, days, d, seed=1):
    """Копия баров, где все бары после конца дня d испорчены (цены и объёмы)."""
    b = bars.copy()
    m = b["begin"] >= pd.Timestamp(days[d]) + pd.Timedelta(days=1)
    rng = np.random.default_rng(seed)
    for c in ("open", "high", "low", "close"):
        b.loc[m, c] = b.loc[m, c] * rng.uniform(0.5, 2.0, m.sum())
    b.loc[m, "high"] = b.loc[m, ["open", "high", "low", "close"]].max(axis=1)
    b.loc[m, "low"] = b.loc[m, ["open", "high", "low", "close"]].min(axis=1)
    b.loc[m, "volume"] = b.loc[m, "volume"] * rng.integers(2, 50, m.sum())
    b.loc[m, "value"] = b.loc[m, "volume"] * b.loc[m, "close"]
    return b


def x_changes(bars, days, d, W) -> bool:
    """Меняется ли X_d от порчи ВСЕХ баров после дня d (True = утечка будущего)."""
    s1, s2 = SQ.build_store(bars, days), SQ.build_store(perturb_after(bars, days, d), days)
    a, b = SQ.x_of(s1, d, W), SQ.x_of(s2, d, W)
    return not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


def test_known_answer_returns_and_rv():
    days = pd.bdate_range("2020-01-01", periods=3)
    rows = []
    for d in days:
        for s in range(SQ.SLOTS):
            c = 100.0 * np.exp(0.001 * (s + 1))
            rows.append((d + pd.Timedelta(minutes=600 + 10 * s), 100.0, c * 1.001, 99.9, c, 1000, 1000 * c))
    f = SQ.bar_table(pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume", "value"]), days)
    assert np.allclose(f["r"], 0.1)                                    # ln-доходность 0,001 -> 0,1 %
    assert np.allclose(f.groupby("day")["r2"].sum(), 53 * 0.01)       # RV = 1e4·Σr² = 0,53
    assert np.allclose(f["rs_pos"] + f["rs_neg"], f["r2"])
    assert np.allclose(f["cum_rv"][f["slot"] == 52], 0.53) and f["tod"].iloc[0] == 0.0 and f["tod"].iloc[52] == 1.0


@pytest.mark.parametrize("W", [1, 5, 22])
def test_future_bars_do_not_change_x_d(W):
    bars, days = make()
    for d in (3, 12, 30):
        assert not x_changes(bars, days, d, W)


def test_truncated_store_gives_same_x_d():
    bars, days = make()
    full = SQ.build_store(bars, days)
    for d in (5, 21, 33):
        b, dd = SQ.truncate(bars, days, days[d])
        a, c = SQ.x_of(full, d, 5), SQ.x_of(SQ.build_store(b, dd), d, 5)
        assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])


def test_normalization_is_expanding_known_answer():
    bars, days = make()
    st = SQ.build_store(bars, days)
    f = SQ.bar_table(bars, days)
    d = 17
    past = f[f["day"] <= days[d]]
    mu, sd = SQ.norm_stats(st, d)
    for k, c in enumerate(SQ.CH):
        if SQ.ZCH[k]:
            x = past[c].fillna(0.0).to_numpy()
            assert abs(mu[k] - x.mean()) < 1e-9 and abs(sd[k] - x.std()) < 1e-9, c


def test_vol_rel_uses_strictly_past_same_slot_median():
    bars, days = make()
    f = SQ.bar_table(bars, days)
    st = SQ.build_store(bars, days)
    d, s = 10, 7
    v = f[(f["slot"] == s)].set_index("day")["volume"]
    want = v[days[d]] / np.median(v[v.index < days[d]].to_numpy())
    assert abs(st["raw"][d, s, SQ.CH.index("vol_rel")] - want) < 1e-12
    assert np.isnan(st["raw"][0, s, SQ.CH.index("vol_rel")])          # первая сессия: прошлого нет


def test_missing_bar_is_masked_and_zero():
    bars, days = make(miss={6: [52], 7: [0, 1]})
    st = SQ.build_store(bars, days)
    X, M, ok, _ = SQ.x_of(st, 7, 5)
    assert not M[-1, 0] and not M[-1, 1] and M[-1, 2] and not M[-2, 52]      # дни 7 и 6 в окне 5 дней
    assert (X[-1, 0] == 0).all() and (X[-2, 52] == 0).all() and np.isfinite(X).all()


def test_window_padding_before_history_is_masked():
    bars, days = make()
    st = SQ.build_store(bars, days)
    X, M, ok, nok = SQ.x_of(st, 2, 5)
    assert list(ok) == [False, False, True, True, True] and not M[:2].any() and not nok
    X22 = SQ.x_of(st, 30, 22)
    assert X22[0].shape == (22, SQ.SLOTS, SQ.K) and X22[2].all()           # окно 22 строится (только построитель)


def test_non_canonical_day_bars_are_excluded():
    bars, days = make(extra_day="2020-03-15")
    st = SQ.build_store(bars, days)
    assert st["present"].shape[0] == N_DAYS and st["present"].all()


# ---- мутации: каждая обязана ронять свойство «будущее не меняет X_d» или проверку маски ----
def test_mutation_window_shifted_plus_one_day_caught(monkeypatch):
    bars, days = make()
    assert not x_changes(bars, days, 12, 5)
    monkeypatch.setattr(SQ, "window_rows", lambda i, W: np.arange(i - W + 2, i + 2))
    assert x_changes(bars, days, 12, 5)


def test_mutation_normalization_on_whole_sample_caught(monkeypatch):
    bars, days = make()
    monkeypatch.setattr(SQ, "stats_row", lambda store, i: len(store["days"]) - 1)
    assert x_changes(bars, days, 12, 5)


def test_mutation_bar_of_t_plus_in_window_caught(monkeypatch):
    bars, days = make()
    orig = SQ.day_block

    def leaky(store, j):
        b = orig(store, j).copy()
        if j + 1 < len(store["days"]):
            b[SQ.SLOTS - 1] = store["raw"][j + 1][0]        # один бар дня t⁺ в последнем слоте окна
        return b
    monkeypatch.setattr(SQ, "day_block", leaky)
    assert x_changes(bars, days, 12, 1)


def test_mutation_schedule_change_without_mask_caught(monkeypatch):
    bars, days = make(miss={6: [52]})
    st = SQ.build_store(bars, days)

    def gap_masked() -> bool:        # свойство: на слоте без бара маска 0 и значения нули
        X, M, _, _ = SQ.x_of(st, 6, 1)
        return (not M[0, 52]) and bool((X[0, 52] == 0).all())
    assert gap_masked()
    monkeypatch.setattr(SQ, "bar_mask", lambda store, j: np.ones(SQ.SLOTS, bool))
    assert not gap_masked()                                                 # мутант (маска из единиц) нарушает свойство -> проверка его ловит


# ---- реальные дни ----
@pytest.fixture(scope="module")
def real():
    from models_v2.s14 import v7_0_pass as PA
    bars, d = PA.load()
    return bars, d.index


@pytest.mark.parametrize("i", [400, 1500, 2600])
def test_real_days_truncation_equals_full(real, i):
    bars, days = real
    full = SQ.build_store(bars, days)
    b, dd = SQ.truncate(bars, days, days[i])
    t = SQ.build_store(b, dd)
    for W in (1, 5):
        a, c = SQ.x_of(full, i, W), SQ.x_of(t, i, W)
        assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]) and a[3] == c[3]


def test_real_schedule_gap_day_has_masked_slots(real):
    bars, days = real
    st = SQ.build_store(bars, days)
    n = st["present"].sum(1)
    i = int(np.argmax(n < SQ.SLOTS))                                        # первый неполный день (2014)
    X, M, ok, _ = SQ.x_of(st, i, 1)
    assert M.sum() == n[i] < SQ.SLOTS and (X[0][~M[0]] == 0).all()
