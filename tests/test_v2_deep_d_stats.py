"""G5 D: статистика — блочный бутстрап (MD2), MCS на известном ответе (MD8), Холм с N (MD3), DM по периодам, bootstrap Δ. Синтетика, без сети."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d

N = 600


def ar1(n, rho, seed):
    rng = np.random.default_rng(seed)
    e, x = rng.normal(size=n), np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_blocks_are_contiguous_circular_and_seeded():
    idx = stats_d.block_idx(N, B_=200)
    assert idx.shape == (200, N) and np.array_equal(idx, stats_d.block_idx(N, B_=200))
    runs = idx[:, :stats_d.BLOCK * 5].reshape(200, 5, stats_d.BLOCK)
    assert (np.diff(runs, axis=2) % N == 1).all()   # внутри блока даты смежны (по кругу)


def test_md2_shuffled_dates_break_blocks_and_underestimate_variance():
    ok, bad = stats_d.block_idx(N, B_=400), stats_d.block_idx(N, B_=400, shuffle_dates=True)
    runs = bad[:, :stats_d.BLOCK * 5].reshape(400, 5, stats_d.BLOCK)
    assert not (np.diff(runs, axis=2) % N == 1).all()   # мутант ловится структурным свойством
    x = ar1(N, 0.8, 3)
    # перемешанные даты разрушают зависимость: дисперсия среднего занижена (на ряду с автокорреляцией)
    assert stats_d.boot_means(x, ok).std() > 1.5 * stats_d.boot_means(x, bad).std()


def test_boot_delta_known_answer():
    rng = np.random.default_rng(1)
    dl = rng.normal(-0.5, 1.0, N)
    r = stats_d.boot_delta(dl, stats_d.block_idx(N))
    assert r["hi95"] < 0 and r["p_boot"] < 0.01 and r["lo95"] < r["mean"] < r["hi95"]
    z = stats_d.boot_delta(rng.normal(0, 1.0, N), stats_d.block_idx(N))
    assert z["lo95"] < 0 < z["hi95"] and 0.05 < z["p_boot"] < 0.95
    assert stats_d.boot_delta(np.full(N, 0.1), stats_d.block_idx(N))["p_boot"] > 0.99   # знак не перепутан: Δ > 0 -> P* ~ 1


def test_mcs_drops_clearly_worse_model_and_keeps_identical():
    rng = np.random.default_rng(2)
    base = rng.normal(1.0, 0.3, N)
    L = np.column_stack([base + rng.normal(0, 0.05, N), base + rng.normal(0, 0.05, N), base + 0.5 + rng.normal(0, 0.05, N)])
    idx = stats_d.block_idx(N, B_=1000)
    r = stats_d.mcs(L, idx)
    assert list(r["in"]) == [True, True, False] and r["order"][0] == 2   # MD8: худшая исключена, равные остаются
    same = stats_d.mcs(np.column_stack([base, base, base]), idx)
    assert same["in"].all() and not same["order"]


def test_mcs_size_under_null_is_not_inflated():
    idx = stats_d.block_idx(300, B_=500)
    drop = 0
    for s in range(30):
        L = np.random.default_rng(100 + s).normal(1, 0.3, (300, 4))
        drop += (~stats_d.mcs(L, idx)["in"]).any()
    assert drop <= 8   # при α = 0,10 ожидаемо ≈ 3 из 30; верхняя граница 8 — вероятность превышения при корректной процедуре < 0,5 %


def test_holm_n_grows_with_N_and_md3():
    p = np.array([0.01, 0.2, 0.03])
    h = stats_d.holm_n(p, 106)
    assert np.isclose(h[0], 1.0) and np.isclose(stats_d.holm_n(np.array([0.004]), 106)[0], 0.424)
    assert (stats_d.holm_n(p, 106) >= stats_d.holm_n(p, 3) - 1e-15).all() and (stats_d.holm_n(p, 106) > stats_d.holm_n(p, 3) + 1e-9).any()
    assert np.isclose(stats_d.holm_n(np.array([0.004]), 1)[0], 0.004) and stats_d.holm_n(np.array([0.004]), 106)[0] > 0.4   # N = 1 — мутант
    with pytest.raises(ValueError):
        stats_d.holm_n(p, 2)


def test_dm_periods_sign_and_masks():
    d = pd.bdate_range("2017-01-02", "2026-02-20")
    rng = np.random.default_rng(4)
    rv = np.exp(rng.normal(0, 0.5, len(d)))
    good, bad = rv * np.exp(rng.normal(0, 0.1, len(d))), rv * 3.0
    td = pd.Series(d)
    rows = stats_d.dm_periods(rv, bad, good, td)
    assert [r["period"] for r in rows] == ["весь период", "2017–2021", "2022", "2023–2025", "без 24.02.2022"]
    assert all(r["d"] < 0 and r["p"] < 0.01 for r in rows)    # конфиг лучше базы -> d < 0
    assert rows[-1]["n"] == rows[0]["n"] - 1 and rows[2]["n"] == (d.year == 2022).sum()
    assert all(r["d"] > 0 for r in stats_d.dm_periods(rv, good, bad, td))   # перепутанный знак ловится
