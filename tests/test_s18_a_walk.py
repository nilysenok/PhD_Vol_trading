"""S-A-IMPL-3 (тестировщик), структура прохода через боевой `walk_variant` / `assemble` / `run` с дешёвой подменой подгонки: 107 помесячных переоценок на D_M = 2198 дат,
пул = закрытые пары (t⁺ ≤ D), валидация только закрытыми, λ-выбор (ничья — больший λ), порог корректора 120, нулевая точка при < 250 парах, отказ `run` без реестра. Только monkeypatch (№ 43)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_layers as L
from models_v2.s18 import a_model as M
from models_v2.s18 import a_walk as W
from s18_a_guard import ROOT, guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def xD():
    x, gap = L.inputs()
    return x, gap, M.load_data()


def stub_fit_month(x, log):
    """Подмена fit_month: пишет (D-дату, пул, дни) и отдаёт lnC = ln C cfg8 дней, lnH = ln K1."""
    def fit_month(D, pool, days, var, seeds=W.SEEDS):
        a = int(np.flatnonzero(pd.DatetimeIndex(D.d) == pd.DatetimeIndex(D.d)[days[0]])[0])
        log.append((pd.Timestamp(D.d[days[0]]), np.asarray(pool), np.asarray(days)))
        return dict(lam=1e-3, n_train=len(pool), corr=True, theta=np.zeros(M.NP), lnH=np.log(D.lnk1[days] * 0 + 1) + D.lnk1[days], lnC=np.log(x.set_index("d")["C"].reindex(pd.DatetimeIndex(D.d[days])).to_numpy()))
    return fit_month


def test_walk_is_monthly_107_covers_2198_dates_pool_is_closed_pairs(xD, monkeypatch):
    x, gap, D = xD
    log = []
    monkeypatch.setattr(W, "fit_month", stub_fit_month(x, log))
    parts = W.walk_variant("A2", D, x)
    xd = pd.DatetimeIndex(x["d"])
    months = xd.to_period("M").unique()
    # ОТСТУПЛЕНИЕ ОТ СПЕЦИФИКАЦИИ: в D_M нет ни одной даты марта 2022 (остановка), поэтому месяцев с датами 106, а не 107 (2018-01…2026-11 = 107 календарных).
    assert len(months) == 106 and pd.Period("2022-03") not in months and len(pd.period_range("2018-01", "2026-11", freq="M")) == 107
    assert len(parts) == 106 == len(log)
    assert parts[0][0] == 0 and parts[-1][1] == 2198 and all(p[1] == q[0] for p, q in zip(parts, parts[1:]))
    assert (xd[[p[0] for p in parts]].to_period("M") == months).all()
    assert all(xd[a - 1].month != xd[a].month for a, _, _ in parts[1:])                        # первый торговый день месяца
    for (a, b, _), (d, pool, days) in zip(parts, log):
        assert d == xd[a] and (D.tp[pool] <= np.datetime64(d)).all() and len(pool) == (D.tp <= np.datetime64(d)).sum()   # закрыто (t⁺ ≤ D), и ВСЕ закрытые
        assert len(days) == b - a and (np.asarray(D.d)[days] >= np.datetime64(d)).all()


def test_mutation_refit_not_monthly_caught(xD, monkeypatch):
    x, gap, D = xD
    monkeypatch.setattr(W, "fit_month", stub_fit_month(x, []))
    monkeypatch.setattr(W, "refit_positions", lambda idx, first: list(range(0, len(idx), 21)))      # «ежедневно/ежегодно»: другой счёт
    assert len(W.walk_variant("A2", D, x)) != 106


def test_mutation_closed_plus_one_pool_caught_by_spy(xD, monkeypatch):
    x, gap, D = xD
    log = []
    monkeypatch.setattr(W, "fit_month", stub_fit_month(x, log))
    monkeypatch.setattr(W, "closed_rows", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(1, "D")))
    W.walk_variant("A2", D, x)
    leaked = [d for d, pool, _ in log if (D.tp[pool] > np.datetime64(d)).any()]
    assert len(leaked) > 50                                       # то же условие в тесте выше уронило бы мутанта


def test_assemble_over_full_stub_gives_w5g_dates_rows_and_x1_a_layer(xD, monkeypatch):
    x, gap, D = xD
    monkeypatch.setattr(W, "fit_month", stub_fit_month(x, []))
    df = W.assemble("A2", x, gap, W.walk_variant("A2", D, x))
    w5g = pd.read_parquet(ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    assert len(df) == 2198 and (df["d"].to_numpy() == x["d"].to_numpy()).all() and df["refit_date"].nunique() == 106
    assert np.abs(df["F"].to_numpy() / w5g["F"].to_numpy() - 1).max() <= 1e-12            # C_A = cfg8 ⇒ F_A = W5-G
    assert np.abs(df["X1_A"].to_numpy() / x["F"].to_numpy() - 1).max() <= 1e-12
    assert list(df.columns[-28:]) == W.NAMES and list(df.columns[:14]) == ["d", "t_plus", "rv", "K1", "H_A", "C_A", "X1_A", "F", "pi", "beta_gap", "refit_date", "lam", "n_train", "corr"]
    assert len(W.NAMES) == 28 and W.LAMS == (1e-4, 1e-3, 1e-2) and (W.VAL, W.MIN_PAIRS, W.MIN_CFG) == (126, 250, 120)


def test_below_250_pairs_is_zero_point_equal_to_cfg8_first_month(xD):
    x, gap, D = xD
    pool = W.closed_rows(D.tp, x["d"].iloc[0])[:W.MIN_PAIRS - 1]          # на первой D реально 253 пары (≥ 250), ветка проверяется урезанным пулом
    assert len(pool) < W.MIN_PAIRS and len(W.closed_rows(D.tp, x["d"].iloc[0])) >= W.MIN_PAIRS
    dp = W._day_pos(D, x)
    r = W.fit_month(D, pool, dp[0:20], "A2")
    assert not r["theta"].any() and np.isnan(r["lam"]) and r["corr"] is False
    assert np.abs(np.exp(r["lnC"]) / x["C"].to_numpy()[0:20] - 1).max() <= 1e-12


def test_lambda_ties_pick_larger_and_minimum_validation_qlike_wins(xD, monkeypatch):
    x, gap, D = xD
    pool = W.closed_rows(D.tp, x["d"].iloc[40])
    monkeypatch.setattr(M, "fit", lambda V, var, lam, seed=0, corr=True, maxiter=500: np.zeros(M.NP))
    assert W.choose_lambda(D, pool, "A0", (0,)) == 1e-2                                       # ничья ⇒ больший λ

    def shifted(V, var, lam, seed=0, corr=True, maxiter=500):                                 # только λ = 1e-3 без сдвига уровня
        v = np.zeros(M.NP)
        v[0] = 0.0 if lam == 1e-3 else 0.5
        return v
    monkeypatch.setattr(M, "fit", shifted)
    assert W.choose_lambda(D, pool, "A0", (0,)) == 1e-3


def test_corrector_switch_at_120_cfg_pairs_uses_pool_not_validation_leak(xD, monkeypatch):
    x, gap, D = xD
    seen = []
    monkeypatch.setattr(M, "fit", lambda V, var, lam, seed=0, corr=True, maxiter=500: (seen.append(corr), np.zeros(M.NP))[1])
    xd = pd.DatetimeIndex(x["d"])
    dp = W._day_pos(D, x)
    for d0, want in (("2018-04-02", False), ("2018-09-03", True)):
        a = int(np.flatnonzero(xd >= d0)[0])
        pool = W.closed_rows(D.tp, xd[a])
        assert (W.n_cfg(D, pool) >= 120) is want
        seen.clear()
        W.fit_month(D, pool, dp[a:a + 5], "A1", seeds=(0,))
        assert seen[-1] is want                                                               # финальная подгонка: корректор по пулу


def test_check_snapshot_sees_any_change_of_outputs(tmp_path, monkeypatch):
    """`--check` сравнивает sha256 выходов до и после: снимок `_tree` обязан различать запись в любой parquet."""
    from models_v2.s18 import a_walk_chk as K
    monkeypatch.setattr(W, "A_DIR", tmp_path)
    for v in W.OUTS.values():
        pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / v)
    t0 = K._tree()
    assert sorted(t0) == sorted(W.OUTS.values()) and K._tree() == t0
    pd.DataFrame({"a": [1, 3]}).to_parquet(tmp_path / "a1.parquet")
    assert K._tree() != t0


def test_run_refuses_without_registry_or_with_failed_check_before_any_work(monkeypatch, tmp_path):
    def reached(*a, **k):
        raise AssertionError("run дошёл до чтения входов — реестр до прогноза нарушен")
    monkeypatch.setattr(L, "inputs", reached)
    monkeypatch.setattr(W.R, "require", lambda cfg, trials=W.R.TRIALS: (_ for _ in ()).throw(SystemExit("нет в реестре")))
    with pytest.raises(SystemExit):
        W.run("A2", out=tmp_path)
    monkeypatch.setattr(W.R, "require", lambda cfg, trials=W.R.TRIALS: None)
    monkeypatch.setattr(W.R, "check", lambda *a, **k: ["реестр позже прогноза"])
    with pytest.raises(SystemExit):
        W.run("A2", out=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_run_refuses_to_overwrite_different_existing_output(xD, monkeypatch, tmp_path):
    x, gap, D = xD
    monkeypatch.setattr(W.R, "require", lambda cfg, trials=W.R.TRIALS: None)
    monkeypatch.setattr(W.R, "check", lambda *a, **k: [])
    monkeypatch.setattr(W, "fit_month", stub_fit_month(x, []))
    (tmp_path / "a2.parquet").write_bytes(b"x")
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "a2.parquet")
    before = (tmp_path / "a2.parquet").read_bytes()
    with pytest.raises(SystemExit):
        W.run("A2", out=tmp_path)
    assert (tmp_path / "a2.parquet").read_bytes() == before
