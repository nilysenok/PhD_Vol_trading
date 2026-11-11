"""S-A-IMPL-3 (тестировщик), PIT помесячной переоценки через боевой путь (`walk_variant` / `fit_month` / `choose_lambda`) на коротком отрезке — один месяц, seed 0 (№ 144, 164):
порча RV пар с t⁺ > D и баров после месяца не меняет θ и прогнозы месяца побитно; «закрыто + 1» и валидация с t⁺ > D ловятся; A2p при π = id ≡ A2, при реальной π — нет. Только monkeypatch (№ 43)."""
import copy

import numpy as np
import pandas as pd
import pytest

from models_v2.s17.fwd_src import LAST_SESSION, Src
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_layers as L
from models_v2.s18 import a_model as M
from models_v2.s18 import a_walk as W
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def ctx():
    x, gap = L.inputs()
    src = Src()
    return x, M.load_data(), src.bars("SBER", LAST_SESSION), src.index_days(LAST_SESSION)


def month(x, date):
    xd = pd.DatetimeIndex(x["d"])
    pos = W.refit_positions(xd, W.FIRST)
    a = next(p for p in pos if xd[p] >= pd.Timestamp(date))
    return a, (pos + [len(xd)])[pos.index(a) + 1]


def corrupt_bars(bars, d_last, seed=7):
    g = np.random.default_rng(seed)
    out = bars.copy()
    late = out["begin"].dt.normalize() > d_last
    for c in ("open", "close"):
        out.loc[late, c] = out.loc[late, c].to_numpy() * np.exp(g.normal(0, 0.2, late.sum()))
    last = out[out["begin"].dt.normalize() == d_last].tail(1)
    ev = pd.concat([last.assign(begin=d_last + pd.Timedelta(hours=19 + i), close=last["close"].to_numpy() * (2.0 + i)) for i in range(3)])
    return pd.concat([out, ev], ignore_index=True)


def month_result(D, x, a, var="A2"):
    return W.walk_variant(var, D, x, positions={a}, seeds=(0,))[0][2]


def corrupted(ctx, a, b, bars_too=True):
    x, D, bars, idays = ctx
    d0, d_last = x["d"].iloc[a], x["d"].iloc[b - 1]
    if bars_too:
        k1, cfg, bmix = M.load_inputs()
        D2 = M.build_data(A.build(corrupt_bars(bars, d_last), idays), k1, cfg, bmix)
    else:
        D2 = copy.copy(D)
    D2.rv_next = np.where(D2.tp > np.datetime64(d0), D2.rv_next * 7.0, D2.rv_next)
    return D2


def same(r1, r2):
    return all(np.array_equal(r1[q], r2[q]) for q in ("theta", "lnH", "lnC"))


def test_corrupting_future_pairs_and_bars_does_not_change_month_bitwise(ctx):
    x, D, bars, idays = ctx
    a, b = month(x, "2020-07-01")
    assert D.tp[D.tp > np.datetime64(x["d"].iloc[a])].size > 0
    ref = month_result(D, x, a)
    assert ref["theta"].any() and ref["n_train"] > 800 and ref["lam"] in W.LAMS   # реально обучено, не нулевая точка; λ из сетки
    assert same(ref, month_result(corrupted(ctx, a, b), x, a))


def test_mutation_closed_plus_one_pool_caught_through_real_path(ctx, monkeypatch):
    x, D, bars, idays = ctx
    a, b = month(x, "2020-07-01")
    ref, D2 = month_result(D, x, a), corrupted(ctx, a, b, bars_too=False)
    monkeypatch.setattr(W, "closed_rows", lambda tp, d0: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(5, "D")))
    assert not same(month_result(D, x, a), month_result(D2, x, a))             # мутант видит порчу будущего ⇒ честный тест (выше) его бы уронил


def spy_views(monkeypatch, log):
    orig = M.Data.view

    def view(self, sel):
        log.append(np.asarray(sel))
        return orig(self, sel)
    monkeypatch.setattr(M.Data, "view", view)


def test_training_and_validation_use_only_closed_pairs_spy_and_mutation(ctx, monkeypatch):
    x, D, bars, idays = ctx
    a, b = month(x, "2019-03-01")
    dp, d0 = W._day_pos(D, x), np.datetime64(x["d"].iloc[a])
    pool = W.closed_rows(D.tp, x["d"].iloc[a])
    monkeypatch.setattr(M, "fit", lambda V, var, lam, seed=0, corr=True, maxiter=500: np.zeros(M.NP))
    log = []
    spy_views(monkeypatch, log)
    W.fit_month(D, pool, dp[a:b], "A1", seeds=(0,))
    fits = [s for s in log if not np.array_equal(s, dp[a:b])]
    assert len(fits) == 3 and all((D.tp[s] <= d0).all() for s in fits)         # обучение, валидация, финал: только t⁺ ≤ D

    def leaky(D_, pool_, var, seeds):                                          # мутант: валидация — последние 126 пар, сдвинутые на 30 вперёд (t⁺ > D)
        n = len(pool_)
        D_.view(np.arange(n - W.VAL - 30, n - 30))
        D_.view(np.arange(n - W.VAL + 30, n + 30))
        return 1e-3
    monkeypatch.setattr(W, "choose_lambda", leaky)
    log.clear()
    W.fit_month(D, pool, dp[a:b], "A1", seeds=(0,))
    assert any((D.tp[s] > d0).any() for s in log if not np.array_equal(s, dp[a:b]))   # тот же критерий видит мутанта


def test_a2p_with_identity_permutation_equals_a2_bitwise_and_real_permutation_does_not(ctx, monkeypatch):
    x, D, bars, idays = ctx
    a, b = month(x, "2019-03-01")
    dp, pool = W._day_pos(D, x), W.closed_rows(D.tp, x["d"].iloc[a])
    run = lambda v: W.fit_month(D, pool, dp[a:b], v, seeds=(0,))
    real_a2, real_a2p = run("A2"), run("A2p")
    assert not np.array_equal(real_a2["theta"], real_a2p["theta"]) and not np.array_equal(real_a2["lnC"], real_a2p["lnC"])
    monkeypatch.setattr(A, "PERM", np.arange(A.NS))
    assert same(run("A2"), run("A2p"))
    monkeypatch.undo()
    monkeypatch.setattr(M, "tau_of", lambda var: A.TAU)                         # мутант: перестановка не применяется
    assert same(run("A2"), run("A2p"))                                          # …и тогда A2p ≡ A2 при реальной π — реальная проверка выше его бы уронила
