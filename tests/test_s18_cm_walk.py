"""S-C-IMPL-3 (тестировщик), проход: помесячные переоценки, пул «t⁺ ≤ D», включение 2018-07-02, выбор λ, итог по минимальному перезапуску, отказы `run`, сборка таблицы.
Даты — реальное окно D_M (`x1_w3_core.load`), данные пар — синтетика с известной структурой. Подмены только через monkeypatch (№ 43); реальные файлы охраняет autouse."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as M
from models_v2.s18 import cm_reg
from models_v2.s18 import cm_walk as W
from s18_cm_guard import guard_real_files  # noqa: F401  (autouse)
from s18_cm_synth import make_data

TH = [0.4, 0.6, 0.1, -0.1, -0.3, 0.2]


@pytest.fixture(scope="module")
def x():
    return C.load()


@pytest.fixture(scope="module")
def D(x):
    return make_data(len(x), seed=7, theta=TH, shape=80.0)


def month_pos(x, ym):
    d = pd.DatetimeIndex(x["d"])
    return [a for a in YA.month_starts(d) if (d[a].year, d[a].month) == ym][0]


def one_month(x, D, ym, var="CK", seeds=(0, 1)):
    a = month_pos(x, ym)
    (aa, b, r), = W.walk_variant(var, D, x, positions=[a], seeds=seeds)
    return a, b, r


def corrupt_pairs(D, idx, fields=("rv", "labc", "labp", "Hc", "Hp", "zc", "zp")):
    kw = {k: getattr(D, k).copy() for k in M.FIELDS}
    for k in fields:
        kw[k][idx] = kw[k][idx] * 4.1
    return M.Data(**kw)


def test_constants_and_calendar_106_months_first_day_activation(x):
    assert (W.VAL, W.MIN_VALID, W.LAM_DEF) == (126, 252, 1e-2) and set(W.IDS) == set(M.VARIANTS) and W.IDS["CK"] == "S-CK"
    pos = YA.month_starts(pd.DatetimeIndex(x["d"]))
    d = pd.DatetimeIndex(x["d"])
    assert len(x) == 2198 and len(pos) == 106 and YA.ACTIVATION == pd.Timestamp("2018-07-02")
    assert all(d[a].month != d[a - 1].month for a in pos[1:]) and (d[pos[0]].year, d[pos[0]].month) == (d[0].year, d[0].month)
    assert not ((d.year == 2022) & (d.month == 3)).any()


def test_before_activation_C_equals_cfg8_bitwise_and_params_zero(x, D):
    a, b, r = one_month(x, D, (2018, 5))
    assert pd.Timestamp(x["d"].iloc[a]) < YA.ACTIVATION and r["n_train"] == 0 and (r["M"] == 1.0).all() and (r["theta"] == 0).all()
    assert ((x["C"].to_numpy(float)[a:b] * r["M"]) == x["C"].to_numpy(float)[a:b]).all()


def test_active_month_uses_only_closed_pairs_forecast_independent_of_future(x, D):
    a, b, r = one_month(x, D, (2019, 3))
    tp, d0 = pd.DatetimeIndex(x["t_plus"]), pd.Timestamp(x["d"].iloc[a])
    pool = YA.closed_pool(tp, x["d"], d0)
    assert r["n_train"] == len(pool) and (tp[pool] <= d0).all() and (np.asarray(x["d"])[pool] < d0).all()
    fut = np.flatnonzero(tp > d0)
    assert len(fut) > 0
    D2 = corrupt_pairs(D, fut, fields=("rv", "labc", "labp"))                  # метки и RV будущих пар
    D2 = corrupt_pairs(D2, np.arange(a, b), fields=("rv", "labc", "labp"))     # метки t⁺ пар самого месяца
    a2, b2, r2 = one_month(x, D2, (2019, 3))
    assert (r2["theta"] == r["theta"]).all() and (r2["M"] == r["M"]).all() and r2["lam"] == r["lam"]


def test_mutation_closed_plus_one_caught_in_training(x, D, monkeypatch):
    a = month_pos(x, (2019, 3))
    d0 = pd.Timestamp(x["d"].iloc[a])
    fut = np.flatnonzero(pd.DatetimeIndex(x["t_plus"]) > d0)
    base = one_month(x, D, (2019, 3))[2]
    D2 = corrupt_pairs(D, fut, fields=("rv", "labc", "labp"))
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)                      # мутант: пара видна в день прогноза d
    m1, m2 = one_month(x, D, (2019, 3))[2], one_month(x, D2, (2019, 3))[2]
    assert m1["n_train"] > base["n_train"]
    assert not (m1["theta"] == m2["theta"]).all() or m1["lam"] != m2["lam"]      # мутант зависит от метки будущей пары — тест выше падает


def test_lambda_default_below_252_and_tie_goes_to_larger_and_split_is_126(monkeypatch):
    Dd = make_data(600, seed=1)
    assert W.choose_lambda(Dd, np.arange(251), "CK", (0,)) == 1e-2
    seen = []
    zero = {"theta": np.zeros(M.NP), "losses": np.zeros(1), "seed": 0, "loss": 0.0}

    def fake_fit(V, var, lam, mu=None, seeds=M.SEEDS, maxiter=500):
        seen.append(len(V.rv))
        return zero
    monkeypatch.setattr(M, "fit", fake_fit)
    assert W.choose_lambda(Dd, np.arange(400), "CK", (0,)) == 1e-2                # равные потери — больший λ
    assert seen == [400 - 126] * 3
    monkeypatch.setattr(W, "M", M)
    qs = iter([0.5, 0.2, 0.3])
    monkeypatch.setattr(M, "qlike_sum", lambda th, V, var: next(qs))
    assert W.choose_lambda(Dd, np.arange(400), "CK", (0,)) == 1e-3                 # лучший по QLIKE суммы на последних 126
    qs2 = iter([0.5, 0.5, 0.9])
    monkeypatch.setattr(M, "qlike_sum", lambda th, V, var: next(qs2))
    assert W.choose_lambda(Dd, np.arange(400), "CK", (0,)) == 1e-3                 # ничья между 1e-4 и 1e-3 — больший


def test_validation_uses_only_pool_pairs_not_future(monkeypatch):
    Dd = make_data(600, seed=2)
    seen = []

    def spy(V, var, lam, mu=None, seeds=M.SEEDS, maxiter=500):
        seen.append(V.sel.copy())
        return {"theta": np.zeros(M.NP), "losses": np.zeros(1), "seed": 0, "loss": 0.0}
    monkeypatch.setattr(M, "fit", spy)
    pool = np.arange(300)
    W.choose_lambda(Dd, pool, "CK", (0,))
    assert max(s.max() for s in seen) < 300 - 126 and seen[0].max() == 173


def test_one_part_per_month_contiguous_cover_and_yearly_mutation_caught(x, D, monkeypatch):
    stub = lambda Dd, pool, days, var, seeds=M.SEEDS: W.zero_month(Dd, days)
    monkeypatch.setattr(W, "fit_month", stub)
    parts = W.walk_variant("CK", D, x)
    assert len(parts) == 106 and parts[0][0] == 0 and parts[-1][1] == len(x) and all(p[1] == q[0] for p, q in zip(parts, parts[1:]))
    monkeypatch.setattr(YA, "month_starts", lambda d: [i for i in range(len(d)) if i == 0 or pd.DatetimeIndex(d)[i].year != pd.DatetimeIndex(d)[i - 1].year])
    assert len(W.walk_variant("CK", D, x)) < 106


def test_assemble_identity_before_activation_C_equals_cfg8_and_sum_of_parts(x, D, monkeypatch):
    def stub(Dd, pool, days, var, seeds=M.SEEDS):
        v = torch.as_tensor(np.array([0.5, 0.5, 0.05, -0.05, 0.1, -0.1]))
        f = M.forward(v, Dd.view(days), var)
        return W._res(f, lam=1e-2, n_train=len(pool), seed=0, theta=v.numpy(), spread=0.0)
    monkeypatch.setattr(W, "fit_month", stub)
    from models_v2.s18 import cm_layers as L
    xs, gap = x.iloc[:400], np.zeros(400, int)
    monkeypatch.setattr(YA, "check_activation", lambda d, tp: 124)
    parts = W.walk_variant("CK", make_data(400, seed=7, theta=TH, shape=80.0, cfg8=xs["C"]), xs.reset_index(drop=True))
    df = W.assemble("CK", xs.reset_index(drop=True), gap, parts)
    off = df["d"] < YA.ACTIVATION
    assert off.any() and (df.loc[off, "C"] == df.loc[off, "cfg8"]).all() and (df.loc[off, "M"] == 1.0).all() and not df.loc[off, "active"].any()
    on = ~off & df["F_c"].notna()
    assert on.any() and (abs(df.loc[on, ["F_c", "F_p", "F_u"]].sum(axis=1) / df.loc[on, "C"] - 1) <= 1e-13).all()
    assert list(df.columns[-6:]) == list(M.NAMES["CK"]) and (df.loc[on, "n_train"] > 0).all()


def test_same_is_exact_and_run_refuses_unregistered_before_reading_inputs(monkeypatch):
    a = pd.DataFrame({"u": [1.0, 2.0]})
    assert W._same(a, a.copy()) and not W._same(a, pd.DataFrame({"u": [1.0, 2.0 + 1e-15]}))

    def refuse(cfg, trials=None):
        raise SystemExit("нет в реестре")
    monkeypatch.setattr(cm_reg, "require", refuse)

    def boom(*a, **k):
        raise AssertionError("входы прочитаны до проверки реестра")
    monkeypatch.setattr(W.L, "inputs", boom)
    with pytest.raises(SystemExit):
        W.run("CK", write=False)


def test_run_refuses_when_registry_check_not_clean(monkeypatch):
    monkeypatch.setattr(cm_reg, "require", lambda cfg, trials=None: None)
    monkeypatch.setattr(cm_reg, "check", lambda *a, **k: ["реестр позже прогноза"])
    monkeypatch.setattr(W.L, "inputs", lambda: (_ for _ in ()).throw(AssertionError("входы прочитаны")))
    with pytest.raises(SystemExit):
        W.run("CK", write=False)
