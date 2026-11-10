"""ZD: известный ответ на синтетике и независимая сверка с файлами — без опоры на `--check`. Ловят: AUC против D > 0; знак доли в (в); гребень без центровки / по −D; метка логита D > 0; половины (б) наоборот;
Y3 из файла Y2; s_d со сдвигом на строку; окно v_d с днём d (через `zfeatures`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_feat as FE, x1_zd_core as C, x1_zd_stage as S, x1_zd_tab as T

DATA = Path(__file__).resolve().parents[1] / "data/regen"


def synth(n: int = 330, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2018-01-03", periods=n + 1)
    Z = rng.normal(size=(n, 4))
    Z[:, 0] = (rng.random(n) < 0.2).astype(float)
    return Z, d[:n], d[1:n + 1]


def test_walk_known_answer_label_direction_sign_and_intercept() -> None:
    Z, d, tp = synth()
    D = 5.0 - 2.0 * Z[:, 2] + 0.1 * np.random.default_rng(1).normal(size=len(d))     # D < 0 ⇔ z_s большой; уровень D ≈ 5
    w = C.walk(Z, D, d, tp)
    i = w["idx"].to_numpy()
    assert T.auc(w["pa"], D[i] < 0) > 0.9                      # метка логита 1[D<0] (мутация D>0 → < 0,1)
    assert T.spearman(w["pb"], D[i]) > 0.9                     # знак гребня (мутация по −D → отрицательная)
    assert abs(w["pb"].mean() - D[i].mean()) < 0.3             # центровка/свободный член (мутация без центровки → ≈ 0)


def test_ridge_closed_form_known_answer() -> None:
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(200, 4)), 5.0 + rng.normal(size=200)
    pool, to = np.arange(150), np.arange(150, 200)
    xc, yc = X[pool] - X[pool].mean(0), y[pool] - y[pool].mean()
    beta = np.linalg.solve(xc.T @ xc + np.eye(4), xc.T @ yc)
    assert np.allclose(C.ridge_fit(X, y, pool, to), y[pool].mean() + (X[to] - X[pool].mean(0)) @ beta)


def test_state_share_known_answer() -> None:
    D, e = np.array([-1.0, -2.0, 3.0, 4.0, -5.0]), np.array([1, 1, 0, 0, 0])
    got = C.state_share(e, D, np.arange(5), np.array([0, 2, 3]))
    assert got.tolist() == [1.0, pytest.approx(1 / 3), pytest.approx(1 / 3)]                # e=1: обе D<0; e=0: одна из трёх
    assert C.state_share(np.array([0, 0, 0]), np.array([-1.0, 1.0, -1.0]), np.arange(3), np.array([0])).tolist() == [pytest.approx(2 / 3)]


def test_pred_table_auc_direction_and_halves_known_answer() -> None:
    rng = np.random.default_rng(4)
    n = 400
    D, e, year = rng.normal(size=n), np.zeros(n, int), np.full(n, 2020)
    pr = {"Y2": dict(pa=(D < 0).astype(float), pb=D.copy(), pc=(D < 0) + 0.0, pg=(D < 0) + 0.0)}
    t = T.pred_table(pr, {"Y2": D}, e, year, np.ones(n, bool)).set_index("period").loc["2020"]
    assert t["auc_a"] == 1.0 and t["auc_c"] == 1.0 and t["auc_g"] == 1.0 and t["rho_b"] == pytest.approx(1.0)       # AUC против 1[D<0]
    assert t["D_hi"] == pytest.approx(D[D >= np.median(D)].mean()) and t["D_hi"] > 0 > t["D_lo"]               # верхняя половина прогноза (б) = худшие D


def test_each_y_uses_its_own_file_and_features_are_independent() -> None:
    inp = S.load()
    own = {"Y1": "harj_x1/YA/y1", "Y2": "harj_x1/YA/y2", "Y3": "harj_x1/YB/y3"}
    t, _ = S.compute(inp)
    for y, p in own.items():
        f = pd.read_parquet(DATA / (p + ".parquet"))
        assert np.array_equal(inp["ys"][y]["F"].to_numpy(), f["F"].to_numpy())
        want = C.dloss(f["rv"].to_numpy(float), f["F"].to_numpy(float), f["X1"].to_numpy(float)).sum()
        got = t["STATES"].query("y == @y and state == 'all' and period == 'all'")["sum_D"].iloc[0]
        assert got == pytest.approx(want, rel=1e-9) and "theta" in inp["ys"]["Y3"].columns
    assert not np.allclose(inp["ys"]["Y3"]["F"], inp["ys"]["Y2"]["F"])
    # s_d: независимый пересчёт по складу M10 для дня d (строка склада = день d)
    store, d = FE.load_store(), inp["d"]
    pos = store["days"].get_indexer(d)
    s = np.array([FE.day_features(store["r2"][p], store["present"][p])[0] for p in pos])
    assert np.array_equal(inp["Z"][:, 2], s) and np.array_equal(inp["Z"][:, 0], pd.read_parquet(DATA / "harj_deep/E1/x1.parquet")["s"].to_numpy(int))


def test_v_window_is_exactly_20_past_days_via_zfeatures() -> None:
    n = 40
    r2 = np.ones((n, 53))
    r2[9] *= 100.0                                  # день 9 — вне окна дня 30 (окно 10…29)
    r2[30] *= 2.0                                   # RV_30 = 106, RV прошлых 20 = 53
    store = dict(days=pd.bdate_range("2018-01-03", periods=n), r2=r2, present=np.ones((n, 53), bool))
    z = C.zfeatures(store, store["days"][[5, 30]], np.zeros(2), np.zeros(2))
    assert z[1, 3] == pytest.approx(np.log(2.0))                      # включение дня d или окно 21 дня дали бы другое число
    assert z[0, 3] == 0.0                                             # меньше 20 прошлых дней → 0
    assert z[1, 1] == pytest.approx(np.log(2.0) - np.log(106.0 / 53))  # x_d: max r² = 2, RV/n = 2·53/53 + … known
