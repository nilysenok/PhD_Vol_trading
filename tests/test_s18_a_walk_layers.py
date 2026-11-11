"""S-A-IMPL-3 (тестировщик), слои над C_A: вложение (C = cfg8 ⇒ X1_A ≡ x1, F_A ≡ W5-G, 1e-12), нулевые параметры, W5 над X1_A (а не над X1), сетка α прежняя, PIT слоёв.
Известный ответ — сохранённые `x1` (через `x1_w3_core.load`) и `data/regen/harj_x1/W5/w5g.parquet` (только чтение). Только monkeypatch (№ 43)."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_w5_gap as W5
from models_v2.s18 import a_layers as L
from models_v2.s18 import a_model as M
from s18_a_guard import ROOT, guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def xg():
    return L.inputs()


@pytest.fixture(scope="module")
def base(xg):
    return L.layers(*(xg[0], xg[0]["C"], xg[1]))


@pytest.fixture(scope="module")
def w5g():
    return pd.read_parquet(ROOT / "data/regen/harj_x1/W5/w5g.parquet")


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def test_dm_is_2198_dates_and_columns(xg, base):
    x, gap = xg
    assert len(x) == 2198 == len(base) == len(gap) and (base["d"].to_numpy() == x["d"].to_numpy()).all()
    assert (x["t_plus"].min(), x["t_plus"].max()) == (pd.Timestamp("2018-01-04"), pd.Timestamp("2026-11-10"))


def test_cfg8_in_gives_saved_x1_and_w5g(xg, base, w5g):
    x = xg[0]
    assert rel(base["X1_A"], x["F"]) <= 1e-12
    for a, b in (("pi", "pi"), ("a0", "a0"), ("a1", "a1")):
        assert rel(base[a], x[b]) <= 1e-12
    assert rel(base["F"], w5g["F"]) <= 1e-12 and np.abs(base["beta_gap"].to_numpy() - w5g["beta"].to_numpy()).max() <= 1e-12


def test_zero_params_of_model_give_f_a_equal_w5g(xg, w5g):
    x, gap = xg
    D = M.load_data()
    _, lnC = M.forward(torch.zeros(M.NP, dtype=torch.float64), D.view(np.arange(len(D.d))), "A2")
    c0 = pd.Series(np.exp(lnC.numpy()), index=pd.DatetimeIndex(D.d)).reindex(pd.DatetimeIndex(x["d"])).to_numpy()
    assert np.isfinite(c0).all()
    assert rel(L.layers(x, c0, gap)["F"], w5g["F"]) <= 1e-12


def moved(xg):
    x, gap = xg
    return x, gap, x["C"].to_numpy(float) * np.exp(0.15 * np.sin(np.arange(len(x)) / 7.0))


def test_w5_layer_is_over_x1_a_not_over_x1(xg, base, monkeypatch):
    x, gap, c = moved(xg)
    out = L.layers(x, c, gap)
    assert rel(out["X1_A"], x["F"]) > 1e-3                                   # C_A другой ⇒ X1_A ≠ x1
    over_a = W5.walk(x["d"], x["t_plus"], x["rv"], out["X1_A"], gap, base_name="X1_A")["F"]
    over_x1 = W5.walk(x["d"], x["t_plus"], x["rv"], x["F"], gap, base_name="X1")["F"]
    assert rel(out["F"], over_a) <= 1e-12 and rel(out["F"], over_x1) > 1e-4
    orig = W5.walk

    def mutant(d, tp, rv, b, g, **kw):                                       # мутант: W5-слой над X1 вместо X1_A
        return orig(d, tp, rv, x["F"], g, **kw)
    monkeypatch.setattr(W5, "walk", mutant)
    bad = L.layers(x, c, gap)
    assert rel(bad["F"], over_a) > 1e-4                                      # тест выше его бы уронил


def test_alpha_grid_is_x1_grid_mutation_caught(xg, base, monkeypatch):
    x, gap = xg
    assert E1.GRID == (0.01, 0.02, 0.05)
    monkeypatch.setattr(E1, "GRID", (0.01, 0.05, 0.2))
    assert rel(L.layers(x, x["C"], gap)["X1_A"], x["F"]) > 1e-9              # другая сетка ⇒ вложение рвётся


def test_layers_pit_future_c_a_does_not_change_past(xg):
    x, gap, c = moved(xg)
    n0 = 1500
    base_out = L.layers(x, c, gap)
    c2 = c.copy()
    c2[n0:] = c2[n0:] * np.exp(np.random.default_rng(3).normal(0, 0.3, len(c) - n0))
    out2 = L.layers(x, c2, gap)
    for col in ("X1_A", "pi", "a0", "a1", "F", "beta_gap"):
        assert np.array_equal(base_out[col].to_numpy()[:n0 - 25], out2[col].to_numpy()[:n0 - 25]), col
