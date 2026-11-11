"""S-C-IMPL-3 (тестировщик), слои: вложение (C_V = cfg8 ⇒ X1_V = x1, F = W5-G), канал доверия над НОВЫМ C_V, W5 над X1_V (не над X1).
Реальные ряды — только чтение (сохранённые x1, w5g); подмены через monkeypatch (№ 43); реальные файлы охраняет autouse."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import cm_layers as L
from s18_cm_guard import ROOT, guard_real_files  # noqa: F401  (autouse)

TOL = 1e-12


@pytest.fixture(scope="module")
def xg():
    return L.inputs()


@pytest.fixture(scope="module")
def w5g():
    return pd.read_parquet(ROOT / "data/regen/harj_x1/W5/w5g.parquet")


@pytest.fixture(scope="module")
def base(xg):
    x, gap = xg
    return L.layers(x, x["C"], gap)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def test_inputs_are_window_dm_2198_and_gap_pit(xg):
    x, gap = xg
    assert len(x) == 2198 and len(gap) == 2198 and gap.dtype.kind == "i" and (gap >= 0).all()
    assert {"d", "t_plus", "rv", "M", "C", "s", "F", "K1", "K2", "K3"} <= set(x.columns)


def test_cfg8_channel_reproduces_saved_x1_and_w5g(xg, base, w5g):
    x, _ = xg
    assert rel(base["X1_V"], x["F"]) <= TOL and rel(base["pi"], x["pi"]) <= TOL
    assert rel(base["F"], w5g["F"]) <= TOL and np.abs(base["beta_gap"].to_numpy() - w5g["beta"].to_numpy()).max() <= TOL
    assert (base["d"].to_numpy() == x["d"].to_numpy()).all() and (base["rv"].to_numpy() == x["rv"].to_numpy()).all()


def test_changed_channel_changes_trust_layer_and_w5_runs_over_x1_v(xg, base):
    x, gap = xg
    c_v = x["C"].to_numpy(float) * np.where(np.arange(len(x)) % 7 == 0, 1.2, 0.9)
    out = L.layers(x, c_v, gap)
    assert rel(out["X1_V"], base["X1_V"]) > 1e-3 and not np.allclose(out["pi"], base["pi"]) and rel(out["F"], base["F"]) > 1e-3
    again = L.W5.walk(x["d"], x["t_plus"], x["rv"], out["X1_V"], gap, base_name="X1_V")      # W5 над X1_V, а не над сохранённым x1
    assert np.array_equal(out["F"].to_numpy(), again["F"].to_numpy())


def test_mutation_trust_layer_over_cfg8_instead_of_C_V_caught(xg, base, monkeypatch):
    x, gap = xg
    c_v = x["C"].to_numpy(float) * 1.3
    honest = L.layers(x, c_v, gap)
    orig = L.E1.run
    monkeypatch.setattr(L.E1, "run", lambda rv, M, c, s, d, tp: orig(rv, M, x["C"].to_numpy(float), s, d, tp))
    mut = L.layers(x, c_v, gap)
    assert rel(mut["X1_V"], base["X1_V"]) <= TOL and rel(honest["X1_V"], base["X1_V"]) > 1e-3     # мутант игнорирует C_V: тест выше падает


def test_mutation_w5_over_x1_instead_of_x1_v_caught(xg, monkeypatch):
    x, gap = xg
    c_v = x["C"].to_numpy(float) * 1.3
    honest = L.layers(x, c_v, gap)
    orig = L.W5.walk
    monkeypatch.setattr(L.W5, "walk", lambda d, tp, rv, base, g, base_name="X1": orig(d, tp, rv, x["F"], g, base_name=base_name))
    mut = L.layers(x, c_v, gap)
    assert rel(mut["F"], honest["F"]) > 1e-4                                                      # F мутанта не равен честному


def test_check_nest_is_clean_and_only_reads_files():
    from models_v2.s18 import cm_feat as CF
    if not CF.FEAT_PATH.exists():
        pytest.skip("feat.parquet не записан")
    dirs = [ROOT / "data/regen/harj_x1/W5", CF.CM_DIR]
    snap = lambda: {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for d in dirs for p in d.rglob("*") if p.is_file()}
    before = snap()
    assert L.TOL == TOL and L.check_nest() == []
    assert snap() == before
