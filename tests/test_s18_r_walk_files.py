"""S-R-IMPL-2: боевые выходы `R/*.parquet` — состав, правило, мутации «D_M сокращён», «θ ≡ 0 ⇔ λ* = 0», «у Rb m ≠ 1», «--check пишет» (W.G4). Мутации — в памяти или в копии (tmp), боевые файлы не трогаются."""
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import r_reg as RG
from models_v2.s18 import r_walk as W
from models_v2.s18 import r_walk_chk as K


@pytest.fixture(scope="module")
def frames():
    from models_v2.s16 import x1_w3_core as C
    return C.load(), {v: pd.read_parquet(W.R_DIR / f) for v, f in W.OUTS.items()}


def test_real_files_pass_state_rule_and_inclusion(frames):
    x, fr = frames
    for v, df in fr.items():
        assert K.check_file(v, df, x) == [], v


def test_real_registry_is_ok_and_before_parquet():
    assert RG.check() == []


def test_mut_dm_shortened_is_caught(frames):
    x, fr = frames
    assert K.check_file("rck", fr["rck"].iloc[:-1], x) != []


def test_mut_theta_nonzero_where_lambda_zero_is_caught(frames):
    x, fr = frames
    df = fr["rck"].copy()
    m = df[(df["active"]) & (df["lam"] == 0.0)]["refit_R"].iloc[0]
    df.loc[df["refit_R"] == m, "gam_c"] = 0.2
    assert any("θ ≡ 0" in b for b in K.check_file("rck", df, x))


def test_mut_lambda_not_derived_from_g_columns_is_caught(frames):
    x, fr = frames
    df = fr["rck"].copy()
    m = df[(df["active"]) & (df["lam"] > 0.0)]["refit_R"].iloc[0]
    df.loc[df["refit_R"] == m, "lam"] = 0.0
    assert any("λ* ≠ правилу" in b for b in K.check_file("rck", df, x))


def test_mut_rb_with_clipping_or_rc_with_three_blocks_is_caught(frames):
    x, fr = frames
    b = fr["rckb"].copy()
    b.loc[b["active"], "m"] = 0.97
    assert K.check_file("rckb", b, x) != []
    c = fr["rckc"].copy()
    c.loc[c["active"], "G_1e-1_b1"] = 0.0
    assert K.check_file("rckc", c, x) != []


def test_mut_different_inclusion_in_one_variant_is_caught(frames):
    x, fr = frames
    df = fr["rck"].copy()
    first = df[df["active"]]["refit_R"].min()
    df.loc[df["refit_R"] == first, "active"] = False
    assert K.check_file("rck", df, x) != []


def test_mut_check_that_writes_is_caught(tmp_path, monkeypatch):
    for f in W.OUTS.values():
        shutil.copy(W.R_DIR / f, tmp_path / f)
    monkeypatch.setattr(W, "R_DIR", tmp_path)
    assert K.check() == []
    orig = K.repro

    def writing(var, df):
        p = tmp_path / "ra2.parquet"
        p.write_bytes(p.read_bytes() + b"0")
        return orig(var, df)
    monkeypatch.setattr(K, "repro", writing)
    assert any("изменил выходы" in b for b in K.check())
