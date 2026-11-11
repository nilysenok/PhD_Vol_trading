"""S-Z-IMPL-2 (Ж-258), тестировщик: боевой путь на реальных данных (только чтение сохранённых рядов) — вложение Н2 побитно, слои X1 / W5 над C_Z, PIT на одном месяце,
состав выходов `check_file` (2198 дат = D_M, включение, θ ≡ 0 ⇔ λ* = 0), `--check` только читает (мутация № 123), «D_M сокращён». Мутанты — monkeypatch."""
import torch  # noqa: F401  (первым, класс № 152)

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_w5_gap as W5
from models_v2.s18 import z_feat as ZF
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_ctx as RC
from models_v2.s18 import z_walk as ZW
from models_v2.s18 import z_walk_chk as K


@pytest.fixture(scope="module")
def world():
    return RC.make_ctx("CK"), ZW.load_z(), pd.read_parquet(ZW.RCK)


def test_nest_rckz_use_z_false_equals_rck_parquet_bitwise_all_months(world):
    ctx, (zm, _), rck = world
    assert K.nest_rck(ctx, zm, rck) == []


def test_mut_nest_rckz_wrong_lambda_is_caught(world, monkeypatch):
    ctx, (zm, _), rck = world
    orig = ZW.fit_heads
    monkeypatch.setattr(ZW, "fit_heads", lambda V, zts, var, use_z, lam, c, m: orig(V, zts, var, use_z, lam * 3.0, c, m))
    assert K.nest_rck(ctx, zm, rck) != []


def test_nest_rz_lambda_zero_and_f_equals_w5g(world):
    ctx, (zm, _), _ = world
    assert K.nest_rz(ctx, zm) == []


def test_mut_nest_rz_nonzero_correction_is_caught(world, monkeypatch):
    ctx, (zm, _), _ = world
    orig = ZW.head_lnc
    monkeypatch.setattr(ZW, "head_lnc", lambda ths, V, zts, var, use_z: orig(ths, V, zts, var, use_z) + (0.0 if use_z else 0.02))
    assert K.nest_rz(ctx, zm) != []


def layer_problems(ctx) -> list:
    """X1 и W5 — над C_Z, не над cfg8: C ≠ cfg8 в одном месяце меняет X1_R и F; значения = независимый E1.run / W5.walk над C."""
    x, bad, c8 = ctx.x, [], ctx.x["C"].to_numpy(float)
    zy = [np.zeros((len(x), 4)) for _ in ZW.SEEDS]
    parts = [(a, b, ZW.fit_month(ctx, zy, np.arange(0), ctx.dp[a:b], c8[a:b], "CKZ")) for a, b in RW.month_bounds(ctx)]
    a, b, r = parts[60]
    r["C"] = r["C"] * 1.3
    df = ZW.assemble(ctx, parts)
    e = E1.run(x["rv"].to_numpy(float), x["M"].to_numpy(float), df["C"].to_numpy(), x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy())
    w = W5.walk(x["d"], x["t_plus"], x["rv"], e["F"], ctx.gap, base_name="X1_V")
    bad += ["X1_R ≠ E1.run над C_Z"] * (float(np.max(np.abs(df["X1_R"] / e["F"] - 1.0))) > 1e-12)
    bad += ["F ≠ W5.walk над X1_R"] * (float(np.max(np.abs(df["F"] / w["F"].to_numpy() - 1.0))) > 1e-12)
    bad += ["слой не реагирует на C_Z"] * (not (df["X1_R"].to_numpy()[a:b] != x["F"].to_numpy(float)[a:b]).any())
    return bad


def test_layers_over_c_z_and_mutations_x1_over_cfg8_and_w5_over_x1_are_caught(world, monkeypatch):
    ctx, _, _ = world
    assert layer_problems(ctx) == []
    orig = RW.CL.layers
    with monkeypatch.context() as mp:
        mp.setattr(RW.CL, "layers", lambda x, c, gap: orig(x, x["C"].to_numpy(float), gap))
        assert layer_problems(ctx) != []
    with monkeypatch.context() as mp:
        def mut(x, c, gap):
            lay = orig(x, c, gap)
            lay["F"] = W5.walk(x["d"], x["t_plus"], x["rv"], x["F"], gap, base_name="X1_V")["F"].to_numpy()
            return lay
        mp.setattr(RW.CL, "layers", mut)
        assert layer_problems(ctx) != []


def test_pit_one_month_three_series_and_mutation_closed_plus_one_is_caught(monkeypatch):
    assert K.check_pit(1, 0) == []
    monkeypatch.setattr(YA, "closed_date", lambda tp, d: d)
    assert K.check_pit(1, 0) != []


def test_real_outputs_pass_check_file_and_mutations_are_caught(world):
    ctx, _, rck = world
    for key, f in ZW.OUTS.items():
        df = pd.read_parquet(ZW.Z_DIR / f)
        assert K.check_file(key, df, ctx.x, rck) == [], key
        assert K.check_file(key, df.iloc[:-1].reset_index(drop=True), ctx.x, rck) != []                              # D_M сокращён
        m = df.copy()
        m["z_ok"] = 0
        assert K.check_file(key, m, ctx.x, rck) != []                                                                 # день без z прошёл незамеченным
        m, pc = df.copy(), RW.names("CK") + ["thz_%d" % j for j in range(4)]
        i = int(np.flatnonzero((m["lam"] > 0.0).to_numpy())[0])
        assert m["lam"].iloc[i] > 0.0
        m.loc[m["refit_R"] == m["refit_R"].iloc[i], pc] = 0.0                                                         # θ ≡ 0 при λ* > 0
        assert any("θ ≡ 0" in b for b in K.check_file(key, m, ctx.x, rck)), key


def dm_ok(dm, x, zm, zs) -> bool:
    return len(dm) == 2198 == len(x) and bool((dm == pd.DatetimeIndex(x["d"])).all()) and all(len(zm[y][s]) == len(dm) == len(zs[y][s]) for y in zm for s in zm[y])


def test_zmaps_cover_all_2198_d_m_dates_and_shortened_d_m_is_caught(world):
    ctx, (zm, zs), _ = world
    dm = ZF.unpack(ZF.ZMAPS)[2]
    assert dm_ok(dm, ctx.x, zm, zs)
    assert not dm_ok(dm[:-1], ctx.x, zm, zs)
    assert not dm_ok(dm, ctx.x.iloc[:-1], zm, zs)
    assert not dm_ok(dm, ctx.x, {y: {s: z[:-1] for s, z in d.items()} for y, d in zm.items()}, zs)


def test_check_reads_only_and_mutation_check_that_writes_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(K.ZW, "Z_DIR", tmp_path)
    monkeypatch.setattr(K.RC, "make_ctx", lambda arch: None)
    monkeypatch.setattr(K.ZW, "load_z", lambda: (None, None))
    monkeypatch.setattr(K.ZR, "check", lambda *a, **k: [])
    assert all("изменил" not in b for b in K.check()) and list(tmp_path.iterdir()) == []
    monkeypatch.setattr(K.ZR, "check", lambda *a, **k: (tmp_path.joinpath("rz.parquet").write_bytes(b"x"), [])[1])
    assert any("изменил выходы" in b for b in K.check())
    assert K.tree(tmp_path) == {"rz.parquet": hashlib.sha256(b"x").hexdigest()}
