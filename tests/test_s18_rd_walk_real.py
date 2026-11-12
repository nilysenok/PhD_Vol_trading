"""S-RCD-IMPL-2 (Ж-259), тестировщик: проход `rd_walk` на боевых входах — вложения Н1, Н2 (на 3 месяцах, не на 106), Н3 (на 3 месяцах), слои над новым C, состав выходов RD/*.parquet (W.G4), градиент заявленной потери (W.G5).
Боевые выходы только читаются; для мутаций проверок копии пишутся в tmp, `RD.RD_DIR` подменён."""
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gap as W5
from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_model_chk as MC
from models_v2.s18 import rd_walk as RD
from models_v2.s18 import rd_walk_chk as RDK
from models_v2.s18 import rd_walk_pit as RP
import test_s18_rd_walk_h as K

ORIG_BLOCKS, ORIG_STATS = W.blocks, W.stats_of
ORIG_LAYERS = W.CL.layers


def nest_h1_problems(ctx=None) -> list:
    ctx, bad, w5 = ctx or RDK.ctx_of(), [], pd.read_parquet(RL.W5G)
    for var in RD.VARS:
        df = RD.assemble(var, ctx, RDK.zero_parts(var, ctx))
        ok = [np.array_equal(df["C"].to_numpy(), ctx.x["C"].to_numpy(float)), np.array_equal(df["F"].to_numpy(), w5["F"].to_numpy()), np.array_equal(df["X1_R"].to_numpy(), ctx.x["F"].to_numpy(float))]
        bad += ["Н1 %s %s" % (var, ok)] * (not all(ok))
    return bad


def layer_problems(ctx=None) -> list:
    """C месяца 60 умножен на 1,3: X1_R и F обязаны совпасть с независимым E1.run / W5.walk над новым C и отличаться от X1 над cfg8."""
    from models_v2.s15 import de_e1_state as E1
    ctx, x, bad = ctx or RDK.ctx_of(), (ctx or RDK.ctx_of()).x, []
    parts = RDK.zero_parts("rcd", ctx)
    a, b, r = parts[60]
    r["C"] = r["C"] * 1.3
    df = RD.assemble("rcd", ctx, parts)
    e = E1.run(x["rv"].to_numpy(float), x["M"].to_numpy(float), df["C"].to_numpy(), x["s"].to_numpy(int), x["d"].to_numpy(), x["t_plus"].to_numpy())
    w = W5.walk(x["d"], x["t_plus"], x["rv"], e["F"], ctx.gap, base_name="X1_V")
    bad += ["X1_R ≠ E1.run над новым C"] * bool(np.max(np.abs(df["X1_R"] / e["F"] - 1.0)) > 1e-12)
    bad += ["F ≠ W5.walk над X1_R"] * bool(np.max(np.abs(df["F"] / w["F"].to_numpy() - 1.0)) > 1e-12)
    return bad + ["слой не реагирует на C"] * (not (df["X1_R"].to_numpy()[a:b] != x["F"].to_numpy(float)[a:b]).any())


def h2_problems(spec=("CK", "R")) -> list:
    """Н2 на 3 месяцах: проход R с формой CK = `R/rck.parquet` побитно (C, lam, c, m, w, G, θ)."""
    ctx, ref, bad = RDK.ctx_of(), pd.read_parquet(RDK.RCK), []
    for a, b, r in RD.walk("rcd", ctx, positions=set(MC.months(ctx.x)), spec=spec):
        eq = lambda u, v: np.array_equal(np.asarray(u, float), np.asarray(v, float), equal_nan=True)
        G = ref[["G_%s_b%d" % (k, i) for k in W.LK for i in (1, 2, 3)]].iloc[a].to_numpy(float)
        ok = [eq(r["C"], ref["C"].to_numpy()[a:b]), eq(r["lam"], ref["lam"].iloc[a]), eq(r["c"], ref["c"].iloc[a]), eq(r["m"], ref["m"].iloc[a]), eq(r["w"], ref[["w_" + k for k in W.LK]].iloc[a]), eq(r["G"].ravel(), G), eq(r["theta"], ref[list(CM.NAMES["CK"])].iloc[a])]
        bad += ["Н2 месяц %d: %s" % (a, ok)] * (not all(ok))
    return bad


def test_nest_h1_theta_zero_everywhere_gives_w5g_and_x1_bitwise_for_three_series():
    assert nest_h1_problems() == []


def test_layers_are_over_new_c_not_cfg8():
    assert layer_problems() == []


def test_mut_layers_over_cfg8_instead_of_new_c_is_caught(monkeypatch):
    monkeypatch.setattr(W.CL, "layers", lambda x, c, gap: ORIG_LAYERS(x, x["C"].to_numpy(float), gap))
    assert layer_problems() != []


def test_nest_h2_form_ck_in_r_pass_equals_rck_parquet_bitwise_on_three_months():
    assert h2_problems() == []


def test_mut_h2_form_cd_in_pass_is_caught():
    assert h2_problems(spec=("CD", "R")) != []


def test_mut_h2_training_block_includes_validation_is_caught(monkeypatch):
    monkeypatch.setattr(W, "blocks", lambda P, nb=W.NB: [(np.asarray(P)[: len(P) - W.VAL * (nb - b - 1)], V) for b, (T, V) in enumerate(ORIG_BLOCKS(P, nb))])
    assert h2_problems() != []


def test_mut_h2_clip_stats_over_all_rows_is_caught(monkeypatch):
    monkeypatch.setattr(W, "stats_of", lambda x0, T, mode: ORIG_STATS(x0, np.arange(len(x0)), mode))
    assert h2_problems() != []


@pytest.mark.parametrize("var,ref_p", [("cdf", RDK.CD), ("ckf", RDK.CK)])
def test_nest_h3_level_c_scdf_sckf_equal_saved_cd_ck_on_three_months(var, ref_p):
    """Н3: λ совпал, max отн. |C − C_сохр| ≤ 1e-4, на 3 месяцах (допуск не расширен)."""
    ctx, ref = RDK.ctx_of("C"), pd.read_parquet(ref_p)
    for a, b, r in RD.walk(var, ctx, positions=set(MC.months(ctx.x)), level="C"):
        assert r["lam"] == ref["lam"].iloc[a] and np.max(np.abs(r["C"] / ref["C"].to_numpy()[a:b] - 1.0)) <= 1e-4


def real_files() -> dict:
    return {v: pd.read_parquet(RD.RD_DIR / f) for v, f in RD.OUTS.items()}


@pytest.mark.parametrize("var", list(RD.VARS))
def test_w_g4_output_files_structure(var):
    assert RDK.check_file(var, real_files()[var]) == []


def edit(df, fn):
    d = df.copy()
    fn(d)
    return d


MUT_FILES = {
    "cdf_zero_lambda": ("cdf", lambda d: d.__setitem__("lam", d["lam"].where(d["lam"].isna() | (d.index % 2 == 1), 0.0))),
    "cdf_clip_c": ("cdf", lambda d: d.__setitem__("c", np.where(d["active"], 5.0, d["c"]))),
    "cdf_m_not_one": ("cdf", lambda d: d.__setitem__("m", np.where(d["active"], 0.9, d["m"]))),
    "ckf_inclusion_630": ("ckf", lambda d: d.__setitem__("active", d["active"] & (d["d"] >= pd.Timestamp("2020-08-03")))),
    "rcd_inclusion_2018_07": ("rcd", lambda d: d.__setitem__("active", d["active"] | ((d["d"] >= pd.Timestamp("2018-07-02")) & (d["d"] < pd.Timestamp("2018-08-01"))))),
    "rcd_theta_nonzero_at_lambda_zero": ("rcd", lambda d: d.__setitem__("th0", np.where((d["lam"] == 0.0), 0.05, d["th0"]))),
    "rcd_c_not_rck": ("rcd", lambda d: d.__setitem__("c", d["c"] * 1.0001)),
    "cdf_c_not_cfg8_times_form": ("cdf", lambda d: d.__setitem__("C", d["C"] * np.where(d["active"], 1.001, 1.0))),
}


@pytest.mark.parametrize("name", list(MUT_FILES))
def test_mut_w_g4_output_file_mutation_is_caught(name):
    var, fn = MUT_FILES[name]
    assert RDK.check_file(var, edit(real_files()[var], fn)) != []


def test_mut_dm_shortened_output_is_caught():     # D_M сокращён: 2198 дат в каждом parquet
    df = real_files()["rcd"].iloc[:-5]
    try:
        bad = RDK.check_file("rcd", df)
    except Exception as e:     # падение проверки на урезанном выходе — тоже отказ
        bad = [repr(e)]
    assert bad != []


def copy_rd(tmp_path):
    for f in RD.OUTS.values():
        shutil.copy(RD.RD_DIR / f, tmp_path / f)
    return tmp_path


def test_check_changes_no_output_file(tmp_path, monkeypatch):     # № 123
    monkeypatch.setattr(RD, "RD_DIR", copy_rd(tmp_path))
    before = RDK.tree(tmp_path)
    assert RDK.check() == [] and RDK.tree(tmp_path) == before and len(before) == 3


def test_mut_check_that_writes_is_caught(tmp_path, monkeypatch):     # № 123
    monkeypatch.setattr(RD, "RD_DIR", copy_rd(tmp_path))
    orig = RDK.check_file
    monkeypatch.setattr(RDK, "check_file", lambda var, df: (df.to_parquet(tmp_path / "extra.parquet"), orig(var, df))[1])
    assert any("изменил" in b for b in RDK.check())


def test_w_g5_gradient_of_declared_loss_small_and_level_c_large():
    assert RP.check_level() == []


def test_mut_sckf_copy_of_sck_level_c_is_caught_by_level_check(tmp_path, monkeypatch):
    copy_rd(tmp_path)
    ck, d = pd.read_parquet(RDK.CK), pd.read_parquet(tmp_path / "ckf.parquet")
    for c in list(CM.NAMES["CK"]) + ["lam"]:
        d[c] = ck[c].to_numpy()
    d.to_parquet(tmp_path / "ckf.parquet")
    monkeypatch.setattr(RD, "RD_DIR", tmp_path)
    assert any("ckf" in b for b in RP.check_level())
