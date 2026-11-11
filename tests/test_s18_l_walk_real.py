"""S-L-IMPL-1/2 на боевых выходах `L/*.parquet`: a = 0 ⇒ W5-G побитно, S-RL1a ≡ W5-G, слои на настоящем π, реестр раньше прогноза, `--check` только читает."""
import os

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import l_reg as LR
from models_v2.s18 import l_reg2 as LR2
from models_v2.s18 import l_walk as LW
from models_v2.s18 import l_walk_chk as K
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_reg as RR


def frames():
    return {v: pd.read_parquet(LW.L_DIR / (v + ".parquet")) for v in LW.VARS_L}


def test_control_a_is_w5g_bitwise_and_zero_a_months_have_C_equal_cfg8():
    f, w5 = frames(), pd.read_parquet(RL.W5G)
    assert np.array_equal(f["rl1a"]["F"].to_numpy(), w5["F"].to_numpy()) and (f["rl1a"]["a"] == 0).all() and (f["rl1a"]["lam"].dropna() == 0).all()
    for k, df in f.items():
        z = (df["a"] == 0).to_numpy()
        assert np.array_equal(df["C"].to_numpy()[z], df["cfg8"].to_numpy()[z]) and (np.abs(df["a"]) <= 0.5).all()
        assert (df.loc[~z, "lam"] > 0).all()


def test_forced_lambda_zero_gives_w5g_bitwise_on_real_layers():
    df, w5 = K.forced_zero("shift252nr"), pd.read_parquet(RL.W5G)
    assert np.array_equal(df["F"].to_numpy(), w5["F"].to_numpy()) and np.array_equal(df["X1_R"].to_numpy(), C.load()["F"].to_numpy(float))


def test_layers_run_on_true_pi_state_not_on_training_pi():
    x, f = C.load(), frames()
    pi = x["pi"].to_numpy(float)
    assert np.array_equal(f["rl1a"]["pi"].to_numpy(float), pi)    # C = cfg8 ⇒ слой X1 даёт исходный π
    for k in ("rl1", "rl1b", "rl1b2"):
        p = f[k]["pi"].to_numpy(float)
        assert np.corrcoef(p, pi)[0, 1] > 0.9 and np.corrcoef(p, np.roll(pi, 252))[0, 1] < 0.5 and p.std() > 0.05, k    # состояние слоя — настоящий π, не сдвинутый и не константа


def test_b2_never_trained_on_first_252_rows_and_differs_from_cyclic_b():
    f = frames()
    assert not np.array_equal(f["rl1b"]["a"].to_numpy(), f["rl1b2"]["a"].to_numpy())
    ctx, A = K.ctx_of(), LW.arrays(K.ctx_of())
    a0 = [a for a, _ in LW.W.month_bounds(ctx)][30]
    P = RL.pool_of("CK", ctx.D, ctx.x, a0)
    assert (P < 252).any()
    import models_v2.s18.l_model as ML
    assert ML.stats_l("shift252nr", A, P) == ML.stats_l("shift252nr", A, P[P >= 252])


def test_registry_before_forecast_for_every_series_and_check_reads_only():
    import datetime as dt
    cp = RR.json.loads(RR.CKPT.read_text(encoding="utf-8")) if hasattr(RR, "json") else __import__("json").loads(RR.CKPT.read_text(encoding="utf-8"))
    for key, ids in ((LR.KEY, ("rl1", "rl1a", "rl1b")), (LR2.KEY, ("rl1b2",))):
        reg = RR._t(cp[key]["registered_at"])
        for v in ids:
            assert reg < dt.datetime.fromtimestamp(os.stat(LW.L_DIR / (v + ".parquet")).st_mtime), v
    assert LR.check() == [] and LR2.check() == []
    h0 = RR.__dict__.get("x") or K.RC.tree(LW.L_DIR)
    assert K.check_nest() == [] and K.RC.tree(LW.L_DIR) == h0
