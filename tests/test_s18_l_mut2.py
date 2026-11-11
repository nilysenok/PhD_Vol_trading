"""Ж-257 п. 3 / проверка S_L_VERIFY: мутации на РЕАЛЬНОМ пути b2 (train_pi), a ≠ 0 при λ* = 0, `--check` пишет (хэш выходов). Только monkeypatch (№ 43)."""
import hashlib
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import l_model as ML
from models_v2.s18 import l_walk as LW
from models_v2.s18 import l_walk_chk as K
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_reg as RR
from models_v2.s18 import r_walk as W

N = 2198


def _h(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _snap():
    s = {p.name: _h(p) for p in LW.L_DIR.glob("*.parquet")}
    s.update({RR.CKPT.name: _h(RR.CKPT), RR.TRIALS.name: _h(RR.TRIALS)})
    return s


@pytest.fixture(autouse=True)
def reals_untouched():
    before = _snap()
    yield
    assert _snap() == before


def data(seed=5, n=N):
    rng = np.random.default_rng(seed)
    return dict(rv=np.exp(rng.normal(size=n) * 0.4), lnb=rng.normal(size=n) * 0.2, pi=rng.uniform(0.1, 0.9, n), bg=np.zeros(n), lnc8=rng.normal(size=n) * 0.1)


def det_b2_real_path(A, T) -> list:
    """Реальный путь b2: stats_l / fit_kind обязаны брать π[i − 252] без переноса."""
    P = T[T >= 252]
    pt = A["pi"][P - 252]
    x = A["rv"][P] * np.exp(-ML.ln_ftrain(0.0, A["lnb"][P], pt, A["bg"][P], A["lnc8"][P]))
    c, m = RL.clip_stats(x, np.arange(len(P)))
    a = ML.fit_l(A["rv"][P], A["lnb"][P], pt, A["bg"][P], A["lnc8"][P], 1e-2, c, m)
    ok = ML.stats_l("shift252nr", A, T) == (c, m) and ML.fit_kind("shift252nr", A, T, 1e-2, c, m) == a
    try:    # i < 252: прошлого значения нет, отказ; перенос через конец запрещён
        ML.train_pi("shift252nr", A["pi"], np.arange(0, 300))
        ok = False
    except ValueError:
        pass
    ok = ok and np.array_equal(ML.train_idx("shift252nr", T), P)
    return [] if ok else ["b2: реальный путь ≠ π[i − 252] без переноса"]


def test_honest_b2_real_path_passes():
    assert det_b2_real_path(data(), np.arange(0, 900)) == []


def test_mutation_b2_real_path_roll_is_caught(monkeypatch):
    orig = ML.train_pi

    def mut(kind, pi, T):
        return np.roll(np.asarray(pi, float), ML.SHIFT)[np.asarray(T)] if kind == "shift252nr" else orig(kind, pi, T)
    monkeypatch.setattr(ML, "train_pi", mut)
    assert det_b2_real_path(data(), np.arange(0, 900))


def test_mutation_b2_real_path_shift_one_row_is_caught(monkeypatch):
    orig = ML.train_pi
    monkeypatch.setattr(ML, "train_pi", lambda k, pi, T: np.asarray(pi, float)[np.asarray(T) - 1] if k == "shift252nr" else orig(k, pi, T))
    assert det_b2_real_path(data(), np.arange(0, 900))


def zero_lam_month(fit):
    ctx, c8 = K.ctx_of(), K.ctx_of().x["C"].to_numpy(float)
    a, b = [ab for ab in W.month_bounds(ctx) if ab[0] > 1000][0]
    return fit(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], "shift252nr"), c8[a:b]


def det_a_zero(res, c0) -> list:
    return [] if res["a"] == 0.0 and np.array_equal(res["C"], c0) else ["a ≠ 0 или C ≠ cfg8 при λ* = 0"]


def test_a_zero_when_lambda_star_zero_honest_and_mutation_caught(monkeypatch):
    monkeypatch.setattr(LW, "choose_l", lambda ctx, P, k: (0.0, {lam: 0 for lam in W.LAMS}, {lam: [0.0] * W.NB for lam in W.LAMS}))
    honest = LW.fit_month_l
    res, c0 = zero_lam_month(honest)
    assert res["active"] and det_a_zero(res, c0) == []

    def mut(ctx, P, days, c, kind):    # мутант: обученный a остаётся в месяце с λ* = 0
        r = honest(ctx, P, days, c, kind)
        a = ML.fit_kind(kind, LW.arrays(ctx), P, W.LAMS[0], r["c"], r["m"])
        r.update(a=a, C=np.asarray(c, float) * np.exp(a))
        return r
    res2, c0 = zero_lam_month(mut)
    assert res2["a"] != 0.0 and det_a_zero(res2, c0)


def test_check_file_catches_nonzero_a_at_lambda_zero():
    df, ctx = pd.read_parquet(LW.L_DIR / "rl1b2.parquet"), K.ctx_of()
    z = df[df["active"] & (df["lam"] == 0.0)]
    assert not z.empty
    bad = df.copy()
    bad.loc[bad["refit_R"] == z["refit_R"].iloc[0], "a"] = 0.1
    assert K.check_file("rl1b2", bad, ctx)


def test_mut_check_that_writes_is_caught(tmp_path, monkeypatch):
    for f in LW.OUTS_L.values():
        shutil.copy(LW.L_DIR / f, tmp_path / f)
    monkeypatch.setattr(LW, "L_DIR", tmp_path)
    assert K.check() == []
    orig = K.repro

    def writing(var, df, k=3):
        res = orig(var, df, k)
        if var == "rl1b2":    # последний прочитанный выход: порча не ломает чтение, только хэш
            p = tmp_path / "rl1.parquet"
            p.write_bytes(p.read_bytes() + b"0")
        return res
    monkeypatch.setattr(K, "repro", writing)
    assert any("изменил выходы" in b for b in K.check())
