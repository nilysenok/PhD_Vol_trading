"""G22–G24, G26–G27: Г3/Г4/Г5 на синтетике с известным ответом; главный тест (прогноз на d не зависит от данных после d);
мутации m3 (closed включает d), m5 (q67 с днём d), m6 (веса EWMA наоборот), m7 (без клипа)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import metrics  # noqa: E402
from models_v2.s3 import post  # noqa: E402

N = 620


def grid(n=N):
    d = np.arange("2018-01-01", n, dtype="datetime64[D]")[:n] if False else np.datetime64("2018-01-01") + np.arange(n)
    return d, d + 1


def toy(n=N, seed=1):
    rng = np.random.default_rng(seed)
    lt = np.cumsum(rng.normal(0, 0.05, n)) * 0.3
    ln_rv = lt + rng.normal(0, 0.5, n)
    return {"ln_rv": ln_rv, "rv": np.exp(ln_rv), "ln_h": lt + 0.3 * rng.normal(size=n),
            "ln_b": ln_rv + 0.05 * rng.normal(size=n), "rvw": np.exp(lt + 0.1 * rng.normal(size=n)),
            "rvi": (rng.random(n) > 0.5).astype(float)}


def test_g3_w_to_one_when_b_exact_and_before_252_is_zero():
    d, td = grid()
    a = toy()
    m = post.closed_count(d, td)
    w = post.g3_weights(a["ln_b"], a["ln_h"], a["rv"], m)
    assert (w[:post.WIN] == 0).all() and np.median(w[post.WIN + 20:]) >= 0.9


def test_g3_w_zero_when_b_is_noise_and_tie_smaller_w():
    d, td = grid()
    a = toy()
    m = post.closed_count(d, td)
    rng = np.random.default_rng(5)
    w = post.g3_weights(rng.normal(3, 1, N), a["ln_rv"], a["rv"], m)  # H = точный ln RV, B — шум
    assert (w == 0).all()
    same = post.g3_weights(a["ln_h"], a["ln_h"], a["rv"], m)  # B == H: все w равны по качеству -> 0
    assert (same == 0).all()


def test_q67_excludes_day_d_and_mutation_caught():
    x = np.arange(300, dtype=float)
    q = post.q67_prior(x)
    assert q[260] == pytest.approx(np.quantile(x[8:260], 2 / 3)) and np.isnan(q[:252]).all()
    mut = np.quantile(x[9:261], 2 / 3)  # м5: окно с днём d
    assert mut != q[260]
    assert post.stress(np.array([np.nan, 1, 0]), np.array([1.0, 1, 1]), np.array([2.0, 2, np.nan])).tolist() == [False, True, False]


def test_logit_recovers_known_ab():
    d, td = grid(700)
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1, 700)
    ln_h, ln_c = np.zeros(700), np.full(700, 2.0)
    pi = 1 / (1 + np.exp(-(1.0 + 2.0 * z)))
    rv = np.exp(pi * ln_h + (1 - pi) * ln_c)
    m = post.closed_count(d, td)
    out, a, b = post.g4_logit(ln_h, ln_c, rv, z, np.zeros(700, bool), m, (d.astype("datetime64[M]")).astype(int))
    assert abs(a[-1] - 1.0) < 0.05 and abs(b[-1] - 2.0) < 0.05
    assert np.isnan(a[:post.WIN]).all()
    assert metrics.qlike(rv[-50:], np.exp(out[-50:])).mean() < 1e-3


def test_ewma_known_step_and_clip():
    m = np.arange(10)
    s2, rv = np.ones(10), np.r_[np.ones(5), np.full(5, 1.2)]
    k = post.ewma_k(rv, s2, m, 2.0)
    j = 9
    w = 0.5 ** (np.arange(j - 1, -1, -1) / 2.0)
    assert k[0] == 1.0 and k[j] == pytest.approx((w * rv[:j]).sum() / w.sum())
    assert k[j] > rv[:j].mean()  # свежие весят больше (м6: наоборот — было бы меньше)
    assert post.ewma_k(np.full(10, 9.0), s2, m, 2.0)[-1] == 1.5 and post.ewma_k(np.full(10, 0.01), s2, m, 2.0)[-1] == 0.7  # м7


def test_g5_only_in_stress_is_one_outside():
    s2, k, st = np.full(4, 2.0), np.full(4, 1.3), np.array([True, False, True, False])
    assert post.g5(s2, k, st).tolist() == [2.6, 2.0, 2.6, 2.0] and post.g5(s2, k).tolist() == [2.6] * 4


def pipe(a, d, td, mut=False):
    m = post.closed_count(d, td) + (1 if mut else 0)
    w = post.g3_weights(a["ln_b"], a["ln_h"], a["rv"], m)
    ln3 = w * a["ln_b"] + (1 - w) * a["ln_h"]
    q = post.q67_prior(a["rvw"])
    st = post.stress(a["rvi"], a["rvw"], q)
    z = np.log(a["rvw"]) - np.log(q)
    ln4, _, _ = post.g4_logit(a["ln_h"], ln3, a["rv"], z, st, m, d.astype("datetime64[M]").astype(int))
    k = post.ewma_k(a["rv"], np.exp(ln3), m, 10.0)
    return {"g3": ln3, "g4": post.g4_hard(a["ln_h"], ln3, st), "g4l": ln4, "g5": np.log(post.g5(np.exp(ln3), k, st))}


def corrupt(a, d, td, kk):
    b = {c: v.copy() for c, v in a.items()}
    fut_t, fut_d = td > d[kk], d > d[kk]
    b["rv"][fut_t] *= 7.0
    for c in ("ln_b", "ln_h", "rvw", "rvi"):
        b[c][fut_d] += 3.0 if c != "rvw" else 5.0
    return b


@pytest.mark.parametrize("kk", [300, 420, 560])
def test_main_future_data_after_d_does_not_change_forecast(kk):
    d, td = grid()
    a = toy()
    base, bad = pipe(a, d, td), pipe(corrupt(a, d, td, kk), d, td)
    for name in base:
        assert base[name][kk].tobytes() == bad[name][kk].tobytes(), name


def test_b7_defense_q90_excludes_d_off_before_252_and_main_invariance():
    x = np.arange(400, dtype=float)
    x[300] = 1e6
    dfn = post.b7_defense(x)
    assert not dfn[:252].any() and dfn[300] and dfn[260]  # растущий ряд выше q90 прошлого окна
    y = x.copy()
    y[301:] = 0.0  # порча после d не меняет защиту на d
    assert post.b7_defense(y)[300] == dfn[300] and post.b7_defense(y)[:301].tobytes() == dfn[:301].tobytes()
    z = np.arange(400, dtype=float)
    assert post.q_prior(z, 0.9)[300] == np.quantile(z[48:300], 0.9) != np.quantile(z[49:301], 0.9)  # м: окно с днём d
    h, b = np.full(3, 5.0), np.full(3, 2.0)
    assert post.b7_sigma2(b, h, np.array([True, False, True])).tolist() == [5.0, 2.0, 5.0]


def test_mut_closed_includes_d_is_caught():
    d, td = grid()
    a = toy()
    kk = 420
    base, bad = pipe(a, d, td, mut=True), pipe(corrupt(a, d, td, kk), d, td, mut=True)
    assert base["g3"][kk].tobytes() != bad["g3"][kk].tobytes() or base["g5"][kk].tobytes() != bad["g5"][kk].tobytes()
