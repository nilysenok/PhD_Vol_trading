"""Ж-214, остаток листа 3 (У22, У23): T2, T4, P1, P2 и Z3, Z7. Синтетика с известным ответом, главный тест обрезки, мутации: будущие пары в QLIKE-подгонке, профиль с днём d, окно профиля игнорируется."""
import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC, hv2_dec2 as D2, hv2_qlog as QG, hv2_sea2 as S2, hv2_sea_in as SI, hv2_world as W


def synth(n=700, seed=5, tail=420):
    rng = np.random.default_rng(seed)
    ln = np.zeros(n)
    for t in range(1, n):
        ln[t] = 0.85 * ln[t - 1] + rng.normal(0, 0.4)
    rv = np.exp(ln + 1.0)
    sh = rng.dirichlet([4, 8, 4], n)
    j = rv * rng.uniform(0, 0.35, n) * (rng.random(n) < 0.6)
    m10 = pd.DataFrame({"rv": rv, "j": j, "c": rv - j, "rs_pos": rv * 0.5, "rs_neg": rv * 0.5, "rv_open": rv * sh[:, 0], "rv_mid": rv * sh[:, 1], "rv_close": rv * sh[:, 2]},
                       index=pd.bdate_range("2018-01-01", periods=n))
    seg = np.zeros(n, int)
    X = np.column_stack([np.log(rv), np.log(DC.wmean(rv, seg, 5)), np.log(DC.wmean(rv, seg, 22)), np.log1p(j)])
    p = H.Panel(m10.index.to_numpy("datetime64[ns]"), seg, rv, X)
    return DC.build(W.World(p, rv ** 2, np.ones(n), H.eligible(p)[-tail:]), m10)


def test_identities_and_positivity():
    w = synth()
    for cfg in D2.CONFIGS:
        s2, comps, fb = D2.run(w, cfg)
        idn = DC.identities(s2, comps, fb)
        assert idn["min_sigma2"] > 0 and np.isfinite(s2).all() and idn["max_rel_sum_err"] < 1e-12 and idn["min_component"] > 0, (cfg, idn)


def test_main_truncation_all_new_configs():
    w = synth()
    for cfg in D2.CONFIGS:
        for k in (150, 300):
            assert DC.trunc_diff(lambda x, cfg=cfg: D2.sigma(x, cfg), w, int(w.el[k])) == 0.0, cfg


def test_qlike_fit_known_answer_and_mutation_future_pairs_caught():
    w = synth()
    F = np.log(w.comp["rv_open"])[:, None]
    g = np.exp(0.5 + 0.3 * F[:, 0])
    y = np.roll(g, 1)                                      # y_{t+1} = g(F_t): детерминированная связь, подгонка точная
    f = D2.forecast_q(w, F, y)
    ok = np.isfinite(f)
    assert ok.sum() > 100 and np.allclose(f[ok], g[w.el[ok]], rtol=1e-6)
    # мутация: пары до i+1 (цель завтрашнего дня в обучении) — обрезка обязана падать
    def future(x):
        out = np.full(len(x.el), np.nan)
        for a, i in enumerate(x.el):
            idx = W.pairs_idx(x.p.seg, F, min(int(i) + 2, len(x.p.rv) - 1))
            gam = QG.wfit(np.column_stack([np.ones(len(idx)), F[idx]]), x.p.rv[idx + 1], np.ones(len(idx)))
            out[a] = np.exp(gam[0] + gam[1] * F[i, 0])
        return out
    assert DC.trunc_diff(future, w, int(w.el[150])) > 0


def test_t2_includes_cross_parts_p1_uses_w_edge():
    w = synth()
    t2 = {k: F for k, F, _ in D2.spec(w, "T2")}
    p1 = {k: F for k, F, _ in D2.spec(w, "P1")}
    assert t2["rv_open"].shape[1] == 3 + 1 + 2 and p1["rv_open"].shape[1] == 3 + 1 + 1
    i = 300
    assert np.isclose(t2["rv_open"][i, 4], np.log(w.comp["rv_mid"][i])) and np.isclose(t2["rv_open"][i, 5], np.log(w.comp["rv_close"][i]))
    assert np.isclose(p1["rv_open"][i, 4], np.log(w.comp["rv_close"][i - 4:i + 1].mean()))


def test_sea2_builder_truncation_and_mutations():
    rng = np.random.default_rng(3)
    n = 700
    M2 = np.exp(rng.normal(0, 1, (n, 53))) * (1 + 3 * (np.arange(53) < 4))
    idx = pd.bdate_range("2018-01-01", periods=n)
    assert S2.trunc_builder(M2, idx, 400) == 0.0
    z = S2.build_extra(M2, idx)
    assert z["z3"].notna().sum() > 500 and z["z7"].notna().sum() > 500
    # окно действует: профиль 60 дней ≠ профиль 500 дней на хвосте
    assert not np.allclose(SI.median_profile(M2, 60)[-1], SI.median_profile(M2, 500)[-1])
    # мутация: профиль с днём d (lag = 0) обязан отличаться от строгого
    assert not np.allclose(SI.median_profile(M2, 60, lag=0)[-1], SI.median_profile(M2, 60)[-1])
    # будущий шок после d не меняет профиль d
    M2c = M2.copy()
    M2c[401:] *= 50
    assert np.array_equal(SI.median_profile(M2, 60)[:401], SI.median_profile(M2c, 60)[:401], equal_nan=True)
