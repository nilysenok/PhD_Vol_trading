"""Ж-212 (HS5, G14, HS13): блоки признаков B на синтетике: PIT-коэффициенты HAR-J, вклады компонент, прошлое расхождение/ошибки — только закрытые;
главный тест обрезки (испорчено всё после d -> строки <= d побитно те же); мутация «закрыто + 1» ловится."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2 import har
from models_v2.s3 import resid
from models_v2.s8 import hs_feat, hs_leak

MP = 60


class SynF:
    def __init__(self, n=500, seed=0):
        rng = np.random.default_rng(seed)
        self.dates = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
        self.seg = np.zeros(n, int)
        self.rv = np.exp(rng.normal(0, 0.5, n))
        self.lnrv = np.log(self.rv)
        roll = lambda k: np.array([self.lnrv[max(0, i - k + 1):i + 1].mean() for i in range(n)])
        X = np.zeros((n, 12))
        X[:, 0], X[:, 1], X[:, 2], X[:, 3] = self.lnrv, roll(5), roll(22), np.log1p(np.abs(rng.normal(0, 0.3, n)))
        X[:, 7], X[:, 8] = np.exp(rng.normal(-1, 0.3, n)), np.exp(rng.normal(-1, 0.3, n))
        self.X = X
        self.X[:21, :4] = np.nan


def world(n=500):
    F = SynF(n)
    lb, beta = hs_feat.har_pit(F, MP)
    ml = {"XGB": np.exp(np.log(np.exp(lb)) + 0.3), "LGBM": np.exp(lb) * 0.8}
    return F, lb, beta, ml


def test_har_pit_matches_harj_ab():
    F = SynF(300)
    lb, _ = hs_feat.har_pit(F, MP)
    ref = resid.harj_ab(har.Panel(F.dates, F.seg, F.rv, F.X[:, :4]), min_pairs=MP)[1]
    assert np.allclose(lb, ref, equal_nan=True, rtol=1e-12, atol=0) and np.isfinite(lb).sum() > 100


def test_contribution_shares_sum_to_one_and_asym_sign():
    F, lb, beta, _ = world()
    cols, _ = hs_feat.block_r(F, beta)
    ok = np.isfinite(cols).all(axis=1)
    assert ok.sum() > 100 and np.allclose(np.abs(cols[ok, :4]).sum(axis=1), 1.0)
    assert np.allclose(cols[ok, 6], np.log(F.X[ok, 8] / F.X[ok, 7]), atol=1e-9)    # ln(RS−/RS+): знак не перепутан


def test_block_g_known_answer_constant_ratio():
    F, lb, _, _ = world()
    ml = np.exp(lb) / 2.0                      # HAR вдвое выше ML: δ = ln 2 всегда
    cols, _ = hs_feat.block_g(F, lb, ml)
    ok = np.isfinite(cols[:, 0])
    assert ok.sum() > 100 and np.allclose(cols[ok, 0], np.log(2.0)) and np.allclose(cols[ok, 1], np.log(2.0))


@pytest.mark.parametrize("d", [330, 420])
def test_truncation_blocks_unchanged_when_future_corrupted(d):
    F, lb, beta, ml = world()
    Fc, mlc = hs_leak.corrupt(F, ml, d)
    lbc, bc = hs_feat.har_pit(Fc, MP)
    a, b = hs_feat.blocks(F, lb, beta, ml["XGB"]), hs_feat.blocks(Fc, lbc, bc, mlc["XGB"])
    for k in hs_feat.BLOCKS:
        assert np.array_equal(a[k][0][:d + 1], b[k][0][:d + 1], equal_nan=True), k
    assert not np.array_equal(F.rv[d + 1:], Fc.rv[d + 1:])                       # порча действительно есть
    sa, sb = hs_feat.state_c(F, lb)[0], hs_feat.state_c(Fc, lbc)[0]
    assert np.array_equal(sa[:d + 1], sb[:d + 1], equal_nan=True)


def test_mutation_closed_plus_one_is_caught_on_errors_and_divergence():
    F, lb, beta, ml = world()
    d = 400
    Fc, mlc = hs_leak.corrupt(F, ml, d)
    lbc, bc = hs_feat.har_pit(Fc, MP)
    for fn, args, argsc in ((hs_feat.block_e, (F, lb), (Fc, lbc)), (hs_feat.block_g, (F, lb, ml["XGB"]), (Fc, lbc, mlc["XGB"]))):
        ok_a, ok_b = fn(*args)[0][:d + 1], fn(*argsc)[0][:d + 1]
        bad_a, bad_b = fn(*args, extra=1)[0][:d + 1], fn(*argsc, extra=1)[0][:d + 1]
        assert np.array_equal(ok_a, ok_b, equal_nan=True)
        assert not np.array_equal(bad_a, bad_b, equal_nan=True), fn.__name__     # с «закрыто + 1» порча будущего просачивается
