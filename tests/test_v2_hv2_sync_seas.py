"""Ж-214: синхронные величины SBER–IMOEX (семиковариации, корреляция) и причинный сезонный профиль. Известный ответ, тождество
cov = pp + nn + mix, дыра в одном ряду, мутации: позиционное сопоставление баров; профиль с днём d (lag 0)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s9 import hv2_seas as Z, hv2_sync as Y, hv2_synth as T

DAYS = T.bdays("2016-06-01", 3)


def index_from(sb, k, drop=()):
    """IMOEX с логдоходностями бар-в-бар в k раз больше SBER; бары `drop` (минуты) удалены."""
    ix = sb.copy()
    p0 = 100.0
    for c in ("open", "high", "low", "close"):
        ix[c] = p0 * np.exp(k * np.log(ix[c] / p0))
    m = ix["begin"].dt.hour * 60 + ix["begin"].dt.minute
    return ix[~m.isin(drop)].reset_index(drop=True)


@pytest.fixture(scope="module")
def sber():
    return T.make([(d, dict(last_min=1120, amp=0.002)) for d in DAYS], seed=11)


def test_known_answer_proportional_and_opposite(sber):
    for k in (2.0, -1.5):
        t = Y.aggregate(Y.sync_returns(sber, index_from(sber, k)))
        assert (t["n_sync"] == 53).all()
        assert t["corr"].round(12).abs().eq(1.0).all() and np.sign(t["corr"].iloc[0]) == np.sign(k)
        assert t["beta"].iloc[0] == pytest.approx(1.0 / k, rel=1e-9)
        assert Y.identity(t) < 1e-12
    pos = Y.aggregate(Y.sync_returns(sber, index_from(sber, 2.0)))
    neg = Y.aggregate(Y.sync_returns(sber, index_from(sber, -1.5)))
    assert (pos["cov_mix"] == 0).all() and (neg["cov_pp"] == 0).all() and (neg["cov_nn"] == 0).all()
    assert neg["cov_mix"].iloc[0] == pytest.approx(neg["cov"].iloc[0], rel=1e-12)


def test_hole_in_one_series_keeps_synchronous_returns_and_positional_mutation_fails(sber):
    ix = index_from(sber, 2.0, drop=(840,))
    sr = Y.sync_returns(sber, ix)
    t = Y.aggregate(sr)
    assert (t["n_sync"] == 52).all() and t["corr"].round(12).eq(1.0).all()
    day = sber[sber["begin"].dt.normalize() == pd.Timestamp(DAYS[0])]
    main = day[(day["begin"].dt.hour * 60 + day["begin"].dt.minute).between(600, 1120)]
    ri = np.log(ix[ix["begin"].dt.normalize() == pd.Timestamp(DAYS[0])]["close"]).diff().to_numpy()
    rs = np.log(main["close"]).diff().to_numpy()
    n = min(len(ri), len(rs))
    assert abs(np.corrcoef(ri[1:n - 1], rs[1:n - 1])[0, 1]) < 0.999        # мутант: сопоставление по номеру, не по begin


def test_pit_profile_known_answer_and_normalisation():
    shape = np.linspace(3.0, 1.0, Z.NSLOT)
    M = np.tile(shape * 0.5, (120, 1))
    s = Z.pit_profile(M)
    assert np.isnan(s[:Z.MIN_DAYS]).all() and not np.isnan(s[Z.MIN_DAYS:]).any()
    assert np.allclose(np.nanmean(s[Z.MIN_DAYS:], axis=1), 1.0) and np.allclose(s[100], shape / shape.mean())
    assert np.allclose(Z.rv_sa(M, s)[Z.MIN_DAYS:], np.nansum(M, axis=1)[Z.MIN_DAYS:])      # чисто сезонный ряд: RV_sa = RV


def test_profile_excludes_day_d_and_lag0_mutation_is_caught():
    rng = np.random.default_rng(2)
    M = rng.uniform(0.5, 1.5, (130, Z.NSLOT))
    d = 100
    bad = M.copy()
    bad[d] *= 50.0
    assert np.array_equal(Z.pit_profile(M)[d], Z.pit_profile(bad)[d])           # день d и позже профиль d не меняют
    bad2 = M.copy()
    bad2[d + 5:] *= 9.0
    assert np.array_equal(Z.pit_profile(M)[:d + 5], Z.pit_profile(bad2)[:d + 5], equal_nan=True)
    assert not np.allclose(Z.pit_profile(M, lag=0)[d], Z.pit_profile(bad, lag=0)[d])      # мутант: день d в профиле
