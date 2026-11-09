"""V7-1a-E, G14: представление z. encode — известный ответ (среднее и последнее по барам с маской); главные компоненты — только по z дат < 01.01.Y;
z дня d не зависит от баров дней > d. Мутации: компоненты по всей выборке, окно со сдвигом +1 день — ловятся."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s14 import v7_0_seq as SQ
from models_v2.s14 import v7_1e_net as NET
from models_v2.s14 import v7_1e_pre as P
from models_v2.s14 import v7_1e_rep as R
from models_v2.s14 import v7_1e_syn as SYN

Y = 2020


class Stub(torch.nn.Module):
    """hidden = первые 8 каналов входа: известный ответ для encode."""
    def hidden(self, x):
        return x[:, :8, :]


def test_encode_known_answer_mean_and_last_over_present_bars():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2, NET.SLOTS, NET.K))
    M = rng.random((3, 2, NET.SLOTS)) > 0.3
    M[2] = False                                               # день без баров -> нули
    M[1, 1, -3:] = False                                       # последний бар окна — не слот 52
    z = R.encode(Stub(), X, M)
    assert z.shape == (3, 16)
    for n in (0, 1):
        pos = np.flatnonzero(M[n].reshape(-1))
        h = (X[n] * M[n][..., None]).reshape(-1, NET.K)[:, :8]
        assert np.allclose(z[n, :8], h[pos].mean(0), atol=1e-6) and np.allclose(z[n, 8:], h[pos[-1]], atol=1e-6)
    assert (z[2] == 0).all()


def zs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    return np.einsum("ij,jk->ik", rng.normal(size=(n, 16)), rng.normal(size=(16, 16))), dates


def test_pca_uses_only_dates_before_year():
    z, dates = zs()
    p = R.fit_pca_past(z, dates, 2019)
    z2 = z.copy()
    z2[dates >= pd.Timestamp(2019, 1, 1)] = np.random.default_rng(9).normal(size=(int((dates >= pd.Timestamp(2019, 1, 1)).sum()), 16)) * 100
    q = R.fit_pca_past(z2, dates, 2019)
    for k in ("mu", "sd", "comps", "ssd"):
        assert np.array_equal(p[k], q[k])
    assert p["n_past"] == int((dates < pd.Timestamp(2019, 1, 1)).sum())


def test_mutation_pca_on_whole_sample_caught(monkeypatch):
    z, dates = zs()
    z2 = z.copy()
    z2[dates >= pd.Timestamp(2019, 1, 1)] *= 100
    monkeypatch.setattr(R, "_past_mask", lambda d, y: np.ones(len(d), bool))
    a, b = R.fit_pca_past(z, dates, 2019), R.fit_pca_past(z2, dates, 2019)
    assert not np.array_equal(a["comps"], b["comps"])


def test_pca_known_answer_scores_standardized_and_orthonormal():
    z, dates = zs(600, 1)
    p = R.fit_pca_past(z, dates, 2019)
    past = z[dates < pd.Timestamp(2019, 1, 1)]
    s = R.transform_pca(p, past)
    assert s.shape == (len(past), 4) and np.allclose(s.mean(0), 0, atol=1e-9) and np.allclose(s.std(0), 1, atol=1e-9)
    assert np.allclose(p["comps"] @ p["comps"].T, np.eye(4), atol=1e-9) and np.allclose(np.corrcoef(s.T), np.eye(4), atol=1e-9)
    assert (np.abs(p["comps"]).argmax(1) == np.abs(p["comps"]).argmax(1)).all() and (p["comps"][np.arange(4), np.abs(p["comps"]).argmax(1)] > 0).all()
    with pytest.raises(ValueError):
        R.fit_pca_past(z, dates, 2018)                         # прошлого нет


def build_pack(bars_days, W=5):
    bars, days = bars_days
    return P.make_pack(SQ.build_store(bars, days), W)


def perturb_after(bars, d_end, seed=1):
    b = bars.copy()
    m = b["begin"] >= pd.Timestamp(d_end) + pd.Timedelta(days=1)
    rng = np.random.default_rng(seed)
    for c in ("open", "close"):
        b.loc[m, c] = b.loc[m, c] * rng.uniform(0.5, 2.0, m.sum())
    b.loc[m, "high"] = b.loc[m, ["open", "high", "low", "close"]].max(axis=1)
    b.loc[m, "low"] = b.loc[m, ["open", "high", "low", "close"]].min(axis=1)
    b.loc[m, "volume"] = b.loc[m, "volume"] * rng.integers(2, 50, m.sum())
    b.loc[m, "value"] = b.loc[m, "volume"] * b.loc[m, "close"]
    return b


@pytest.fixture(scope="module")
def world():
    bars, days = SYN.make_bars(140, 0, "2019-08-01")
    di = int(np.flatnonzero(days >= pd.Timestamp(2020, 1, 6))[0])
    return bars, days, di


def z_upto(bars, days, di, model):
    pk = build_pack((bars, days))
    rows = np.arange(len(days))
    return R.z_pit(model, pk, rows, Y)[0][: di + 1]


def trained(bars, days):
    return P.train_encoder("tcn", 5, Y, 0, build_pack((bars, days)), max_epochs=2, min_rows=30)


def test_z_of_day_d_ignores_bars_after_d(world):
    bars, days, di = world
    m = trained(bars, days)
    a = z_upto(bars, days, di, m)
    b = z_upto(perturb_after(bars, days[di]), days, di, m)
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_z_changes_if_bars_of_day_d_change(world):
    """Не вырожденно: порча баров ДО d (включая день d) меняет z дня d."""
    bars, days, di = world
    m = trained(bars, days)
    a = z_upto(bars, days, di, m)
    b = z_upto(perturb_after(bars, days[di - 1]), days, di, m)
    assert not np.array_equal(a[di], b[di]) and np.array_equal(a[:di], b[:di])


def test_mutation_window_shifted_plus_one_day_caught(world, monkeypatch):
    bars, days, di = world
    m = trained(bars, days)
    monkeypatch.setattr(SQ, "window_rows", lambda i, W: np.arange(i - W + 2, i + 2))
    a = z_upto(bars, days, di, m)
    b = z_upto(perturb_after(bars, days[di]), days, di, m)
    assert not np.array_equal(a, b)


def test_z_pit_returns_four_components_with_year_pca(world):
    bars, days, di = world
    pk = build_pack((bars, days))
    m = P.train_encoder("cnn", 5, Y, 0, pk, max_epochs=2, min_rows=30)
    z, p = R.z_pit(m, pk, np.arange(len(days)), Y)
    assert z.shape == (len(days), 4) and p["year"] == Y and p["n_past"] == int((pd.DatetimeIndex(days) < pd.Timestamp(Y, 1, 1)).sum())
    past = pd.DatetimeIndex(days) < pd.Timestamp(Y, 1, 1)
    assert np.allclose(z[past].mean(0), 0, atol=1e-8) and np.allclose(z[past].std(0), 1, atol=1e-8)
