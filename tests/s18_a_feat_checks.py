"""Проверки S-A-IMPL-1 как функции: честный код их проходит, подложенные ошибки (test_s18_a_feat_mut.py) — нет. Не тест."""
import copy

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import a_feat as A
from s18_a_guard import synth_r2, two_seg_dates


def synth2(T=130, seed=0, miss=0.05):
    """Два сегмента (остановка 2022), пропущенные бары; rv, rv_w, rv_m — как `daily.add_windows`."""
    dates = two_seg_dates(60, T - 60)
    F = A.Feat.from_r2(synth_r2(T, seed, miss), dates)
    F.seg = (dates > pd.Timestamp("2022-03-24")).astype(int)
    F.pos = pd.Series(F.seg).groupby(F.seg).cumcount().to_numpy()
    return F


def theta_rand(seed=1):
    g = np.random.default_rng(seed)
    z = torch.float64
    return dict(omega=torch.as_tensor(g.uniform(0.2, 0.9, 3), dtype=z), psi=torch.as_tensor(g.uniform(-3, 3, (3, 2)), dtype=z),
                c=torch.as_tensor(g.normal(0, 0.4, (3, 4)), dtype=z))


def ln_s(F, theta, rows=None):
    rows = np.arange(len(F.dates)) if rows is None else np.asarray(rows)
    with torch.no_grad():
        return A.ln_S(F, theta, A.lag_idx(F, rows)).numpy()


def check_eq(F):
    """S(HAR, b=1) = rv / rv_w / rv_m (1e-12 отн.) на всех строках; усечение по сегменту — независимый эталон: среднее своих строк сегмента."""
    ls = ln_s(F, A.har_theta())
    for k, ref in enumerate((F.rv, F.rv_w, F.rv_m)):
        ok = np.isfinite(ref)
        assert (ok == (F.pos + 1 >= A.RECT[k])).all()
        assert np.abs(np.exp(ls[ok, k]) / ref[ok] - 1).max() <= 1e-12
        assert np.abs(ls[ok, k] - np.log(ref[ok])).max() <= 1e-12
        for i in range(len(F.rv)):
            m = min(A.RECT[k], F.pos[i] + 1)
            assert abs(np.exp(ls[i, k]) / F.rv[i - m + 1:i + 1].mean() - 1) <= 1e-12, (k, i)
    assert np.abs(A.rv_b_exact(F, np.ones((3, A.NS)))[:, 0] / F.rv - 1).max() <= 1e-12   # побитно — только на реальном ряду (тот же путь суммирования)


def check_pit(F, d=80, seed=5):
    """S строк ≤ d не зависит от баров сессий > d (мусор, NaN, другая маска) при случайных (a, b)."""
    th = theta_rand()
    rows = np.arange(d + 1)
    base = ln_s(F, th, rows)
    g = np.random.default_rng(seed)
    r2 = F.r2.copy()
    r2[d + 1:] = np.abs(g.normal(size=r2[d + 1:].shape)) * 1e-3
    r2[d + 1:][g.random(r2[d + 1:].shape) < 0.3] = np.nan
    F2 = A.Feat.from_r2(r2, F.dates)
    F2.seg, F2.pos = F.seg, F.pos
    assert np.array_equal(ln_s(F2, th, rows), base)


def check_bmean(F, rv_b_fn=None):
    """mean_present b = 1 в каждой сессии: при r² ≡ 1 на присутствующих барах RV_b = 1e4·n при любых c."""
    fn = rv_b_fn or A.rv_b
    one = np.where(F.pres, 1.0, np.nan)
    G = A.Feat.from_r2(one * 1e-4, F.dates)          # r² = 1e-4 → RV = 1e4·1e-4·n = n
    for s in (3, 4):
        E = A.slot_exp(theta_rand(s)["c"])
        with torch.no_grad():
            got = np.asarray(fn(G, E))
        assert np.abs(got / G.n[:, None] - 1).max() <= 1e-12


def check_simplex(F=None):
    """a ≥ 0, Σa = 1 (1e-12) для всех ω ∈ [0, 1], ψ ∈ [−8, 8], любых усечений; ω = 0 ⇒ a = прямоугольник HAR."""
    F = F or synth2()
    g = np.random.default_rng(7)
    lag = A.lag_idx(F, np.arange(len(F.dates)))
    for _ in range(5):
        for k, (ix, ok) in enumerate(lag):
            w, psi = float(g.uniform(0, 1)), torch.as_tensor(g.uniform(-8, 8, 2))
            a = A.lag_weights(ok, k, torch.as_tensor(w, dtype=torch.float64), psi).numpy()
            assert (a >= 0).all() and np.abs(a.sum(1) - 1).max() <= 1e-12
    for k, (ix, ok) in enumerate(lag):
        z = torch.float64
        a0 = A.lag_weights(ok, k, torch.as_tensor(0.0, dtype=z), torch.zeros(2, dtype=z)).numpy()
        m = np.minimum(A.RECT[k], F.pos + 1)
        want = (np.arange(A.LAGS[k])[None, :] < m[:, None]) / m[:, None]
        assert np.abs(a0 - want).max() <= 1e-15


def bars_missing_first(day_missing=1):
    """3 сессии подряд по сетке 10 минут; в сессии с номером `day_missing` нет бара 10:00 (52 бара)."""
    g = np.random.default_rng(2)
    rows, c = [], 100.0
    for i, day in enumerate(pd.bdate_range("2019-03-04", periods=3)):
        for j in range(53):
            if i == day_missing and j == 0:
                continue
            o, c = c, c * np.exp(0.001 * g.normal())
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, c))
    return pd.DataFrame(rows, columns=["begin", "open", "close"])


def check_slots(F, day=1):
    """Вес привязан ко времени: у сессии без бара 10:00 пропуск — слот 0, а не «последний»; слот = (минуты − 600)/10."""
    assert not F.pres[day, 0] and F.pres[day, 1:].all() and F.n[day] == 52
    assert F.pres[day - 1].all()
    assert (F.lslot[F.lrow == day] == np.arange(1, 53)).all()


def pack_left(F):
    """Мутант: бары нумеруются по порядку внутри сессии (слот = номер бара), а не по времени."""
    G = copy.copy(F)
    r2 = np.full_like(F.r2, np.nan)
    for i in range(len(F.r2)):
        v = F.r2[i][F.pres[i]]
        r2[i, :len(v)] = v
    G.r2, G.pres, G.n, G._t = r2, np.isfinite(r2), F.n, None
    G.lslot = np.concatenate([np.arange(c) for c in F.n])
    return G
