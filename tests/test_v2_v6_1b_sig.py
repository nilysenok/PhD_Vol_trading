"""V6-1b, G17: признаки подписи. Известные ответы: площадь Леви квадрата против часовой +1, по часовой −1, прямая 0; тождество Чена (уровни 1–2); обращение времени меняет знак всех A_ij и не трогает уровень 1 (основу S);
Σ³ (базис Линдона, 8 слов) = прямой расчёт логарифма подписи (произведения экспонент сегментов, независимый код) ≤ 1e-10 и = формула Бейкера–Кэмпбелла–Хаусдорфа для двух сегментов; Σ на реальных днях без NaN.
Мутации (через боевой `features`, как в `test_v2_v6_1_feat`): бары дня t⁺ в Σ дня d; среднее за 5 сессий с днём d+1 — строки ≤ d меняются при замене баров после d."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s13 import v6_1_feat as FE, v6_1b_sig as SG

SQ = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)


def make_bars(seed: int = 3, days: int = 70, nb: int = 53) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for k in range(days):
        day = pd.Timestamp("2016-01-04") + pd.offsets.BDay(k)
        for j in range(nb):
            o = px
            px = px * np.exp(rng.normal(0, 0.002))
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, max(o, px), min(o, px), px, float(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def daily_of(bars):
    return D.build_daily(bars, index_days=None)[0]


def corrupt_after(bars, day):
    b = bars.copy()
    late = b["begin"].dt.normalize() > day
    rng = np.random.default_rng(99)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, int(late.sum()))))
    b.loc[late, "close"], b.loc[late, "open"] = px, np.r_[px[:1], px[:-1]]
    b.loc[late, "volume"] = rng.integers(1, 5000, int(late.sum())).astype(float)
    return b


def leaks(day_idx: int = 45) -> bool:
    bars = make_bars()
    day = daily_of(bars).index[day_idx]
    b2 = corrupt_after(bars, day)
    a, c = SG.features(bars, daily_of(bars)), SG.features(b2, daily_of(b2))
    return any(not np.allclose(x[x.index <= day].to_numpy(), y[y.index <= day].to_numpy(), equal_nan=True, rtol=0, atol=1e-12) for x, y in zip(a[:2], c[:2]))


def test_levy_square_known_answer():
    assert np.isclose(SG.areas(SQ)[0, 1], 1.0)
    cw = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], float)
    assert np.isclose(SG.areas(cw)[0, 1], -1.0) and abs(SG.areas(np.tile([[0.3, -0.2]], (7, 1)))[0, 1]) < 1e-15


def test_chen_levels_1_2():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(9, 3)), rng.normal(size=(6, 3))
    sa, sb, sc = SG.sig3(a), SG.sig3(b), SG.sig3(np.vstack([a, b]))
    assert np.abs(sc[0] - (sa[0] + sb[0])).max() < 1e-12
    assert np.abs(sc[1] - (sa[1] + sb[1] + np.outer(sa[0], sb[0]))).max() < 1e-12
    s3 = sa[2] + sb[2] + np.einsum("ij,k->ijk", sa[1], sb[0]) + np.einsum("i,jk->ijk", sa[0], sb[1])
    assert np.abs(sc[2] - s3).max() < 1e-12                                          # Чен, уровень 3


def test_time_reversal_flips_areas_keeps_level_one():
    rng = np.random.default_rng(1)
    inc = rng.normal(size=(30, 5))
    a, b = SG.areas(inc), SG.areas(inc[::-1])
    assert np.abs(a + b).max() < 1e-12 and np.abs(inc.sum(0) - inc[::-1].sum(0)).max() < 1e-12
    assert np.abs(SG.sig3(inc)[1] + SG.sig3(inc)[1].T - SG.sig3(inc[::-1])[1] - SG.sig3(inc[::-1])[1].T).max() < 1e-12      # симметричная часть уровня 2 не меняется


def expo(v):
    """Усечённая экспонента сегмента в тензорной алгебре до уровня 3: (v, v⊗v/2, v⊗v⊗v/6)."""
    return v, np.outer(v, v) / 2, np.einsum("i,j,k->ijk", v, v, v) / 6


def mul(x, y):
    return (x[0] + y[0], x[1] + y[1] + np.outer(x[0], y[0]),
            x[2] + y[2] + np.einsum("ij,k->ijk", x[1], y[0]) + np.einsum("i,jk->ijk", x[0], y[1]))


def direct_log(inc):
    s = expo(inc[0])
    for v in inc[1:]:
        s = mul(s, expo(v))
    x1, x2, x3 = s
    return x1, x2 - np.outer(x1, x1) / 2, x3 - (np.einsum("i,jk->ijk", x1, x2) + np.einsum("ij,k->ijk", x2, x1)) / 2 + np.einsum("i,j,k->ijk", x1, x1, x1) / 3


def test_log3_equals_direct_product_of_exponentials():
    inc = np.random.default_rng(2).normal(size=(25, 3))
    for x, y in zip(SG.log3(inc), direct_log(inc)):
        assert np.abs(x - y).max() < 1e-10


def test_bch_two_segments_level3_known_answer():
    u, v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    l3 = SG.log3(np.array([u, v]))[2]
    ad = lambda x, y: np.einsum("i,j->ij", x, y) - np.einsum("i,j->ij", y, x)
    br = lambda x, y2: np.einsum("i,jk->ijk", x, y2) - np.einsum("ij,k->ijk", y2, x)       # [x, Y2] = x⊗Y2 − Y2⊗x
    ref = (br(u, ad(u, v)) + br(v, ad(v, u))) / 12.0
    assert np.abs(l3 - ref).max() < 1e-12


def test_lyndon_basis_and_coordinates():
    ws, B = SG.lyndon_matrix(3, 3)
    assert len(ws) == 8 and [w for w in ws][:2] == [(0, 0, 1), (0, 0, 2)] and np.linalg.matrix_rank(B) == 8
    c = np.arange(1.0, 9.0)
    back, res = SG.lyndon_coords((B @ c).reshape(3, 3, 3))
    assert np.abs(back - c).max() < 1e-12 and res < 1e-12
    inc = np.random.default_rng(4).normal(size=(40, 3))
    c1, res1 = SG.lyndon_coords(SG.log3(inc)[2])
    assert res1 < 1e-10                                                                   # настоящий логарифм подписи — элемент алгебры Ли
    c2, _ = SG.lyndon_coords(direct_log(inc)[2])
    assert np.abs(c1 - c2).max() < 1e-10
    ws_all = list(itertools.product(range(3), repeat=3))
    for j, w in enumerate(ws):                                                            # треугольность: слово w входит в P_w с коэффициентом 1
        assert B[ws_all.index(w), j] == 1.0


def test_flags_and_zero_channels():
    inc, f_rs, f_vol = SG.increments([0.001, 0.002, 0.003], [10.0, 20.0, 30.0])
    assert f_rs and not f_vol and np.all(inc[:, 3] == 0.0) and np.isclose(inc[:, 4].sum(), 1.0) and np.isclose(inc[:, 2].sum(), 1.0)
    inc, f_rs, f_vol = SG.increments([0.001, -0.002, 0.003], [0.0, 0.0, 0.0])
    assert f_vol and not f_rs and np.all(inc[:, 4] == 0.0)


def test_no_leak_synthetic_and_mutations_caught(monkeypatch):
    assert leaks() is False
    real_table, real_mean = SG.day_table, SG.five_mean
    monkeypatch.setattr(SG, "day_table", lambda bars: real_table(bars).shift(-1))                       # МУТАЦИЯ: площади дня t⁺ в строке дня d
    assert leaks() is True
    monkeypatch.setattr(SG, "day_table", real_table)
    monkeypatch.setattr(SG, "five_mean", lambda x, win=5: x.rolling(win, min_periods=win).mean().shift(-1))   # МУТАЦИЯ: среднее с днём d+1
    assert leaks() is True
    monkeypatch.setattr(SG, "five_mean", real_mean)
    assert leaks() is False


def test_real_sigma_no_nan_on_panel_days():
    bars, daily = FE.load_inputs()
    sigma, sig3, flags = SG.features(bars, daily)
    ix = daily.index[(daily.index >= "2015-07-07") & (daily.index <= "2026-02-26")]
    assert sigma.shape[1] == 12 and sig3.shape[1] == 8 and not sigma.loc[ix].isna().any().any() and not sig3.loc[ix].isna().any().any()
