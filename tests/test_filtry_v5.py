"""Фильтры входа v5 (Ж-127): известные ответы, сверка RSI/наклона с формулами источника, мутации."""
import numpy as np
import pytest

from fastbt import filtry_v5 as F
from fastbt.indicators import compute_base

RNG = np.random.default_rng(7)
CL = 100 + np.cumsum(RNG.normal(0, 1, 400))
HI, LO = CL + RNG.uniform(0.1, 1, 400), CL - RNG.uniform(0.1, 1, 400)


def prichinnost(fn, x, ds=(60, 120, 250, 399)):
    """Маска на обрезке x[:d+1] в точке d == маска на полном ряду: данные > d не влияют."""
    full = fn(x)
    return all(fn(x[:d + 1])[d] == full[d] for d in ds)


def test_rsi_ruchnoi_primer_15_tochek():
    c = [10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17, 16]
    r = F.rsi_wilder(c, 14)
    assert np.isnan(r[:14]).all()
    assert r[14] == pytest.approx(65.0, abs=1e-12)          # gains 13, losses 7 -> 100*13/20
    ag, al = 13 / 14, 7 / 14
    r2 = F.rsi_wilder(c + [17], 14)[15]
    ag, al = (ag * 13 + 1) / 14, al * 13 / 14
    assert r2 == pytest.approx(100 - 100 / (1 + ag / al), abs=1e-12)


def test_rsi_monotonny_rost_100_i_nan_do_period():
    r = F.rsi_wilder(np.arange(1.0, 30.0), 14)
    assert r[14:].min() == 100.0 and np.isnan(F.rsi_wilder([1, 2, 3], 14)).all()


def test_maska_rsi_kontrtrend_i_po_trendu_i_nan():
    rsi = np.array([np.nan, 20.0, 50.0, 80.0])
    lo, sh = F.maska_rsi(rsi, 30)
    assert lo.tolist() == [False, True, False, False] and sh.tolist() == [False, False, False, True]
    lo, sh = F.maska_rsi(rsi, 30, po_trendu=True)
    assert lo.tolist() == [False, False, False, True] and sh.tolist() == [False, True, False, False]
    assert all(m.all() for m in F.maska_rsi(rsi, 0))


def test_mutatsiya_inversiya_rsi_lovitsya():
    rsi = np.array([20.0, 80.0])
    lo, _ = F.maska_rsi(rsi, 30)
    assert lo.tolist() == [True, False] and (~lo).tolist() != lo.tolist()
    assert lo.tolist() != F.maska_rsi(100 - rsi, 30)[0].tolist()


def test_atr_pctl_izvestny_poryadok():
    up, down = np.arange(1.0, 101.0), np.arange(100.0, 0.0, -1.0)
    m_up, m_dn = F.maska_atr_pctl(up, 60, 50), F.maska_atr_pctl(down, 60, 50)
    assert not m_up[:50].any() and m_up[50:].all()          # min_n = 51: до 51-го значения запрет; растущий ATR > медианы окна
    assert not m_dn.any()                                    # убывающий: ATR[d] — минимум окна
    assert not F.maska_atr_pctl(np.full(100, 3.0), 60, 50).any()   # равенство — не строго больше
    assert F.maska_atr_pctl(down, 60, 0).all()               # pct = 0 — выкл


def test_atr_pctl_okno_vklyuchaet_d_formula():
    atr = np.abs(RNG.normal(1, .3, 300))
    m = F.maska_atr_pctl(atr, 100, 40)
    for d in (60, 150, 299):
        w = atr[max(0, d - 99):d + 1]
        assert m[d] == (atr[d] > np.percentile(w, 40))
    assert atr[:50].size and not m[:50].any()


def test_atr_pctl_prichinnost():
    atr = np.abs(RNG.normal(1, .3, 400))
    assert prichinnost(lambda x: F.maska_atr_pctl(x, 100, 40), atr)


def test_mutatsiya_atr_pctl_po_vsemu_ryadu_lovitsya():
    def po_vsemu_ryadu(x):                                   # как в daily_upper_bound.py:313-314
        va = x[~np.isnan(x)]
        return ~np.isnan(x) & (x > np.percentile(va, 40))
    atr = np.abs(RNG.normal(1, .3, 400))
    assert not prichinnost(po_vsemu_ryadu, atr)
    assert prichinnost(lambda x: F.maska_atr_pctl(x, 100, 40), atr)


def test_mutatsiya_atr_pctl_d_plus_1_lovitsya():
    atr = np.abs(RNG.normal(1, .3, 400))
    base = lambda x: F.maska_atr_pctl(x, 100, 40)
    sdvig = lambda x: np.append(base(x)[1:], False)          # mask[d] := mask[d+1]
    assert prichinnost(base, atr) and not prichinnost(sdvig, atr)


def test_mutatsiya_atr_pctl_okno_bez_d_lovitsya():
    atr = np.abs(np.random.default_rng(3).normal(1, .3, 5000))
    thr_bez_d = np.roll(F.atr_veroyatnost(atr, 100, 40), 1)  # порог по окну d-100..d-1, без текущего дня
    mut = atr > thr_bez_d
    assert (mut[100:] != F.maska_atr_pctl(atr, 100, 40)[100:]).any()   # маска отличается (на 5000 днях — заведомо)


def test_naklon_linejny_ryad_izvestny():
    c = 100 + 2.0 * np.arange(60)
    s = F.naklon_ma(c, 20, 10, "sma")
    assert np.isnan(s[28]) and not np.isnan(s[29])           # первое значение: MA с 19-го, лаг +10 -> 29
    sma = lambda i: 100 + 2 * (i - 9.5)
    assert s[40] == pytest.approx(20 / sma(30), abs=1e-12)
    se = F.naklon_ma(c, 20, 10, "ema")
    assert (se[30:] > 0).all()
    up, dn = F.maska_naklon(s, 0.01, "po_trendu")
    assert up[40] and not dn[40]
    assert not F.maska_naklon(s, 0.01, "abs")[0][40] and F.maska_naklon(np.zeros(5), 0.01)[0][0] == True
    with pytest.raises(ValueError):
        F.maska_naklon(s, 0.01, "xx")


def test_naklon_sverka_s_istochnikom_compute_base():
    ind = compute_base(CL, HI, LO, CL, np.full(400, 1e6))
    s = F.naklon_ma(CL, 20, 10, "sma")
    m = ~np.isnan(ind["sma20_slope10"])
    assert (m == ~np.isnan(s)).all()
    assert np.abs(s[m] - ind["sma20_slope10"][m]).max() <= 1e-12
    assert (F.maska_naklon(s, 0.01)[0] == (m & (np.abs(ind["sma20_slope10"]) < 0.01))).all()


def test_naklon_prichinnost_i_mutatsii():
    base = lambda x: F.maska_naklon(F.naklon_ma(x, 20, 10), 0.01, "po_trendu")[0]
    assert prichinnost(base, CL)
    assert not prichinnost(lambda x: np.append(base(x)[1:], False), CL)
    s = F.naklon_ma(100 + 2.0 * np.arange(60), 20, 10)
    assert F.maska_naklon(s, 0.01, "po_trendu")[0][40] != F.maska_naklon(-s, 0.01, "po_trendu")[0][40]   # знак


def test_szhatie_padayushchaya_volatilnost():
    i = np.arange(400)
    c_pad = 100 + 5 * 0.985 ** i * (-1.0) ** i
    c_rost = 100 + 0.1 * 1.01 ** i * (-1.0) ** i
    w_pad, w_rost = F.shirina_bb(c_pad), F.shirina_bb(c_rost)
    assert F.maska_szhatie(w_pad)[100:].all()                # ширина — минимум окна
    assert not F.maska_szhatie(w_rost).any()
    assert not F.maska_szhatie(w_pad)[:60].any()             # min_periods = 50 (+ прогрев 20)


def test_szhatie_shirina_keltner_i_prichinnost():
    wk = F.shirina_keltner(HI, LO, CL)
    assert np.isnan(wk[:19]).all() and (wk[19:] > 0).all()
    i = np.arange(400)
    c = 100 + 5 * 0.985 ** i * (-1.0) ** i
    assert prichinnost(lambda x: F.maska_szhatie(F.shirina_bb(x)), c)


def test_mutatsiya_szhatie_po_vsemu_ryadu_i_inversiya_lovyatsya():
    i = np.arange(400)
    c = 100 + 5 * 0.985 ** i * (-1.0) ** i + RNG.normal(0, .05, 400)
    po_vsemu = lambda x: F.shirina_bb(x) < np.nanquantile(F.shirina_bb(x), 0.3)
    base = lambda x: F.maska_szhatie(F.shirina_bb(x))
    assert prichinnost(base, c) and not prichinnost(po_vsemu, c)
    assert (~base(c))[300] != base(c)[300]
    assert not F.maska_szhatie(F.shirina_bb(c[::-1].copy()))[100:].all()   # развёрнутое время: волатильность растёт
