"""Сборщик входа МСП `fastbt.msp_v5_dohod`: синтетика, без сети и данных (период 2018).

Мутации: комиссия со знаком позиции (шорт получает возврат, Ж-149); день-d попадает в доходность дня d
(M-d1, mutations.md № 55 — лаг исполнения Д2); сторона берётся не с последнего бара дня; чтение ≥ 2020.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars_agg as ba
from fastbt import msp_v5, tf_os_v5 as TF
from fastbt import msp_v5_dohod as M

RATE = TF.KOMISSIYA


def _bary(n_dni=4, bars=3, seed=1):
    rng = np.random.default_rng(seed)
    n = n_dni * bars
    o = 100 + np.cumsum(rng.normal(0, 1, n))
    c = o + rng.normal(0, 1, n)
    d = np.repeat(pd.bdate_range("2018-01-09", periods=n_dni).values.astype("datetime64[ns]"), bars)
    return o, c, d.astype("int64")


def test_odin_bar_v_den_izvestnyj_otvet():
    o = np.array([100.0, 110.0, 90.0])
    c = np.array([105.0, 99.0, 99.0])
    den = pd.bdate_range("2018-01-09", periods=3).values.astype("datetime64[ns]").astype("int64")
    sig = np.array([1, -1, 1], np.int8)
    dni, st, r, t = M.ryad_dnya(sig, o, c, den)
    r, t = M.sdvig_d_plus_1(r, t)
    assert list(st) == [1, -1, 1]
    assert r.iloc[0] == pytest.approx(99 / 110 - 1) and r.iloc[1] == pytest.approx(99 / 90 - 1)   # знак из u, не из R
    assert np.isnan(r.iloc[2]) and list(t.iloc[:2]) == [2.0, 2.0]


def _pos(den, znak):
    """Конвенция `tf_os_v5.pozicii`: первый бар дня без позиции, вход — open 2-го бара, выход — close последнего."""
    p = np.full(len(den), znak, np.int8)
    p[np.r_[True, den[1:] != den[:-1]]] = 0
    return p


def test_tri_bara_izvestnyj_otvet():
    o = np.array([100.0, 102.0, 101.0])
    c = np.array([101.0, 101.0, 103.0])
    den = np.full(3, pd.Timestamp("2018-01-09").value)
    _, _, r, t = M.ryad_dnya(np.zeros(3, np.int8), o, c, den)
    assert r.iloc[0] == pytest.approx((101 / 102 - 1) + (103 / 101 - 1))      # бар 1 без позиции
    assert t.iloc[0] == 2.0


def test_long_i_short_platyat_odinakovo():
    o, c, den = _bary()
    _, _, r, t = M.ryad_dnya(np.zeros(len(o), np.int8), o, c, den)
    assert (RATE * abs(1) * t).equals(RATE * abs(-1) * t) and (t == 2.0).all()


@pytest.mark.parametrize("znak", [1, -1])
def test_summa_ravna_dohod_tf_os(znak):
    """u·R − ставка·|u|·T = чистый доход `tf_os_v5.dohod` (позиция по конвенции pozicii) на дне d+1, 1e-12."""
    o, c, den = _bary()
    dni, st, r, t = M.ryad_dnya(np.zeros(len(o), np.int8), o, c, den)
    r, t = M.sdvig_d_plus_1(r, t)
    g, cost, _ = TF.dohod(_pos(den, znak), o, c, den, RATE)
    net = pd.Series(g - cost, index=den).groupby(level=0).sum().set_axis(dni)
    mine = znak * r - RATE * abs(znak) * t
    assert np.allclose(mine.iloc[:-1].values, net.iloc[1:].values, rtol=0, atol=1e-12)


def test_mutaciya_komissiya_so_znakom_lovitsya():
    o, c, den = _bary()
    dni, st, r, t = M.ryad_dnya(np.zeros(len(o), np.int8), o, c, den)
    r, t = M.sdvig_d_plus_1(r, t)
    g, cost, _ = TF.dohod(_pos(den, -1), o, c, den, RATE)
    net = pd.Series(g - cost, index=den).groupby(level=0).sum().set_axis(dni).iloc[1:].values
    zlo = (-1 * r - RATE * (-1) * t).iloc[:-1].values          # комиссия умножена на u: шорт получает возврат
    assert not np.allclose(zlo, net, rtol=0, atol=1e-12)


def _utechka(sdvig) -> bool:
    """True, если доходность строки d меняется от бумаг дня d (утечка, M-d1)."""
    o, c, den = _bary()
    def r_of(o_, c_):
        _, _, r, t = M.ryad_dnya(np.zeros(len(o_), np.int8), o_, c_, den)
        return sdvig(r, t)[0]
    o2, c2 = o.copy(), c.copy()
    d1 = den == np.unique(den)[1]
    o2[d1], c2[d1] = o[d1] * 1.1, c[d1] * 0.9
    base, pert = r_of(o, c), r_of(o2, c2)
    return not np.isclose(base.iloc[1], pert.iloc[1])


def test_den_d_ne_popadaet_v_dohodnost_dnya_d():
    assert not _utechka(M.sdvig_d_plus_1)
    assert _utechka(lambda r, t: (r, t))                       # мутация: без сдвига — ловится


def test_storona_s_poslednego_bara_dnya():
    o, c, den = _bary(bars=3)
    sig = np.array([1, -1, 0, 0, 0, 1, 1, 1, -1, 0, 0, 0], np.int8)
    _, st, _, _ = M.ryad_dnya(sig, o, c, den)
    assert list(st) == [0, 1, -1, 0]


def test_period_do_2020_otkaz():
    with pytest.raises(ValueError):
        M.sobrat(["S3"], ["X"], ("2019-06-01", "2020-01-02"), "1h", {"S3": {}})


def _b10(days=60, seed=0):
    rng = np.random.default_rng(seed)
    n = days * ba.SLOTS
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    op = np.r_[100.0, cl[:-1]]
    d = pd.bdate_range("2018-01-01", periods=days)
    return pd.DataFrame(dict(date=np.repeat(d, ba.SLOTS), slot=np.tile(np.arange(ba.SLOTS), days), open=op,
                             high=np.maximum(op, cl) * 1.001, low=np.minimum(op, cl) * 0.999, close=cl,
                             volume=rng.uniform(1e3, 5e3, n)))


def test_sobrat_forma_i_msp_v5_prohodit():
    b10 = {"A": _b10(seed=1), "B": _b10(seed=2)}
    tochki = {"S3": dict(okno=10, podtv="none")}
    res = M.sobrat(["S3"], ["A", "B"], ("2018-01-01", "2018-03-01"), "1h", tochki, b10)
    R, T, st = res["R_brutto"], res["T"], res["storona"]["S3"]
    assert R.shape == T.shape == st.shape and list(R.columns) == ["A", "B"] and (T.dropna() >= 0).all().all()
    assert R.index.max() < pd.Timestamp("2018-03-01") and np.isnan(R.iloc[-1]).all()
    one = pd.DataFrame(1.0, index=R.index, columns=R.columns)
    sig = {"S3": {"storona": st, "sila": one, "g": one}}
    out = msp_v5.progon(sig, one * 0.02, {"A": "x", "B": "y"}, {"R_brutto": R, "T": T}, pd.Series(0.0, index=R.index),
                        0.0, k=2)
    assert len(out.ex_kniga) == len(R) and np.isfinite(out.ex_kniga).all()
