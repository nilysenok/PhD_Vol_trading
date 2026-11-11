"""Тестировщик #869b: известные ответы и мутации `fastbt.msp_v5_dohod` (R_brutto — лонг d+1 в конвенции `pozicii`).

R[d] = доходность единицы за день d+1 (бар 1 без позиции, вход open бара 2, выход close последнего; день из одного
бара — целиком), без комиссии и знака; T[d] = 2. МСП — книга ставок, автомат члена не сверяется.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp, msp_v5, tf_os_v5 as TF
from fastbt import msp_v5_dohod as M
from test_msp_v5_dohod import _b10

TOCHKA = dict(okno=10, podtv="none")


def _den(*chisla):
    d = pd.bdate_range("2018-01-09", periods=len(chisla)).values.astype("datetime64[ns]").astype("int64")
    return np.repeat(d, chisla)


def test_izvestnyj_otvet_tri_bara_ruchkoj():
    """День 1: бары o=[100,110,90], c=[105,99,99]. Позиция [0,1,1]: бар 2: 90/110-1, бар 3: 99/90-1; T = 2."""
    o = np.array([50.0, 100, 110, 90])
    c = np.array([51.0, 105, 99, 99])
    _, _, r, t = M.ryad_dnya(np.zeros(4, np.int8), o, c, _den(1, 3))
    r, t = M.sdvig_d_plus_1(r, t)
    assert r.iloc[0] == pytest.approx((90 / 110 - 1) + (99 / 90 - 1)) and t.iloc[0] == 2.0
    assert np.isnan(r.iloc[1]) and np.isnan(t.iloc[1])


def test_den_iz_odnogo_bara_derzhitsya_celikom():
    o = np.array([50.0, 100, 70])
    c = np.array([51.0, 108, 71])
    _, _, r, t = M.ryad_dnya(np.zeros(3, np.int8), o, c, _den(1, 1, 1))
    r, t = M.sdvig_d_plus_1(r, t)
    assert r.iloc[0] == pytest.approx(108 / 100 - 1) and t.iloc[0] == 2.0
    assert r.iloc[1] == pytest.approx(71 / 70 - 1)


def test_mut_vhod_na_pervom_bare_lovitsya():
    """Мутация: позиция с бара 1 (вход open бара 1) даёт иное R, чем конвенция pozicii."""
    o, c = np.array([50.0, 100, 110, 90]), np.array([51.0, 105, 99, 99])
    den = _den(1, 3)
    g, _, _ = TF.dohod(np.ones(4, np.int8), o, c, den, 0.0)
    zlo = g[1:].sum()
    _, _, r, _ = M.ryad_dnya(np.zeros(4, np.int8), o, c, den)
    assert r.iloc[1] != pytest.approx(zlo)


def test_mut_sdvig_nulevoj_lovitsya_na_sobrat():
    """R без сдвига (строка d = доходность дня d) меняется от возмущения дня d; штатная — нет (M-d1)."""
    def r_d1(sdvig, mult):
        b = _b10(30, 4)
        d1 = b["date"] == sorted(b["date"].unique())[5]
        b = b.assign(close=np.where(d1, b["close"] * mult, b["close"]))
        td = TF.ticker_data(TF.bary(b, "1h"), "X")
        den = TF.den_id(td)
        _, _, r, t = M.ryad_dnya(np.zeros(len(den), np.int8), td.open, td.close, den)
        return sdvig(r, t)[0].iloc[5]
    assert r_d1(M.sdvig_d_plus_1, 1.0) == pytest.approx(r_d1(M.sdvig_d_plus_1, 1.1), nan_ok=True)
    assert r_d1(lambda r, t: (r, t), 1.0) != pytest.approx(r_d1(lambda r, t: (r, t), 1.1))


def test_mut_T_tolko_vhod_lovitsya():
    """T за полный день с несколькими барами = 2; мутация «только вход» = 1 ловится."""
    o, c = np.array([1., 1.1, 1.2]), np.array([1.05, 1.15, 1.25])
    _, _, _, t = M.ryad_dnya(np.zeros(3, np.int8), o, c, _den(3))
    assert t.iloc[0] == 2.0 != 1.0


def test_mut_storona_s_pervogo_bara_lovitsya():
    o = np.arange(1., 7.)
    den = _den(3, 3)
    sig = np.array([1, 1, -1, -1, -1, 1], np.int8)
    st = M.ryad_dnya(sig, o, o + .1, den)[1]
    assert list(st) == [-1, 1] and list(st) != list(sig[::3])


def _sobrat_dlinnyj():
    res = M.sobrat(["S3"], ["A"], ("2018-01-01", "2019-02-01"), "1h", {"S3": TOCHKA}, {"A": _b10(300, 1)})
    return res["R_brutto"], res["T"]


def _sig(R, znak, rng_seed=0):
    one = pd.DataFrame(1.0, index=R.index, columns=R.columns)
    sila = one * np.random.default_rng(rng_seed).uniform(0.1, 1, R.shape)      # ранг силы: окно 252 д.
    return {"S3": {"storona": one * znak, "sila": sila, "g": one}}, one


def test_d4_den_bez_barov_pri_stavke_otkaz_knigi():
    """Последний день: R/T = NaN; при ставке ≠ 0 книга отказывает (не нули); без ставки — не отказывает."""
    R, T = _sobrat_dlinnyj()
    assert np.isnan(R.iloc[-1, 0]) and np.isnan(T.iloc[-1, 0])
    sig, one = _sig(R, 1.0)
    with pytest.raises(ValueError, match="NaN"):
        msp_v5.progon(sig, one * 0.02, {"A": "x"}, {"R_brutto": R, "T": T}, pd.Series(0.0, index=R.index), 0.0, k=1)
    out = msp_v5.progon(*_sig(R.iloc[:-1], 1.0)[:1], _sig(R.iloc[:-1], 1.0)[1] * 0.02, {"A": "x"},
                        {"R_brutto": R, "T": T}, pd.Series(0.0, index=R.index), 0.0, k=1)
    assert np.isfinite(out.ex_kniga).all()


def test_long_i_short_platyat_odinakovo_cherez_knigu():
    """ex_kniga = u·R − ставка·|u|·T при storona = +1 и −1 (шорт не получает возврат комиссии, Ж-149)."""
    res = M.sobrat(["S3"], ["A", "B"], ("2018-01-01", "2019-02-01"), "1h", {"S3": TOCHKA},
                   {"A": _b10(300, 1), "B": _b10(300, 2)})
    T1 = res["T"].iloc[:-1]
    assert kniga_msp.KOMISSIYA > 0
    for z in (1.0, -1.0):              # книга из двух ног (нетто шага 8 гасит одностороннюю): A стороной z, B стороной −z
        R1 = res["R_brutto"].iloc[:-1].abs() * [z, -z]       # своя сторона прибыльна: самоотключение Sk не гасит u
        sig, one = _sig(R1, 1.0)
        sig["S3"]["storona"] = one * [z, -z]
        out = msp_v5.progon(sig, one * 0.02, {"A": "x", "B": "y"}, {"R_brutto": R1, "T": T1},
                            pd.Series(0.0, index=R1.index), 0.0, k=2)
        u = out.u.to_numpy()
        assert (u[:, 0] * z >= 0).all() and (u[:, 1] * z <= 0).all() and (u != 0).any()
        ozhid = (u * R1.to_numpy()).sum(axis=1) - kniga_msp.KOMISSIYA * (np.abs(u) * T1.to_numpy()).sum(axis=1)
        assert np.allclose(out.ex_kniga.to_numpy(), ozhid, rtol=0, atol=1e-12)
        mut = (u * R1.to_numpy()).sum(axis=1) - kniga_msp.KOMISSIYA * (u * T1.to_numpy()).sum(axis=1)
        assert z > 0 or not np.allclose(out.ex_kniga.to_numpy(), mut, atol=1e-12)    # комиссия со знаком — ловится
