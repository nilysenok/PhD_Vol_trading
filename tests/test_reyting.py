"""Рейтинг МСП (fastbt/reyting.py): синтетика + мутации М-Ф (фаза по d, сектор с днём d, без нормировки, f1/f2 по d+1, λ вне веса)."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from fastbt import fazy, intraday_priznaki, reyting

IDX = pd.bdate_range("2016-01-04", periods=700)
D = IDX[600]


def dannye(mult_d=None, kol="imoex"):
    """imoex — дрейф + шум; A, B (сектор X) — копии imoex, C (сектор Y) — одинокий. mult_d: умножить close в день D."""
    rng = np.random.default_rng(7)
    sg = np.linspace(0.012, 0.002, len(IDX))        # σ убывает → к D ранг σ низкий → «спокойно»; дрейф вверх → «рост»
    im = pd.Series(100 * np.exp(np.cumsum(0.003 + sg * rng.standard_normal(len(IDX)))), index=IDX)
    ak = pd.DataFrame({"A": im, "B": im, "C": im})
    if mult_d is not None and kol == "imoex":
        im = im.copy(); im[D] *= mult_d
    if mult_d is not None and kol == "sektor":
        ak = ak.copy(); ak.loc[D, ["A", "B"]] *= mult_d
    return {"imoex": im, "akcii": ak, "karta": {"A": "X", "B": "X", "C": "Y"}}


def ves_d(d, sk="S3", side=1.0, lam=1.0):
    fr, fs = reyting.fazy_dnya(["A"], [D], d)
    st = pd.DataFrame({"A": [side]}, index=[D])
    return reyting.ves_panel(sk, st, fr, fs, lam).iloc[0, 0], fr.iloc[0], fs.iloc[0, 0]


def test_primer_spec_kontrtrend_long_padenie_pri_lyuboj_lam():
    assert reyting.ves("S1", "падение_спокойно", "падение_спокойно", 1, 1.0) == pytest.approx(0.09)
    assert reyting.ves("S1", "падение_спокойно", "падение_спокойно", 1, 0.5) == pytest.approx(0.545)


def test_tablica_yachejki():
    v = reyting.ves
    assert v("S3", "рост_спокойно", "рост_спокойно", -1, 1) == 0          # тренд: шорт против роста
    assert v("S3", "падение_стресс", "падение_стресс", 1, 1) == 0         # лонг против падения, даже при стрессе
    assert v("S3", "боковик_стресс", "боковик_стресс", 1, 1) == pytest.approx(0.7)
    assert v("S1", "рост_спокойно", "рост_спокойно", -1, 1) == 0          # Ж-103: шорт контртренда в росте
    assert v("S5", "рост_спокойно", "рост_спокойно", -1, 1) == 0
    assert v("S5", "боковик_спокойно", "боковик_спокойно", -1, 1) == 1
    assert v("S5", "падение_спокойно", "падение_спокойно", 1, 1) == pytest.approx(0.2)   # 0,4 · Q 0,5
    assert v("S3", "рост_спокойно", "боковик_спокойно", 1, 1) == pytest.approx(0.7)   # сектор расходится
    assert v("S7", "рост_стресс", "падение_стресс", -1, 1) == 1                       # кросс-секция нейтральна
    assert v("S1", "боковик_спокойно", "рост_спокойно", -1, 1) == pytest.approx(0.7 * 0.5)  # M и Q


def test_net_fazy_ves_edinica():
    assert reyting.ves("S3", float("nan"), "рост_спокойно", 1, 1.0) == 1.0
    assert reyting.ves("S3", "рост_спокойно", None, 1, 1.0) == 1.0


def lam_vhodit():
    a = reyting.ves("S1", "падение_спокойно", "падение_спокойно", 1, 0.0)
    b = reyting.ves("S1", "падение_спокойно", "падение_спокойно", 1, 0.5)
    return a == 1.0 and abs(b - 0.545) < 1e-12


def test_lam_nol_vyklyuchaet_fazy():
    assert lam_vhodit()
    for sk in reyting.SEMYA:
        for f in fazy.FAZY:
            assert reyting.ves(sk, f, f, -1, 0.0) == 1.0


def test_mutaciya_lam_ne_vhodit_v_ves_lovitsya(monkeypatch):
    monkeypatch.setattr(reyting, "zhestkost", lambda w, lam: w)
    assert not lam_vhodit()


def test_ves_ne_zavisit_ot_dnya_d():
    w0, fr0, fs0 = ves_d(dannye())
    assert fr0 == fs0 == "рост_спокойно" and w0 == 1.0
    for kol in ("imoex", "sektor"):
        assert ves_d(dannye(0.2, kol))[0] == w0


def test_mutaciya_faza_rynka_po_dannym_d_lovitsya(monkeypatch):
    w0 = ves_d(dannye())[0]
    monkeypatch.setattr(fazy, "_LAG", 0)
    assert ves_d(dannye(0.2, "imoex"))[0] != w0


def test_mutaciya_ves_po_faze_sektora_s_dnyom_d_lovitsya():
    w0 = ves_d(dannye())[0]
    orig = fazy.faza_tikera

    def s_dnyom_d(t, daty, d):
        with mock.patch.object(fazy, "_LAG", 0):
            return orig(t, daty, d)
    with mock.patch.object(fazy, "faza_tikera", s_dnyom_d):
        assert ves_d(dannye(0.2, "sektor"))[0] != w0


def test_daty_2020_ne_chitayutsya():
    with pytest.raises(ValueError):
        reyting.fazy_dnya(["A"], [pd.Timestamp("2020-01-03")], dannye())


def sila_synt(n=400, k=3):
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.random((n, k)) + 0.1, index=pd.bdate_range("2016-01-04", periods=n), columns=list("ABC"))


def test_rang_isklyuchaet_tekushchiy_den_i_budushchee():
    s = sila_synt()
    s.iloc[300] = 99.0
    r = reyting.rang_sily(s)
    assert (r.iloc[300] == 1.0).all()                 # окно без d: максимум строго больше всех
    s2 = s.copy(); s2.iloc[301] = -5.0
    pd.testing.assert_series_equal(reyting.rang_sily(s2).iloc[300], r.iloc[300])
    assert r.iloc[:7].isna().all().all() and r.iloc[7].notna().all()   # пул: 7 строк × 3 = 21 ≥ 20; раньше — < 20 наблюдений — кандидата нет


def test_pustaya_lenta_ne_padaet():
    r = reyting.rang_sily(pd.DataFrame(np.nan, index=sila_synt().index, columns=list("ABC")))
    assert r.isna().all().all()


def panel(scale=1.0):
    s = sila_synt() * scale
    zn = pd.Series(np.where(np.arange(len(s)) % 2 == 0, 1.0, -1.0), index=s.index)
    st = pd.DataFrame({c: zn for c in s.columns})
    nan = pd.Series(np.nan, index=s.index)
    fs = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    return reyting.reyting("S1", st, s, pd.DataFrame(1.0, index=s.index, columns=s.columns), nan, fs, 1.0)


def normirovan():
    a, b = panel(1.0), panel(100.0)
    return bool(np.allclose(a.fillna(0), b.fillna(0)) and np.nanmax(np.abs(a.to_numpy())) <= 1.0)


def test_skor_normirovan_i_znak_storony():
    assert normirovan()
    p = panel()
    assert (np.sign(p.iloc[300:310].to_numpy()) * np.where(np.arange(300, 310)[:, None] % 2 == 0, 1, -1) >= 0).all()


def test_mutaciya_skor_bez_normirovki_lovitsya(monkeypatch):
    monkeypatch.setattr(reyting, "rang_sily", lambda s, *a, **k: s)
    assert not normirovan()


def f_synt(n=400, d=300, f_d=0.5, f_next=0.99):
    rng = np.random.default_rng(3)
    f = pd.Series(rng.random(n), index=pd.bdate_range("2016-01-04", periods=n))
    f.iloc[d], f.iloc[d + 1] = f_d, f_next
    return f, d


def g_ne_smotrit_vpered(znak):
    f, d = f_synt()
    sd = pd.Series(-float(znak), index=f.index)         # сторона, которую ослабил бы верхний терциль
    return reyting.g_f(sd, f, znak).iloc[d] == 1.0       # d — средний терциль; d+1 крайний


def test_g_f_napravlenie():
    f, d = f_synt(f_d=0.99)
    sh, lg = pd.Series(-1.0, index=f.index), pd.Series(1.0, index=f.index)
    assert reyting.g_f(sh, f, +1).iloc[d] == 0.5 and reyting.g_f(lg, f, +1).iloc[d] == 1.0   # f2 вверху: шорт ослаблен
    assert reyting.g_f(lg, f, -1).iloc[d] == 0.5 and reyting.g_f(sh, f, -1).iloc[d] == 1.0   # f1 вверху: лонг ослаблен
    assert reyting.g_f(sh, f, +1).iloc[0] == 1.0                                              # нет окна → 1


@pytest.mark.parametrize("znak", [+1, -1])      # f2: +1, f1: −1
def test_g_f_d_plus_1_ne_vlyaet(znak):
    assert g_ne_smotrit_vpered(znak)


@pytest.mark.parametrize("znak", [+1, -1])
def test_mutaciya_f_po_d_plus_1_lovitsya(monkeypatch, znak):
    orig = intraday_priznaki.tercile_cell
    monkeypatch.setattr(reyting, "tercile_cell", lambda x: orig(x.shift(-1)))
    assert not g_ne_smotrit_vpered(znak)


def test_g_novost():
    ix = pd.bdate_range("2016-01-04", periods=40)
    r, sg, v = pd.Series(0.001, index=ix), pd.Series(0.01, index=ix), pd.Series(100.0, index=ix)
    r.iloc[30], v.iloc[31], v.iloc[32] = 0.05, 250.0, 150.0
    g = reyting.g_novost(r, sg, v)
    assert (g.iloc[30], g.iloc[31], g.iloc[32], g.iloc[33]) == (0.0, 0.0, 1.0, 1.0)
    assert g.iloc[:20].isna().all()


def test_oborot_bez_barov_posle_1850_i_g_likvidnost():
    ts = pd.to_datetime(["2016-01-04 18:49", "2016-01-04 18:50", "2016-01-04 10:00"])
    b = pd.DataFrame({"timestamp": ts, "close": [10.0, 10.0, 20.0], "volume": [1.0, 1e9, 2.0]})
    assert reyting.oborot_dnya(b).iloc[0] == 50.0
    ob = pd.Series(100e6, index=pd.bdate_range("2016-01-04", periods=25))
    rz = pd.Series(5e6, index=ob.index); rz.iloc[22] = 5.01e6
    g = reyting.g_likvidnost(rz, ob)
    assert g.iloc[:19].isna().all() and (g.iloc[19], g.iloc[22]) == (1.0, 0.0)   # ровно 5 % проходит
    assert reyting.g_likvidnost(rz * 0.0, ob * 0.0).iloc[22] == 1.0               # нулевой оборот и нулевая ставка
    with pytest.raises(ValueError):
        reyting.oborot_dnya(b.assign(timestamp=pd.Timestamp("2020-01-02 10:00")))
