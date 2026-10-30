"""V7-2, тестировщик #895: мутации через monkeypatch (окно с днём d; веса без перенормировки; издержки без платы за шорт) и граничные случаи."""
import numpy as np
import pandas as pd

from fastbt import k_okup_v7 as K
from tests.test_k_okup_v7 import D, _tr, _vh, _zhurnal

ORIG = K._ozhidanie


def _s(n, tercil, hold, gross, kom, short, start=0):
    return _tr(n, tercil, hold, gross, kom, start).assign(short=short)


def _zh_short():
    # терциль 1: 20 сделок удержание 1: gross 1,0, kom 0,1, short 0,3 -> издержки 0,4; 1,0 >= 2*0,4 нет (1,0 < 0,8? 1,0 >= 0,8 да), >= 3*0,4 нет
    return _s(20, 1, 1, 1.0, 0.1, 0.3)


def test_izderzhki_vklyuchayut_platu_za_short_i_mutaciya_lovitsya(monkeypatch):
    r = K.maska_vhoda(_zh_short(), _vh(1), 3).iloc[0]
    assert np.isclose(r.izd_hat, 0.4) and not r.razresheno           # 1,0 < 3*0,4
    assert K.maska_vhoda(_zh_short(), _vh(1), 2).razresheno.iloc[0]  # 1,0 >= 2*0,4
    monkeypatch.setattr(K, "_ozhidanie", lambda g, d, t: ORIG(g.assign(short=0.0), d, t))
    m = K.maska_vhoda(_zh_short(), _vh(1), 3).iloc[0]
    assert np.isclose(m.izd_hat, 0.1) and m.razresheno               # мутация: 1,0 >= 3*0,1 — расхождение с каноном видно


def test_okno_s_dnyom_d_mutaciya_lovitsya(monkeypatch):
    z = pd.concat([_tr(19, 3, 1, 0.5, 0.1), _tr(1, 3, 1, 9.0, 0.1, 19)], ignore_index=True)
    z.loc[19, "vyhod"] = D
    ok = K.maska_vhoda(z, _vh(3), 2).iloc[0]
    monkeypatch.setattr(K, "_ozhidanie", lambda g, d, t: ORIG(g, pd.Timestamp(d) + pd.Timedelta(days=1), t))
    m = K.maska_vhoda(z, _vh(3), 2).iloc[0]
    assert ok.n == 19 and m.n == 20 and m.pnl_hat > ok.pnl_hat       # день d попал в оценку — ловится


def test_vesa_bez_perenormirovki_mutaciya_lovitsya(monkeypatch):
    ok = K.maska_vhoda(_zhurnal(), _vh(2), 2).iloc[0]
    assert np.isclose(ok.pnl_hat, 0.5)

    def bez_renorm(g, d, t):
        p, i, n = ORIG(g, d, t)
        w = g[g.vyhod <= pd.Timestamp(d) - pd.Timedelta(days=1)]
        dolya = (w[w.tercil == t].korz.value_counts().sum()) / len(w)   # вес ячеек от окна без деления на сумму весов ячеек
        return p * dolya, i * dolya, n

    monkeypatch.setattr(K, "_ozhidanie", bez_renorm)
    assert not np.isclose(K.maska_vhoda(_zhurnal(), _vh(2), 2).pnl_hat.iloc[0], 0.5)


def test_vesa_ne_delyat_na_nol_i_nan_tercil():
    z = _zhurnal()
    r = K.maska_vhoda(z, pd.DataFrame({"vhod": [D, D], "tercil": [99, np.nan]}), 2)    # терцили без сделок
    assert (r.n == 0).all() and r.razresheno.all() and r.malo.all() and r.pnl_hat.isna().all()
    assert np.isfinite(K.maska_vhoda(z, _vh(1), 2).pnl_hat.iloc[0])


def test_porog_20_po_tercilyu_a_ne_po_oknu():
    z = pd.concat([_s(19, 1, 1, -1.0, 0.5, 0.0), _s(30, 2, 1, 1.0, 0.1, 0.0, 19)], ignore_index=True)   # окно 49, терциль 1 — 19
    r = K.maska_vhoda(z, _vh(1), 3).iloc[0]
    assert r.n == 19 and r.malo and r.razresheno                      # по терцилю: мало -> разрешено (по окну было бы 49 >= 20 -> заблокировано)
    z2 = pd.concat([_s(20, 1, 1, -1.0, 0.5, 0.0), _s(1, 2, 1, 1.0, 0.1, 0.0, 20)], ignore_index=True)
    r2 = K.maska_vhoda(z2, _vh(1), 3).iloc[0]
    assert r2.n == 20 and not r2.malo and not r2.razresheno           # ровно 20 — оценка работает


def test_zakrytye_posle_d_i_otkrytye_ne_uchastvuyut():
    z = pd.concat([_s(20, 1, 1, 1.0, 0.1, 0.0), _s(5, 1, 1, -50.0, 0.1, 0.0)], ignore_index=True)
    z.loc[20:, "vyhod"] = [D + pd.Timedelta(days=i) for i in range(1, 6)]   # закрываются после d
    z.loc[20, "vyhod"] = pd.NaT                                              # ещё открыта
    r = K.maska_vhoda(z, _vh(1), 2).iloc[0]
    assert r.n == 20 and np.isclose(r.pnl_hat, 1.0)


def test_znak_gross_beretsya_kak_dan_dlya_shorta():
    # шорт: gross уже со знаком стороны (прибыль при падении = +); модуль знак не меняет
    z = _s(20, 1, 1, 1.0, 0.1, 0.1)
    assert K.maska_vhoda(z, _vh(1), 2).pnl_hat.iloc[0] > 0
    assert K.maska_vhoda(z.assign(gross=-1.0), _vh(1), 2).pnl_hat.iloc[0] < 0
