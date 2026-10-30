"""Л1.3 (Ж-146): fastbt/golaya_v7_isp.py — дневной исполнитель голой идеи. Только синтетика с известным ответом.

Мутации (каждая ловится): вход в баре сигнала (mutations.md № 55; Д2) — test_mutaciya_vhod_v_den_signala;
цена исполнения не из T — test_vhod_T_*; комиссия выхода не взята на конце окна (№ 5, 6) — test_konec_okna_*;
знак позиции удвоен/потерян (№ 20) — test_znak_*; комиссия — копия вместо импорта — test_konstanty_te_zhe.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import golaya_v7_isp as G
from fastbt import kross_seksiya as KS
from fastbt import tf_os_v5

KOM = 0.0004
DATES = pd.bdate_range("2016-01-04", periods=8).to_numpy()   # Пн 4.01 … Ср 13.01; между Пт 8.01 и Пн 11.01 — 3 дня
O = 100.0 + np.arange(8)
C = O + 0.5


def run(sig, o=O, c=C, dates=DATES, rate=0.0, **kw):
    n = len(dates)
    return G.ispolnit(np.asarray(sig, int), np.asarray(o, float), np.asarray(c, float), dates=dates,
                      rate=np.full(n, rate) if np.isscalar(rate) else rate, **kw)


def sig_v(i, s=1, n=8):
    x = np.zeros(n, int)
    x[i] = s
    return x


def test_izvestnyj_otvet_long_d1():
    r = run(sig_v(1), vhod="d1", tau=1)               # вход open[2] = 102, выход close[2] = 102,5
    assert r["pnl"][2] == pytest.approx(0.5 / 102)
    assert r["pnl"].sum() == pytest.approx(0.5 / 102)
    assert r["kom"][2] == pytest.approx(KOM + KOM * 102.5 / 102)
    assert np.flatnonzero(r["poz"]).tolist() == [2] and r["poz"][2] == 1
    assert r["zaem"].sum() == 0
    assert r["excess"] == pytest.approx(r["pnl"] - r["kom"] - r["zaem"] - r["stavka"])


def test_mutaciya_vhod_v_den_signala():
    """Д2/№ 55: вход по open дня сигнала дал бы другое число; ispolnit обязан давать open дня d+1."""
    r = run(sig_v(1), vhod="d1", tau=2)               # вход open[2]=102, выход close[3]=103,5
    assert r["pnl"].sum() == pytest.approx(1.5 / 102)
    assert r["pnl"].sum() != pytest.approx(2.5 / 101)  # мутация: вход open[1]
    assert r["poz"][1] == 0


def test_vhod_v_den_signala_otkaz():
    with pytest.raises(ValueError, match="Д2"):
        run(sig_v(1), vhod="d0", tau=1)
    with pytest.raises(ValueError, match="Д2"):
        G.proverka_d2(3, 3)
    G.proverka_d2(3, 4)


def test_znak_short_na_rostushchey_cene_ubytok():
    r = run(sig_v(0, -1), vhod="d1", tau=3)           # вход open[1]=101, выход close[3]=103,5
    assert r["pnl"].sum() == pytest.approx(-2.5 / 101)
    assert (r["poz"][1:4] == -1).all()
    # дневная разбивка: 101 → c1 101,5 → c2 102,5 → c3 103,5 (без удвоения и потери знака)
    assert r["pnl"][1:4] == pytest.approx([-0.5 / 101, -1 / 101, -1 / 101])


def test_znak_long_i_short_zerkalny():
    a, b = run(sig_v(1, 1), tau=2), run(sig_v(1, -1), tau=2)
    assert a["pnl"] == pytest.approx(-b["pnl"])
    assert a["kom"] == pytest.approx(b["kom"])        # комиссия не зависит от стороны


def test_zaem_tolko_za_nochi_shorta_s_vyhodnymi():
    r = run(sig_v(3, -1), tau=2)                      # вход Пт 8.01 (e=4), выход Пн 11.01 (x=5): ночь Пт→Пн = 3 дня
    assert r["zaem"][4] == pytest.approx(0.05 * 3 / 365)
    assert r["zaem"].sum() == pytest.approx(0.05 * 3 / 365)
    assert run(sig_v(3, 1), tau=2)["zaem"].sum() == 0                  # лонг — 0
    assert run(sig_v(2, -1), vhod="T", p_T=O, tau=0)["zaem"].sum() == 0   # вход и выход в один день — 0


def test_stavka_na_zanyatyj_kapital():
    r = run(sig_v(0, 1), tau=3, rate=0.10)            # ночи 1, 2 по 1 дню
    assert r["stavka"][1] == pytest.approx(0.10 / 365) and r["stavka"][2] == pytest.approx(0.10 / 365)
    assert r["stavka"][3] == 0 and r["stavka"][0] == 0
    assert r["excess"].sum() == pytest.approx(r["pnl"].sum() - r["kom"].sum() - r["stavka"].sum())


def test_vhod_T_cena_iz_T_a_ne_zakrytie_i_ne_zavtra():
    pT = np.array([100.2, 101.2, 102.2, 103.2, 104.2, 105.2, 106.2, 107.2])
    r = run(sig_v(1), vhod="T", p_T=pT, tau=1)        # вход p_T[1]=101,2 в день сигнала, выход close[2]=102,5
    assert r["pnl"].sum() == pytest.approx(1.3 / 101.2)
    assert r["poz"][1] == 1 and r["poz"][2] == 1
    o2 = O.copy()
    o2[2] += 5
    assert run(sig_v(1), o=o2, vhod="T", p_T=pT, tau=1)["pnl"].sum() == pytest.approx(r["pnl"].sum())  # open d+1 не нужен
    pT2 = pT.copy()
    pT2[1] += 1                                       # цена T сдвинута → результат обязан измениться ровно на неё
    assert run(sig_v(1), vhod="T", p_T=pT2, tau=1)["pnl"].sum() == pytest.approx(0.3 / 102.2)


def test_vhod_T_bez_ceny_otkaz():
    with pytest.raises(ValueError, match="p_T"):
        run(sig_v(1), vhod="T", tau=1)
    with pytest.raises(ValueError):
        run(sig_v(1), vhod="T", p_T=O[:5], tau=1)


def test_vhod_T_dyra_v_cene_T_propusk():
    pT = O.copy()
    pT[1] = np.nan
    r = run(sig_v(1), vhod="T", p_T=pT, tau=1)
    assert r["propusk"] == 1 and r["pnl"].sum() == 0 and r["kom"].sum() == 0


def test_pustaya_lenta():
    e = np.array([])
    r = G.ispolnit(e.astype(int), e, e, dates=e.astype("datetime64[D]"), rate=e, vhod="d1", tau=1)
    assert all(len(r[k]) == 0 for k in ("pnl", "kom", "zaem", "stavka", "excess", "poz"))


def test_net_signalov_vse_nuli():
    r = run(np.zeros(8, int), tau=1)
    assert not r["pnl"].any() and not r["kom"].any() and not r["poz"].any()


def test_signal_v_posledniy_den_vhoda_net():
    r = run(sig_v(7), vhod="d1", tau=1)
    assert not r["kom"].any() and not r["poz"].any()


def test_dyra_i_nulevaya_cena_vhoda_propusk_bez_nan():
    o = O.copy()
    o[2] = np.nan
    r = run(sig_v(1), o=o, tau=1)
    assert r["propusk"] == 1 and np.isfinite(r["excess"]).all() and r["kom"].sum() == 0
    o[2] = 0.0                                        # нулевая цена: деления на ноль нет
    r = run(sig_v(1), o=o, tau=1)
    assert r["propusk"] == 1 and np.isfinite(r["excess"]).all()


def test_dyra_v_puti_otkaz_a_ne_ffill():
    c = C.copy()
    c[3] = np.nan
    with pytest.raises(ValueError, match="пропуск"):
        run(sig_v(1), c=c, tau=3)


def test_nepravilnyj_signal_i_formy():
    with pytest.raises(ValueError):
        run(sig_v(1, 2), tau=1)
    with pytest.raises(ValueError):
        run(sig_v(1), c=C[:5], tau=1)
    with pytest.raises(ValueError):
        run(sig_v(1), vhod="d1", tau=0)               # выход раньше входа


def test_konec_okna_komissiya_vyhoda_platitsya():
    """№ 5, 6: позиция, открытая на конце окна, закрывается по последнему close и платит обе стороны."""
    r = run(sig_v(5), vhod="d1", tau=5)               # вход open[6]=106, выход — close[7]=107,5 (окно кончилось)
    assert r["pnl"].sum() == pytest.approx(1.5 / 106)
    assert r["kom"].sum() == pytest.approx(KOM + KOM * 107.5 / 106)
    assert r["kom"][7] > 0 and r["poz"][7] == 1
    r2 = run(sig_v(1), vhod="d1", tau=None)           # без τ и флагов — держим до конца окна
    assert r2["kom"][7] == pytest.approx(KOM * 107.5 / 102)


def test_bez_nakrytiya_sdelok():
    r = run(np.array([1, 1, 1, 1, 1, 1, 0, 0]), tau=2)
    assert (np.abs(r["poz"]) <= 1).all()
    assert r["kom"][1] == pytest.approx(KOM)
    # сделки: сигнал 0 → e=1,x=2; сигнал 2 → e=3,x=4; сигнал 4 → e=5,x=6
    assert np.flatnonzero(r["poz"]).tolist() == [1, 2, 3, 4, 5, 6]
    assert len(r["sdelki"]) == 3


def test_flag_vyhoda_ispolnyaetsya_po_open_sleduyushchego_dnya():
    xl = np.zeros(8, bool)
    xl[3] = True
    r = run(sig_v(1), exit_l=xl, vhod="d1")           # вход open[2]=102, флаг в день 3 → выход open[4]=104
    assert r["pnl"].sum() == pytest.approx(2 / 102)
    assert r["kom"].sum() == pytest.approx(KOM + KOM * 104 / 102)
    assert np.flatnonzero(r["poz"]).tolist() == [2, 3, 4]
    assert run(sig_v(1, -1), exit_l=xl, vhod="d1")["poz"][7] == -1    # exit_l шорта не трогает
    xl2 = np.zeros(8, bool)
    xl2[1] = True                                      # флаг до входа игнорируется
    assert run(sig_v(1), exit_l=xl2, vhod="d1")["poz"][7] == 1


def test_flag_i_tau_ranny_vyigryvaet():
    xl = np.zeros(8, bool)
    xl[2] = True                                       # open[3] раньше close[3]
    r = run(sig_v(1), exit_l=xl, vhod="d1", tau=2)
    assert r["pnl"].sum() == pytest.approx(1 / 102)


def test_dividend_shorta_umenshaet_pnl():
    D = np.zeros(8)
    D[3] = 2.0
    r = run(sig_v(0, -1), D=D, tau=3)                  # вход open[1]=101, держим через ex-день 3
    assert r["pnl"].sum() == pytest.approx(-(2.5 + 2.0) / 101)


def test_konstanty_te_zhe():
    """Д7: комиссия и плата — те же объекты, что в движке/книге, не копия."""
    assert G.KOMISSIYA is tf_os_v5.KOMISSIYA and G.KOMISSIYA == KS.KOM == 0.0004
    assert G.ZAEM is KS.ZAEM and G.DNEI is KS.DNEI


def test_sovpadenie_s_golaya_tf_v5_po_brutto():
    """G6: брутто = sdelki_impulsa при τ ∈ {1,2,3} (допуск 1e-12), без сигналов в хвосте."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "server"))
    import golaya_tf_v5 as GT
    rng = np.random.default_rng(7)
    n = 120
    dates = pd.bdate_range("2016-01-04", periods=n)
    o = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    c = o * np.exp(rng.normal(0, 0.005, n))
    sig = rng.choice([-1, 0, 0, 1], n)
    sig[-5:] = 0
    d = pd.DataFrame(dict(date=dates, open=o, close=c))
    for tau in (1, 2, 3):
        ref = sum(s * (x / e - 1.0) for s, e, x in
                  GT.sdelki_impulsa(d, pd.Series(sig, index=dates), tau, "2016-01-01", "2020-01-01"))
        r = G.ispolnit(sig, o, c, dates=dates.to_numpy(), rate=np.zeros(n), vhod="d1", tau=tau)
        assert abs(r["pnl"].sum() - ref) < 1e-12
