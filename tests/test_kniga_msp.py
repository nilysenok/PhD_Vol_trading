"""Книга МСП (Ф4): синтетика с известным ответом + 6 мутаций (каждая роняет инвариант). Без сети и реальных данных."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp as km
from fastbt import mix_skleyka as mx

RNG = np.random.default_rng(7)


# ---------- инварианты (принимают проверяемую функцию; мутант обязан их ронять) ----------

def inv_sektor(fn):
    score = np.array([9, 8, 7, 6, 5, 4, 3, 2.0])
    u = fn(score, np.full(8, 0.02), np.array([0, 0, 0, 0, 0, 1, 1, 1]), 8, 0.01)
    assert (u[:5] != 0).sum() <= 3 and (u != 0).sum() == 6


def inv_netto(fn):
    u = fn(np.array([5.0, 4.0, 3.0, -1.0]))
    L, S = u[u > 0].sum(), -u[u < 0].sum()
    assert abs(L - S) <= 0.3 * (L + S) + 1e-9


def inv_sigma_bez_dnya_d(fn):
    ex = RNG.normal(0, 0.01, 60)
    a = fn(ex, 40)
    ex2 = ex.copy()
    ex2[40:] += 5.0                      # день d и будущее
    assert fn(ex2, 40) == a


def inv_beta_bez_dnya_d(fn):
    x = RNG.normal(0, 0.01, 100)
    y = 0.8 * x + RNG.normal(0, 0.002, 100)
    a = fn(y, x, 80)
    y2 = y.copy()
    y2[80:] = 9.0 * x[80:] + 1.0
    assert fn(y2, x, 80) == a


def inv_hedzh_snizhaet_betu(fn):
    x = RNG.normal(0, 0.01, 200)
    y = 1.2 * x + RNG.normal(0, 0.001, 200)
    b = km.beta(y, x, 150)
    assert abs(np.cov(y + fn(b) * x, x)[0, 1]) < 0.2 * abs(np.cov(y, x)[0, 1])


def _sr_dva_kvartala(vtoroj_konec):
    """T=40, квартал = 10 дней. SR<−0,5 на конце q0 всегда, на конце q1 — по условию."""
    sr = np.full((40, 1), -0.2)                    # ниже 0 (не снимает), выше −0,5 (не плохо)
    sr[9] = -1.0
    sr[19] = -1.0 if vtoroj_konec else -0.2
    return sr, np.arange(40) // 10


def inv_samootkl_dva_kvartala(fn):
    sr, kv = _sr_dva_kvartala(False)               # плохой конец только один -> не отключать
    assert not fn(sr, kv, 25)[0]
    sr, kv = _sr_dva_kvartala(True)
    assert fn(sr, kv, 19)[0] == 0 and fn(sr, kv, 25)[0]   # отключено с первого дня q2, не раньше


# ---------- реальные функции проходят ----------

def test_invarianty_na_pravke():
    inv_sektor(km.vybor_stavok)
    inv_netto(km.netto)
    inv_sigma_bez_dnya_d(lambda ex, i: km.mnozhitel_vola(ex, i, 1.0))
    inv_beta_bez_dnya_d(km.beta)
    inv_hedzh_snizhaet_betu(km.hedzh_mix)
    inv_samootkl_dva_kvartala(km.otklyucheno)


# ---------- мутации ----------

def test_mut_sigma_portfelya_s_dnem_d():
    with pytest.raises(AssertionError):
        inv_sigma_bez_dnya_d(lambda ex, i: km.mnozhitel_vola(ex, i + 1, 1.0))


def test_mut_limit_sektora_4():
    with pytest.raises(AssertionError):
        inv_sektor(lambda *a: km.vybor_stavok(*a, max_sek=4))


def test_mut_netto_bez_ogranicheniya():
    with pytest.raises(AssertionError):
        inv_netto(lambda u: km.netto(u, maks=1.0))


def test_mut_beta_po_dnyu_d():
    with pytest.raises(AssertionError):
        inv_beta_bez_dnya_d(lambda y, x, i: km.beta(y, x, i + 1))


def test_mut_samootklyuchenie_po_odnomu_kvartalu():
    with pytest.raises(AssertionError):
        inv_samootkl_dva_kvartala(lambda sr, kv, i: km.otklyucheno(sr, kv, i, podryad=1))


def test_mut_hedzh_v_storonu_bety():
    with pytest.raises(AssertionError):
        inv_hedzh_snizhaet_betu(lambda b: +b)


# ---------- известные ответы ----------

def test_razmer_ravnogo_riska_i_znak():
    u = km.vybor_stavok([3.0, -2.0, 0.0, np.nan], [0.02, 0.04, 0.01, 0.01], [0, 1, 2, 3], 4, 0.01, ks=2.0)
    assert u == pytest.approx([0.01 / (2 * 0.02), -0.01 / (2 * 0.04), 0, 0])


def test_top_k_po_modulyu_skora():
    u = km.vybor_stavok([1.0, -5.0, 3.0, 2.0], np.full(4, 0.01), [0, 1, 2, 3], 2, 0.01)
    assert list(np.flatnonzero(u)) == [1, 2] and u[1] < 0


def test_netto_tochnoe_znachenie():
    u = km.netto(np.array([10.0, 2.0, -2.0]))     # L=12, S=2 -> L' = 2·1,3/0,7
    assert u[0] + u[1] == pytest.approx(2 * 1.3 / 0.7) and u[2] == -2


def test_netto_nulevaya_kniga_i_odna_storona():
    assert not km.netto(np.zeros(3)).any()
    assert not km.netto(np.array([1.0, 2.0])).any()          # одна сторона: нетто 100 % -> до нуля


def test_vola_tsel_i_plecho():
    ex = np.tile([0.01, -0.01], 20)
    assert km.mnozhitel_vola(ex, 40, 1.0, 0.10, 10.0) == pytest.approx(0.10 / km._ann(ex[19:40]))
    assert km.mnozhitel_vola(ex, 40, 1.0, 0.10, 0.5) == pytest.approx(0.5)   # плечо: m·валовая ≤ 0,5
    assert km.mnozhitel_vola(ex, 10, 1.0) == 1.0           # короче 21 строки
    assert km.mnozhitel_vola(np.zeros(40), 30, 1.0) == 1.0  # нулевая вола


def test_marzha_tolko_sverh_100():
    assert km.marzha_v_den(0.8, 0.1) == 0
    assert km.marzha_v_den(1.5, 0.1) == pytest.approx(0.5 * 0.12 / 252)


def test_vyklyuchatel_vola_i_prosadka():
    assert km.vyklyuchatel(np.tile([0.001, -0.001], 30), 40) == 1.0
    assert km.vyklyuchatel(np.tile([0.02, -0.02], 30), 40) == 0.5          # вола 0,5 > 0,2
    ex = np.zeros(40)
    ex[5] = -0.06
    assert km.vyklyuchatel(ex, 20) == 0.5 and km.vyklyuchatel(ex, 5) == 1.0   # просадка видна на день позже
    assert km.vyklyuchatel(np.array([]), 0) == 1.0


def test_samootklyuchenie_snyatie_pri_sr_vyshe_nulya():
    sr, kv = _sr_dva_kvartala(True)
    sr[27] = 0.4
    off = km.otklyucheno(sr, kv, 30)
    assert km.otklyucheno(sr, kv, 26)[0] and not off[0]


def test_vesa_sk_obratnaya_vola_i_nakloni():
    T = 330
    ex = np.stack([RNG.normal(0.0005, 0.01, T), RNG.normal(0.0005, 0.02, T)], axis=1)
    kv = np.arange(T) // 63
    w = km.vesa_sk(ex, kv, 320)
    assert w[0] > w[1] and w.sum() <= 1 + 1e-12
    ex2 = ex.copy()
    ex2[320:] = 7.0
    assert km.vesa_sk(ex2, kv, 320) == pytest.approx(w)        # день d и позже не читаются
    assert km.vesa_sk(ex[:10], kv, 1) == pytest.approx([0.5, 0.5])   # нет данных — поровну
    ex3 = ex.copy()
    ex3[:, 1] = -np.abs(ex3[:, 1]) - 0.01                      # SR второй Sk ≪ −1: наклон −0,5 против +0,5
    assert km.vesa_sk(ex3, kv, 320)[1] < w[1]


def _ist(T=330, S=2):
    return {"ex_sk": RNG.normal(0, 0.01, (T, S)), "ex_port": RNG.normal(0, 0.005, T),
            "ex_kniga": RNG.normal(0, 0.005, T), "imoex": RNG.normal(0, 0.01, T), "kvartal": np.arange(T) // 63}


def test_kniga_dnya_ne_chitaet_dni_d_i_pozzhe():
    ist, i = _ist(), 300
    args = (i, [3.0, -2.0, 1.0, 4.0], np.full(4, 0.02), [0, 0, 1, 1], [0, 1, 0, 1])
    a = km.kniga_dnya(*args, ist, k=3)
    ist2 = {k: v.copy() for k, v in ist.items()}
    for k in ("ex_sk", "ex_port", "ex_kniga", "imoex"):
        ist2[k][i:] += 9.0
    b = km.kniga_dnya(*args, ist2, k=3)
    assert b.u == pytest.approx(a.u) and b.hedzh == a.hedzh and b.m == a.m and b.vykl == a.vykl
    assert (a.u != 0).sum() == 3


# ---------- склейка MIX ----------

def _kontrakty():
    daty = pd.bdate_range("2016-02-01", "2016-07-01")
    h = pd.DataFrame({"OPEN": 100.0 + np.arange(len(daty))}, index=daty)[daty <= "2016-03-15"]
    m = pd.DataFrame({"OPEN": 300.0 + np.arange(len(daty))}, index=daty)[daty <= "2016-06-15"]
    return {"MXH6_2016": h, "MXM6_2016": m}, daty


def test_skleika_roll_za_dva_dnya_bez_skachka():
    k, daty = _kontrakty()
    r = mx.skleit(k, daty)
    assert r.loc["2016-03-11", "secid"] == "MXH6_2016"        # 15.03 − 2 торговых дня = 11.03
    assert r.loc["2016-03-14", "secid"] == "MXM6_2016"
    assert r.loc["2016-03-11", "ret_oo"] == pytest.approx(h_ret(k, "2016-03-11", "2016-03-14"))
    ok = r["ret_oo"].loc[:"2016-06-13"]                       # до ролла последнего контракта
    assert ok.abs().max() < 0.02 and not ok.isna().any()      # разрыв 100→300 в ряд не попал


def h_ret(k, a, b):
    o = k["MXH6_2016"]["OPEN"]
    return o[b] / o[a] - 1


def test_mut_roll_posle_ekspiracii_lovitsya():
    k, daty = _kontrakty()
    r = mx.skleit(k, daty, dney=-1)                            # ролл на день позже экспирации
    with pytest.raises(AssertionError):                        # инвариант «цена есть на t и t+1 активного» падает
        assert not r["ret_oo"].loc[:"2016-06-13"].isna().any()


def test_zagruzka_obrezaet_2020(tmp_path):
    pd.DataFrame({"TRADEDATE": ["2019-12-30", "2020-01-09"], "SECID": "MXH0", "OPEN": [1.0, 2.0]}).to_csv(
        tmp_path / "MXH0.csv", index=False)
    assert mx.zagruzit_kontrakty(tmp_path)["MXH0"].index.max() == pd.Timestamp("2019-12-30")
