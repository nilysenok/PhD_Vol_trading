"""К3: размер, режим, события (Ж-047, тесты 2.11, 2.12 + размер). Синтетика с известным ответом;
мутации подменяют константы *_LAG в памяти и обязаны ронять проверку."""
import numpy as np
import pandas as pd
import pytest

from fastbt import rezhim_e1 as R
from fastbt import sobytiya_e1 as S

D = pd.bdate_range("2019-01-01", periods=40)


# ---- размер ----
def test_razmer_obratnyj_i_limit():
    sig = np.array([0.02, 0.04, 0.08, 0.0005, np.nan, 0.0, -1.0])
    w = R.razmer_ves(sig, "obratnyj", 0.20)
    assert w[0] == pytest.approx(0.20 / (0.02 * np.sqrt(252)))
    assert w[1] == pytest.approx(0.20 / (0.04 * np.sqrt(252)))
    assert w[2] == pytest.approx(0.20 / (0.08 * np.sqrt(252)))
    assert w[3] == 1.0                              # 0.2/(0.0005·15.9) = 25 → обрезан по L
    assert list(w[4:]) == [0.0, 0.0, 0.0]
    assert R.razmer_ves(sig, "ravnyj").max() <= 1.0


def test_razmer_limit_L_ne_narushen_mutaciya():
    sig = np.full(50, 0.0001)
    assert R.razmer_ves(sig, "obratnyj", 0.25, limit=1.0).max() == 1.0
    assert R.razmer_ves(sig, "obratnyj", 0.25, limit=0.5).max() == 0.5
    # мутация: убрать обрезку → вес 157 ≠ 1
    raw = 0.25 / (0.0001 * np.sqrt(252))
    assert raw > 1.0 and R.razmer_ves(sig, "obratnyj", 0.25).max() != raw


def test_razmer_vhod_validaciya():
    with pytest.raises(ValueError):
        R.razmer_ves(np.ones(3), "kelly")


def test_ves_ne_zavisit_ot_sigma_d_plus_1(monkeypatch):
    a = np.full(30, 0.01)
    b = a.copy()
    b[11] = 0.09                                    # скачок σ в день d+1 = 11
    wa, wb = R.razmer_ves(a, "obratnyj"), R.razmer_ves(b, "obratnyj")
    assert wa[10] == wb[10]
    monkeypatch.setattr(R, "WEIGHT_LAG", -1)       # мутация: вес с текущим днём
    assert R.razmer_ves(a, "obratnyj")[10] != R.razmer_ves(b, "obratnyj")[10]


# ---- режим ----
def test_ma200_po_proshlomu():
    c = np.linspace(100.0, 101.0, 300)
    c[250] = 50.0                                   # d = 249: завтра обвал
    f = R.flag_ma200(c)
    assert f[249] and not f[250] and not f[100]     # нет истории → False
    # ручной ответ: close_249 > mean(c[50:250])
    assert f[249] == (c[249] > c[50:250].mean())
    c2 = c.copy()
    c2[251] = 1e6                                   # будущее не меняет флаг d = 249
    assert R.flag_ma200(c2)[249] == f[249]


def test_ma200_mutaciya_menyaet_flag(monkeypatch):
    c = np.full(260, 100.0)
    c[210] = 100.5                                  # d = 210 выше плоской MA, d+1 = 211 — обычный
    c[211] = 400.0                                  # завтра сильный рост поднимает MA
    honest = R.flag_ma200(c)
    assert honest[210]                              # 100.5 > среднее(c[11:211]) ≈ 100.002
    monkeypatch.setattr(R, "MA_LAG", -1)
    assert not R.flag_ma200(c)[210]                 # MA с d+1 = 114.7+ > 100.5


def _sig(n=300, seed=1):
    return np.random.default_rng(seed).uniform(0.5, 1.5, n)


def test_rank_ravno_perebor_i_bez_tekushchego_dnya():
    s = _sig()
    r = R.rank_sigma(s)
    for i in (252, 260, 299):
        assert r[i] == pytest.approx(np.mean(s[i - 252:i] < s[i]))
    assert np.isnan(r[:252]).all()
    assert not R.flag_rank(s)[:252].any()


def test_rank_mutaciya_tekushchij_den_lovitsya(monkeypatch):
    s = np.full(300, 1.0)
    s[280] = 2.0                                    # максимум: честный ранг = 1,0 → режим «нет»
    assert R.rank_sigma(s)[280] == 1.0 and not R.flag_rank(s)[280]
    monkeypatch.setattr(R, "RANK_LAG", 0)          # мутация: день d в окне (mutations.md № 8)
    assert R.rank_sigma(s)[280] != 1.0
    # окно с d: 252 значений из 253 строго меньше → 252/253; и при равных значениях сдвиг есть
    assert R.rank_sigma(s)[280] == pytest.approx(252 / 253) or R.rank_sigma(s)[280] < 1.0


def test_rank_bud_den_d_plus_1_ne_vliyaet():
    s = _sig()
    t = s.copy()
    t[281] = 99.0
    assert R.rank_sigma(s)[280] == R.rank_sigma(t)[280]


def test_korotkij_den_beret_flag_vcherashnego():
    c = np.linspace(100, 200, 300)
    nb = np.full(300, 54)
    nb[270] = 10
    f = R.rezhim_arrays(c, _sig(), nb)
    assert f["ma200"][270] == f["ma200"][269]
    assert f["net"].all() and f["rank08"].dtype == bool


def test_imoex_smoke_do_2022():
    p = R.pd.Timestamp  # noqa: F841
    from fastbt import daily_1850 as d18
    f = d18.default_main() / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet"
    if not f.exists():
        pytest.skip("нет IMOEX 10-минуток")
    x = R.imoex_daily(f)
    assert x["date"].max() < pd.Timestamp("2022-01-01")
    assert (x["close"] > 0).all() and x["sig_mkt"].dropna().gt(0).all()


# ---- события (тест 2.11) ----
CBR = pd.DataFrame({"date": [D[20]], "known_from": [D[5]]})


def test_cbr_blokiruet_den_pered_zasedaniem():
    b = S.cbr_block(D, CBR)
    assert b.sum() == 1 and b[19]                   # заседание D[20] → блок на d = D[19]


def test_cbr_known_from_posle_d_ne_blokiruet():
    late = pd.DataFrame({"date": [D[20]], "known_from": [D[19] + pd.Timedelta(days=1)]})
    assert not S.cbr_block(D, late).any()           # known_from = d+1 → не известно в d
    # мутация: фильтр по date без known_from заблокировал бы
    assert S.cbr_block(D, CBR)[19]


def test_cbr_zasedanie_v_vyhodnoj_perenos_na_sled_torgovyj():
    sat = pd.DataFrame({"date": [pd.Timestamp("2019-02-02")], "known_from": [D[0]]})  # суббота
    b = S.cbr_block(D, sat)
    nxt = D.searchsorted(pd.Timestamp("2019-02-02"))
    assert b[nxt - 1] and b.sum() == 1


DIV = pd.DataFrame({"ticker": ["AAA"], "record_date": [D[12]], "last_cum_date": [D[10]],
                    "announce_date": [D[3]]})


def test_div_blokiruet_last_cum_i_announce_pit():
    b = S.div_block(D, DIV, "AAA")
    assert b.sum() == 1 and b[10]
    assert not S.div_block(D, DIV, "BBB").any()     # у другого тикера выплат нет — пусто, не ошибка
    late = DIV.assign(announce_date=D[10] + pd.Timedelta(days=5))
    assert not S.div_block(D, late, "AAA").any()    # announce после d → событие не известно в d
    after = DIV.assign(announce_date=D[10])
    assert S.div_block(D, after, "AAA")[10]         # announce = d — известно


def test_div_os_vyklyuchena_po_umolchaniyu():
    o = S.sobytiya(D, ["AAA"], CBR, DIV)
    assert set(o) == {"net", "cbr"}
    o2 = S.sobytiya(D, ["AAA"], CBR, DIV, div_verified=True)
    assert set(o2) == {"net", "cbr", "div", "cbr_div"}
    assert o2["cbr_div"]["AAA"][10] and o2["cbr_div"]["AAA"][19]
    assert not o["net"]["AAA"].any()
    with pytest.raises(ValueError):
        S.sobytiya(D, ["AAA"], CBR, None, div_verified=True)


def test_dannye_2022_zapreshcheny():
    with pytest.raises(PermissionError):
        S.cbr_block(pd.bdate_range("2021-12-20", periods=20), CBR)


def test_realnye_fajly_cbr_div():
    dates = pd.bdate_range("2017-01-02", "2021-12-31")
    c = S.load_cbr()
    n = len(c[(c["date"] >= "2017-01-01")])
    assert 0 < n
    b = S.cbr_block(dates, c)
    assert 30 <= b.sum() <= 45                      # 40 заседаний 2017–2021 (часть на выходных → тот же счёт)
    dv = S.load_div()
    n_aflt = (dv["ticker"] == "AFLT").sum()
    assert S.div_block(dates, dv, "AFLT").sum() == n_aflt > 0   # каждая выплата — один день блока
