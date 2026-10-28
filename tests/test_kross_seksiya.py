"""Кросс-секционные книги S7–S9 (fastbt/kross_seksiya.py): синтетика с известным ответом и 5 мутаций (Ж-096 п. 5б).
Мутации: ранг с днём d+1; книга не нейтральна по стоимости; сектор по будущей разметке; исполнение по цене дня d;
шорт без платы. Каждая подложена в тесте и обязана уронить проверку, которая на правильном коде проходит."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kross_seksiya as ks

N, T = 8, 320
IDX = pd.bdate_range("2018-01-01", periods=T)
COLS = [f"T{i}" for i in range(N)]


def _ceny(seed=1):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, N)), axis=0)), index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, T))), index=IDX)
    return c, m


def _skor_ozhid(x_row):
    p = x_row.rank().sub(1) / (len(x_row) - 1)
    return 2 * p - 1


def test_s7_znak_i_ranzh():
    c, m = _ceny()
    d = 200
    x = (c.iloc[d] / c.iloc[d - 5] - 1) - (m.iloc[d] / m.iloc[d - 5] - 1)
    s = ks.skor_s7(c, m).iloc[d]
    assert np.allclose(s, -_skor_ozhid(x))                      # худшие за 5 дней — лонг (+1), лучшие — шорт (−1)
    assert s[x.idxmin()] == 1 and s[x.idxmax()] == -1


def test_s8_po_trendu_12_1():
    tr = np.linspace(-0.001, 0.002, N)
    c = pd.DataFrame(100 * np.exp(np.outer(np.arange(T), tr)), index=IDX, columns=COLS)
    s = ks.skor_s8(c).iloc[300]
    assert np.allclose(s, 2 * np.arange(N) / (N - 1) - 1) and s["T7"] == 1


def test_pustoe_sechenie_dayot_nan_i_nolyu():
    c, m = _ceny()
    c.iloc[200, 3:] = np.nan                                    # меньше MIN_TIK наблюдений
    assert ks.skor_s7(c, m).iloc[200].isna().all()
    W = ks.vesa(np.full((3, N), np.nan), np.ones((3, N)), np.full((3, N), "a", dtype=object), np.ones((3, N), bool))
    assert not W.any()


def _bez_budushchego(fn, nov, d=290):
    """Скор дня d не меняется, если испортить цены, которых сигнал по определению не видит (строки ≥ d + nov)."""
    c, m = _ceny()
    r = np.random.default_rng(9)
    c2, m2 = c.copy(), m.copy()
    c2.iloc[d + nov:] *= np.exp(r.normal(0, 0.2, (T - d - nov, N)))
    m2.iloc[d + nov:] *= np.exp(r.normal(0, 0.2, T - d - nov))
    assert np.allclose(fn(c, m).iloc[d], fn(c2, m2).iloc[d], equal_nan=True), "скор дня d зависит от лишних строк"


def test_ranzh_bez_dnya_d_plyus_1_i_mutaciya():
    metki = np.full((T, N), "", dtype=object)
    # (сигнал, первая строка, которую он не видит): S7, S9 — d+1; S8 берёт цены d−21 и d−252, строки ≥ d−20 ему не нужны
    for f, nov in ((ks.skor_s7, 1), (lambda c, m: ks.skor_s8(c), -20), (lambda c, m: ks.skor_s9(c, m, metki), 1)):
        _bez_budushchego(f, nov)
        with pytest.raises(AssertionError):                     # мутация: ранг дня d+1 вместо d
            _bez_budushchego(lambda c, m, f=f: f(c, m).shift(-1), nov)


KARTA = pd.DataFrame({"ticker": ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"],
                      "indeks": ["X", "X", "Y", "Y", "Z", "Z", "W", "V"], "s": pd.Timestamp("2000-01-01")})


def test_s9_otnositelno_sektora_izvestnyy_otvet():
    c, m = _ceny()
    metki = ks.sektor_na_d(KARTA, IDX, COLS)
    d = 250
    r = c.iloc[d] / c.iloc[d - 62] - 1
    rm = m.iloc[d] / m.iloc[d - 62] - 1
    sek = pd.Series({"T0": r.T1, "T1": r.T0, "T2": r.T3, "T3": r.T2, "T4": r.T5, "T5": r.T4, "T6": rm, "T7": rm})
    assert np.allclose(ks.skor_s9(c, m, metki).iloc[d], _skor_ozhid(r - sek))


def _proverka_sektora(fn):
    karta = pd.concat([KARTA, pd.DataFrame({"ticker": ["T0"], "indeks": ["Y"], "s": [IDX[200]]})], ignore_index=True)
    lab = fn(karta, IDX, COLS)
    assert lab[100, 0] == "X" and lab[250, 0] == "Y" and lab[250, 1] == "X"     # до даты записи — старая метка


def test_sektor_na_den_i_mutaciya_budushchey_razmetki():
    _proverka_sektora(ks.sektor_na_d)

    def budushchaya(karta, dates, tickers):                      # мутация: последняя запись для всех дней
        lab = ks.sektor_na_d(karta, dates, tickers)
        return np.repeat(lab[-1:], len(dates), axis=0)
    with pytest.raises(AssertionError):
        _proverka_sektora(budushchaya)


def _sechenie():
    s = np.array([[0.9, 0.8, 0.7, 0.6, 0.0, -0.6, -0.7, -0.9]])
    return s, np.array([[1, 1, 1, 1, 1, 1, 1, 2.0]]), np.ones((1, 8), bool)


def test_kniga_neytralna_po_stoimosti_i_limit_sektora():
    s, sg, ok = _sechenie()
    one = np.full((1, 8), "a", dtype=object)
    W = ks.vesa(s, sg, one, ok)[0]
    assert W[W > 0].sum() == pytest.approx(0.5) and W[W < 0].sum() == pytest.approx(-0.5)
    assert (W > 0).sum() == 3 and (W < 0).sum() == 3            # 4 лонга-кандидата в одном секторе → не больше 3
    assert W[5] / W[6] == pytest.approx(0.6 / 0.7) and W[7] / W[6] == pytest.approx((0.9 / 2) / 0.7)   # вес ∝ |скор|/σ
    assert not ks.vesa(s * 0 + 0.9, sg, one, ok).any()          # только лонги — нет книги


def _neytralnost(W):
    return ks.kniga(W, np.full((12, 8), 100.0), np.zeros((12, 8)), IDX[:12], np.zeros(12), 2, 1, 0)


def test_neytralnost_po_dnyam_i_mutaciya():
    s, sg, ok = _sechenie()
    W = np.repeat(ks.vesa(s, sg, np.full((1, 8), "a", dtype=object), ok), 12, axis=0)

    def proverka(Wx):
        o = _neytralnost(Wx)
        assert np.allclose(o["lon"], o["sht"]) and o["gross"].max() > 0
    proverka(W)
    with pytest.raises(AssertionError):                          # мутация: лонг-ноги вдвое тяжелее
        proverka(np.where(W > 0, 2 * W, W))


DATES = pd.bdate_range("2019-01-10", periods=8)                  # чт, пт, пн, ...


def _prostoy():
    P = np.full((8, 2), 100.0)
    P[2:, 0], P[:, 1], P[3:, 1] = 110.0, 50.0, 55.0
    W = np.zeros((8, 2))
    W[0] = [0.5, -0.5]
    return W, P


def _proverka_knigi(o):
    gap1 = 3 / 365                                               # пт → пн: шорт платит 3 календарных дня
    assert np.allclose(o["pnl"][:4], [0, 0.05, -0.05, 0])        # вход open[1] (d+1), не open[0]
    assert o["kom"][1] == pytest.approx(0.0004) and o["kom"][2] == pytest.approx(0.0004 * 1.1)
    assert o["zaem"][1] == pytest.approx(0.5 * 0.05 * gap1) and o["zaem"][2] == pytest.approx(0.5 * 0.05 / 365)
    assert o["stavka"][1] == pytest.approx(0.1 * 3 / 365)
    assert o["excess"][1] == pytest.approx(0.05 - 0.0004 - 0.5 * 0.05 * gap1 - 0.1 * gap1)
    assert o["vhody"][1] == 2 and o["vhody"].sum() == 2


def _kniga_p(**kw):
    W, P = _prostoy()
    return ks.kniga(W, P, np.zeros((8, 2)), DATES, np.full(8, 0.1), 2, 1, 0, **kw)


def test_kniga_izvestnyy_otvet():
    _proverka_knigi(_kniga_p())


def test_mutaciya_ispolneniya_po_cene_dnya_d(monkeypatch):
    monkeypatch.setattr(ks, "LAG", 0)
    with pytest.raises(AssertionError):
        _proverka_knigi(_kniga_p())


def test_mutaciya_short_bez_platy():
    with pytest.raises(AssertionError):
        _proverka_knigi(_kniga_p(zaem=0.0))


def test_kniga_konec_dannyh_platit_vyhod_i_pustaya_lenta():
    W, P = _prostoy()
    o = ks.kniga(W, P[:3], np.zeros((3, 2)), DATES[:3], np.zeros(3), 10, 1, 0)   # τ длиннее ленты
    assert o["kom"][1] > 0 and o["kom"][0] == 0                  # выход на последнем open платит комиссию
    z = ks.kniga(np.zeros((8, 2)), P, np.zeros((8, 2)), DATES, np.full(8, 0.1), 2, 1, 0)
    assert not z["excess"].any() and not np.isnan(z["excess"]).any()


def test_dividend_shorta_vychitaetsya():
    W, P = _prostoy()
    D = np.zeros((8, 2))
    D[2, 1] = 1.0                                                # ex-день в t=2: шорт платит 1 ₽ с 50
    o = ks.kniga(W, P, D, DATES, np.zeros(8), 2, 1, 0)
    assert o["pnl"][1] == pytest.approx(0.05 - 0.5 * 1.0 / 50)


def test_k2_sdvig_nol_toshdestvo_i_sechenie_sohranyaetsya():
    s = np.random.default_rng(3).normal(size=(50, 4))
    assert np.array_equal(ks.k2_skor(s, 0), s)
    k = ks.k2_skor(s, 17)
    assert not np.array_equal(k, s) and np.allclose(np.sort(k, axis=0), np.sort(s, axis=0))
    assert np.array_equal(k[17], s[0])
