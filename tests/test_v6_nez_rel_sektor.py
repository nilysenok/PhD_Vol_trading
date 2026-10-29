"""Независимый тест (тестировщик, не автор) V6-1b (#662): rel_ryad — сектор PIT, откат на IMOEX с флагами, режимы, Д1.
Синтетика с известным ответом: цена в дне плоская, поэтому дневная лог-доходность относительного ряда = r_тикера - r_бенчмарка дня
(ночной переход при смене бенчмарка берётся по новому бенчмарку). Данных и сети нет (CSV состава - файл в docs/, даты < 2022)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18
from fastbt import rel_ryad as R
from fastbt.e1_rel_sektor import sektor_pit

ND = 12
DAYS = pd.bdate_range("2021-03-01", periods=ND)
F_PERV, F_NET_S, F_NET_I = "do_pervoj_rebalansirovki", "net_sektora", "net_indeksa"


def ryad(seed, p0, sigma=0.01):
    rng = np.random.default_rng(seed)
    return p0 * np.exp(np.cumsum(rng.normal(0, sigma, ND)))


def bary(p, extra_late=None):
    """54 бара 09:50...18:40 на день; цена плоская = p[d]. extra_late - цены баров 18:50, 19:00 (должны игнорироваться)."""
    rows = []
    for d, x in zip(DAYS, p):
        for ts in pd.date_range(d + pd.Timedelta(hours=9, minutes=50), d + pd.Timedelta(hours=18, minutes=40), freq="10min"):
            rows.append((ts, x, x, x, x, 100))
        if extra_late is not None:
            for ts in pd.date_range(d + pd.Timedelta(hours=18, minutes=50), d + pd.Timedelta(hours=19), freq="10min"):
                rows.append((ts, extra_late[d], extra_late[d], extra_late[d], extra_late[d], 100))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


@pytest.fixture
def mir():
    p = {"T": ryad(1, 100), "A": ryad(2, 50), "B": ryad(3, 70), "M": ryad(4, 3000)}
    return p, {k: bary(v) for k, v in p.items()}


def r_log(p):
    return np.diff(np.log(p))


def ozhid(p, kto):
    """Ожидаемые дневные лог-доходности rel для дней 1..ND-1: r_T - r_bench(день); kto[d] - ключ бенчмарка дня d."""
    return np.array([r_log(p["T"])[d - 1] - r_log(p[kto[d]])[d - 1] for d in range(1, ND)])


def fakt(td):
    return np.diff(np.log(td["close"].to_numpy(float)))


# --- известный ответ: смена состава сектора ---------------------------------------------------------------------------
def test_smena_sostava_sektora_izvestnyj_otvet(mir):
    p, b = mir
    S = 6
    rows = [(pd.Timestamp("2021-01-01"), "A"), (DAYS[S], "B")]
    td, fl = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"], "B": b["B"]}, rows, "T")
    kto = ["A" if d < S else "B" for d in range(ND)]
    assert len(td) == ND
    np.testing.assert_allclose(fakt(td), ozhid(p, kto), atol=1e-9)
    assert list(fl["benchmark"]) == kto and (fl["flag"] == "").all()
    # sanity: ответ с неверным днём смены отличается (тест различает)
    kto_bad = ["A" if d < S - 1 else "B" for d in range(ND)]
    assert np.abs(fakt(td) - ozhid(p, kto_bad)).max() > 1e-3


def test_smena_sostava_rebalans_v_den_s_mozhet_byt_tolko_s_etogo_dnya(mir):
    """PIT: состав новой ребалансировки не действует раньше её даты, ни на день."""
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A"), (DAYS[8], "B")]
    rez = [R.indeks_dnya(rows, d, {"A", "B"}) for d in DAYS]
    assert [x[0] for x in rez] == ["A"] * 8 + ["B"] * (ND - 8)
    assert R.indeks_dnya(rows, DAYS[8] - pd.Timedelta(days=1), {"A", "B"})[0] == "A"
    assert R.indeks_dnya(rows, DAYS[8] + pd.Timedelta(days=3), {"A", "B"})[0] == "B"
    td, _ = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"], "B": b["B"]}, rows, "T")
    kto = ["A"] * 8 + ["B"] * (ND - 8)
    np.testing.assert_allclose(fakt(td), ozhid(p, kto), atol=1e-9)


def test_tri_rebalansirovki_a_b_a(mir):
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A"), (DAYS[4], "B"), (DAYS[8], "A")]
    td, fl = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"], "B": b["B"]}, rows, "T")
    kto = ["A"] * 4 + ["B"] * 4 + ["A"] * (ND - 8)
    np.testing.assert_allclose(fakt(td), ozhid(p, kto), atol=1e-9)
    assert list(fl["benchmark"]) == kto


# --- откат на IMOEX с флагами -----------------------------------------------------------------------------------------
def test_otkat_do_pervoj_rebalansirovki(mir):
    p, b = mir
    rows = [(DAYS[5], "A")]
    td, fl = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"]}, rows, "T")
    kto = ["M"] * 5 + ["A"] * (ND - 5)
    np.testing.assert_allclose(fakt(td), ozhid(p, kto), atol=1e-9)
    assert list(fl["flag"]) == [F_PERV] * 5 + [""] * (ND - 5)
    assert list(fl["benchmark"]) == ["IMOEX"] * 5 + ["A"] * (ND - 5)


def test_otkat_net_sektora_i_net_indeksa_s_flagami(mir):
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A"), (DAYS[3], None), (DAYS[6], "Z"), (DAYS[9], "B")]
    td, fl = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"], "B": b["B"]}, rows, "T")
    kto = ["A"] * 3 + ["M"] * 6 + ["B"] * 3
    np.testing.assert_allclose(fakt(td), ozhid(p, kto), atol=1e-9)
    assert list(fl["flag"]) == [""] * 3 + [F_NET_S] * 3 + [F_NET_I] * 3 + [""] * 3
    assert list(fl["benchmark"]) == ["A"] * 3 + ["IMOEX"] * 6 + ["B"] * 3


def test_otkat_ne_molchalivyj_flag_tolko_na_otkatnykh_dnyakh(mir):
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A")]
    _, fl = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"]}, rows, "T")
    assert (fl["flag"] == "").all() and (fl["benchmark"] == "A").all()
    _, fl = R.td_rel_sektor(b["T"], b["M"], {}, rows, "T")                 # индекса A нет в данных
    assert (fl["flag"] == F_NET_I).all() and (fl["benchmark"] == "IMOEX").all()


def test_rezhimy_syroy_rynok_sektor_i_oshibki(mir):
    p, b = mir
    td, fl = R.td_rel_rezhim("syroy", b["T"], "T")
    pd.testing.assert_frame_equal(td, d18.aggregate(b["T"], "T"), check_exact=True)
    assert fl is None
    td, fl = R.td_rel_rezhim("rynok", b["T"], "T", imoex=b["M"])
    np.testing.assert_allclose(fakt(td), ozhid(p, ["M"] * ND), atol=1e-9)
    assert fl is None
    td, fl = R.td_rel_rezhim("sektor", b["T"], "T", imoex=b["M"], indeksy={"A": b["A"]}, rows=[(pd.Timestamp("2021-01-01"), "A")])
    np.testing.assert_allclose(fakt(td), ozhid(p, ["A"] * ND), atol=1e-9)
    assert fl is not None and len(fl) == ND
    for rez in ("rynok", "sektor"):
        with pytest.raises(ValueError):
            R.td_rel_rezhim(rez, b["T"], "T")                              # молчаливое syroy запрещено
    with pytest.raises(ValueError):
        R.td_rel_rezhim("beta", b["T"], "T", imoex=b["M"])


def test_znachenie_dnya_ne_zavisit_ot_barov_posle_1850(mir):
    p, b = mir
    pozdnie = {d: 7777.0 * (i + 1) for i, d in enumerate(DAYS)}
    t2 = bary(p["T"], extra_late=pozdnie)
    a, _ = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"]}, [(DAYS[0], "A")], "T")
    c, _ = R.td_rel_sektor(t2, b["M"], {"A": b["A"]}, [(DAYS[0], "A")], "T")
    pd.testing.assert_frame_equal(a, c, check_exact=True)


# --- реальный состав (CSV < 2022) -------------------------------------------------------------------------------------
def test_real_sostav_pit_bez_zagljadyvaniya_vpered():
    pit = sektor_pit()
    assert len(pit) == 17
    for t, rows in pit.items():
        ds = [d for d, _ in rows]
        assert ds == sorted(ds) and ds[0] == pd.Timestamp("2017-03-20"), t
        assert R.indeks_dnya(rows, pd.Timestamp("2017-03-19"), {n for _, n in rows if n})[1] == F_PERV
    rows = pit["AFLT"]
    have = {n for _, n in rows if n}
    ind, fl = R.indeks_dnya(rows, pd.Timestamp("2017-06-18"), have)
    assert (ind, fl) == (rows[0][1], "") and rows[1][0] == pd.Timestamp("2017-06-19")
    assert R.indeks_dnya(rows, pd.Timestamp("2017-06-19"), have)[0] == rows[1][1]
    assert R.indeks_dnya(pit["LSRG"], pd.Timestamp("2018-01-10"), {"IMOEX"}) == (None, F_NET_S)


# --- Д1 ---------------------------------------------------------------------------------------------------------------
def _ref(p, kto):
    ref = pd.Series(ozhid(p, kto), index=DAYS[1:])
    return ref


def test_d1_pik_na_lag_0_na_pravilnom_ryade(mir):
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A"), (DAYS[6], "B")]
    td, _ = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"], "B": b["B"]}, rows, "T")
    kto = ["A"] * 6 + ["B"] * (ND - 6)
    res = R.proverka_d1(td, _ref(p, kto))
    assert set(res) == {-2, -1, 0, 1} and max(res, key=res.get) == 0 and res[0] == pytest.approx(1.0, abs=1e-6)


def test_d1_blokiruet_utechku_i_sdvig(mir):
    p, b = mir
    rows = [(pd.Timestamp("2021-01-01"), "A")]
    td, _ = R.td_rel_sektor(b["T"], b["M"], {"A": b["A"]}, rows, "T")
    ref = _ref(p, ["A"] * ND)
    # ряд тянет значение «завтрашнего» дня: x_d = ref_{d+1}  -> пик на lag +1
    utechka = td.copy()
    lc = np.r_[0.0, np.cumsum(ref.shift(-1).fillna(0.0).to_numpy())]
    utechka["close"] = 100 * np.exp(lc)
    with pytest.raises(ValueError):
        R.proverka_d1(utechka, ref)
    # ряд запаздывает на день: x_d = ref_{d-1} -> пик на lag -1
    pozdno = td.copy()
    lc = np.r_[0.0, np.cumsum(ref.shift(1).fillna(0.0).to_numpy())]
    pozdno["close"] = 100 * np.exp(lc)
    with pytest.raises(ValueError):
        R.proverka_d1(pozdno, ref)
    with pytest.raises(ValueError):                                          # мало наблюдений - корреляция не определена
        R.proverka_d1(td.iloc[:2], ref)
