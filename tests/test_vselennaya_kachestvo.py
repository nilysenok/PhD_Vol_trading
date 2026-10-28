"""Правило качества данных и вселенная (Ж-114 п. 1): синтетика, без данных проекта и сети."""
import pandas as pd
import pytest

from fastbt import vselennaya_kachestvo as vk

OK, BAD = 50, 29


def _cal(start="2016-01-04", n=260):
    return pd.bdate_range(start, periods=n)


def _first_i(dates, month):
    return int(dates.searchsorted(pd.Timestamp(month + "-01")))


def _kach(dates, bad_by_ticker, extra=()):
    """Тикеры из bad_by_ticker (позиции плохих дней) + extra — безупречные."""
    df = pd.DataFrame(index=dates)
    for t, bad in bad_by_ticker.items():
        s = pd.Series(OK, index=dates)
        s.iloc[list(bad)] = BAD
        df[t] = s
    for t in extra:
        df[t] = OK
    return vk.Kachestvo(df)


def test_valid_den_porog_30_barov():
    d = _cal(n=5)
    df = pd.DataFrame({"AAA": [30, 29, 0, 31, 100]}, index=d)
    k = vk.Kachestvo(df)
    assert [k.valid_den("AAA", x) for x in d] == [True, False, False, True, True]
    assert k.valid_den("AAA", pd.Timestamp("2016-01-09")) is False  # не торговый день


def test_vhod_pri_95_i_gisterezis():
    d = _cal()
    i = _first_i(d, "2016-06")
    k57 = _kach(d, {"AAA": range(i - 60, i - 57)})   # 3 плохих из 60 -> 57/60 = 0,95 -> вход
    k56 = _kach(d, {"AAA": range(i - 60, i - 56)})   # 4 плохих -> 0,933 -> ещё не вошёл
    assert k57.vo_vselennoj("AAA", d[i]) and k57.vo_vselennoj("AAA", d[i + 5])
    assert not k56.vo_vselennoj("AAA", d[i])


def test_vyhod_pri_menshe_80_i_vozvrat():
    d = _cal(n=400)
    i = _first_i(d, "2016-06")
    ok = _kach(d, {"AAA": range(i - 12, i)})          # 48/60 = 0,80: не ниже 80 -> остаётся
    assert ok.vo_vselennoj("AAA", d[_first_i(d, "2016-05")]) and ok.vo_vselennoj("AAA", d[i])
    k = _kach(d, {"AAA": range(i - 13, i)})           # 47/60 < 0,80 -> выходит
    assert k.vo_vselennoj("AAA", d[_first_i(d, "2016-05")])
    assert not k.vo_vselennoj("AAA", d[i]) and not k.vo_vselennoj("AAA", d[_first_i(d, "2016-08")])
    # полоса вышла из 60-дневного окна -> возврат по тому же правилу (>= 95 %)
    assert k.vo_vselennoj("AAA", d[_first_i(d, "2016-09")])


def test_mutaciya_porog_vyhoda_95_vmesto_80_lovitsya(monkeypatch):
    d = _cal()
    i = _first_i(d, "2016-06")
    j = _first_i(d, "2016-07")
    bad2 = {"AAA": list(range(i - 3, i)) + list(range(j - 6, j))}  # в июле доля 51/60 = 0,85: между 80 и 95
    assert _kach(d, bad2).vo_vselennoj("AAA", d[j]) is True        # был в мае, гистерезис держит
    monkeypatch.setattr(vk, "EXIT", 0.95)
    assert _kach(d, bad2).vo_vselennoj("AAA", d[j]) is False


def test_okno_ne_vklyuchaet_den_resheniya_i_budushchee():
    d = _cal()
    i = _first_i(d, "2016-06")
    plohie = list(range(i - 3, i))
    base = _kach(d, {"AAA": plohie})
    assert base.vo_vselennoj("AAA", d[i])
    # плохой день решения и дни после него состав июня не меняют
    k = _kach(d, {"AAA": plohie + [i, i + 1, i + 2, i + 3]})
    assert k.vo_vselennoj("AAA", d[i]) and k.vo_vselennoj("AAA", d[i + 10])


def test_mutaciya_okno_s_dnem_d_lovitsya(monkeypatch):
    d = _cal(n=130)
    i = _first_i(d, "2016-04")      # первое решение, где уже есть 60 дней истории: вход без гистерезиса
    assert i >= vk.WINDOW
    bad = {"AAA": list(range(i - 3, i)) + [i]}
    assert _kach(d, bad).vo_vselennoj("AAA", d[i]) is True
    monkeypatch.setattr(vk, "_okno", lambda p: slice(p - vk.WINDOW + 1, p + 1))
    assert _kach(d, bad).vo_vselennoj("AAA", d[i]) is False


def test_irao_dekabr_fevral_nevalidny_i_mutaciya(monkeypatch):
    d = pd.bdate_range("2014-06-02", "2015-04-30")
    df = pd.DataFrame({"IRAO": OK, "SBER": OK}, index=d)
    k = vk.Kachestvo(df)
    assert not k.valid_den("IRAO", "2014-12-01") and not k.valid_den("IRAO", "2015-02-27")
    assert k.valid_den("IRAO", "2014-11-28") and k.valid_den("IRAO", "2015-03-02")
    assert k.valid_den("SBER", "2014-12-01")
    assert k.vo_vselennoj("IRAO", "2015-03-02") is False   # доля после дыры < 80 %
    monkeypatch.setattr(vk, "IRAO_BAD", {})
    assert vk.Kachestvo(df).valid_den("IRAO", "2014-12-01") is True


def _vselennaya(n):
    d = _cal()
    return vk.Kachestvo(pd.DataFrame({f"T{j:02d}": OK for j in range(n)}, index=d)), d


def test_s7_s9_minimum_10_tikerov():
    k10, d = _vselennaya(10)
    k9, _ = _vselennaya(9)
    day = d[-1]
    assert len(k10.sostav(day)) == 10 and k10.den_ok_s7_s9(day)
    assert len(k9.sostav(day)) == 9 and not k9.den_ok_s7_s9(day)


def test_mutaciya_s7_s9_schitaet_den_s_9_tikerami(monkeypatch):
    k9, d = _vselennaya(9)
    assert not k9.den_ok_s7_s9(d[-1])
    monkeypatch.setattr(vk, "MIN_TIK", 9)
    assert k9.den_ok_s7_s9(d[-1])  # мутант пропускает день, которого быть не должно


def test_nachalo_bez_60_dnei_vne_vselennoj():
    k, d = _vselennaya(3)
    assert not k.vo_vselennoj("T00", d[5])      # истории < 60 дней — доли нет, вне вселенной


def test_pustaya_lenta_i_nulevoj_obyom_vne_vselennoj():
    d = _cal()
    df = pd.DataFrame({"AAA": 0, "BBB": OK}, index=d)
    k = vk.Kachestvo(df)
    assert not k.vo_vselennoj("AAA", d[-1]) and k.vo_vselennoj("BBB", d[-1])
    tab = k.sostav_po_mesyacam()
    assert set(tab.columns) == {"mesyac", "tiker", "vo_vselennoj", "dolya_60"}
    assert (tab[tab.tiker == "AAA"].vo_vselennoj == 0).all()
    iv = k.intervaly_vne()
    assert list(iv[iv.tiker == "AAA"].mesyacev)[0] == tab[tab.tiker == "AAA"].shape[0]


@pytest.mark.parametrize("n", [29, 30])
def test_granica_30_barov(n):
    d = _cal(n=3)
    k = vk.Kachestvo(pd.DataFrame({"AAA": n}, index=d))
    assert k.valid_den("AAA", d[0]) is (n >= 30)
