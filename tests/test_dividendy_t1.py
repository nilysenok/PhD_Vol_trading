"""Правило last_cum по дате реестра: T+2 до T1_OT (31.07.2023), T+1 с T1_OT. Синтетика, календарь баров подменён.
dev-digest побайтно прежний (значение до правки зафиксировано константой)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy, zona

DEV_DIGEST_DO_PRAVKI = "820d4cf772c63aff1158c1ca15a835b7b481a98c"   # digest() кода PR #535 в зоне dev


@pytest.fixture(autouse=True)
def _kal(monkeypatch):
    cal = pd.bdate_range("2023-07-10", "2023-08-31").to_numpy("datetime64[ns]")
    monkeypatch.setattr(dividendy, "_calendar", lambda: cal)
    yield
    zona.set_zone("dev")


def _lc(*dates):
    s = pd.Series(pd.to_datetime(list(dates)))
    return [str(x.date()) for x in dividendy.last_cum(s)]


def test_reestr_do_t1_t_plus_2():
    assert _lc("2023-07-20") == ["2023-07-18"]


def test_reestr_posle_t1_t_plus_1():
    assert _lc("2023-08-15") == ["2023-08-14"]


def test_reestr_v_vyhodnoy_predydushchiy_torgovyy_zatem_t_plus_1():
    assert _lc("2023-08-12") == ["2023-08-10"]      # сб -> пт 11.08, T+1 -> чт 10.08
    assert _lc("2023-07-15") == ["2023-07-12"]      # сб -> пт 14.07, T+2 -> ср 12.07


def test_granitsa_t1_vklyuchitelno():
    assert _lc("2023-07-31") == ["2023-07-28"]      # пн 31.07, T+1 -> пт 28.07
    assert _lc("2023-07-28") == ["2023-07-26"]      # пт 28.07 < T1_OT, T+2


def test_dev_digest_pobaytno_prezhniy():
    assert zona.name() == "dev"
    assert dividendy.digest() == DEV_DIGEST_DO_PRAVKI


def test_test1_digest_otlichaetsya_ot_dev(monkeypatch):
    monkeypatch.setenv("FASTBT_Z2_DIV", "NET")
    with zona.zone_ctx("test1"):
        assert dividendy.digest() != DEV_DIGEST_DO_PRAVKI


def test_mutatsiya_t2_dlya_vseh_let_lovitsya(monkeypatch):
    monkeypatch.setattr(dividendy, "T1_OT", "2262-01-01")
    with pytest.raises(AssertionError):
        test_reestr_posle_t1_t_plus_1()


def test_mutatsiya_t1_dlya_vseh_let_lovitsya(monkeypatch):
    monkeypatch.setattr(dividendy, "T1_OT", "2000-01-01")
    with pytest.raises(AssertionError):
        test_reestr_do_t1_t_plus_2()
