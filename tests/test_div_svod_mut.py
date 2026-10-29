"""Тестировщик #717: мутации, которых нет в test_div_svod.py. Реальный путь Fajly.kalendar()/zagruzit (а не подмена kalendar в тесте).
Синтетика, сеть и данные не нужны."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from fastbt import dividendy, istochniki_v6 as iv, sobytiya_e1

_p = Path(__file__).resolve().parent.parent / "scripts" / "div_svod.py"
_spec = importlib.util.spec_from_file_location("div_svod_skript_mut", _p)
ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds)


def _sv(rec, ann, godna, t="AAA"):
    return {"ticker": t, "record_date": rec, "dps_rub": "10", "announce_pit": ann, "godna": godna, "prichina": "" if godna else "x"}


class _F(iv.Fajly):
    """Fajly без чтения баров: kalendar/div_otfiltrovan — настоящие (реальный путь), ряды индексов и Brent пусты."""

    def __init__(self):
        self.end, self.main = pd.Timestamp("2022-01-01"), None

    def indeks(self, name):
        return None

    def brent(self):
        return {}, pd.DataFrame({"secid": [], "lsttrade": [], "roll": []})


@pytest.fixture
def kal(monkeypatch):
    cal = pd.bdate_range("2018-01-01", "2018-12-31").to_numpy("datetime64[ns]")
    monkeypatch.setattr(dividendy, "_calendar", lambda: cal)
    return pd.DatetimeIndex(cal)


def _svod(tmp_path, monkeypatch, rows):
    p = tmp_path / "svod.csv"
    ds.zapisat(rows, p)
    monkeypatch.setattr(sobytiya_e1, "SVOD_CSV", p)


def test_kalendar_otdayot_tolko_godnye_a_ne_polnuyu_tablicu(tmp_path, monkeypatch):
    """Мутация (г): istochniki_v6 берёт несвёрнутую исходную таблицу (все выплаты) при наличии свода."""
    _svod(tmp_path, monkeypatch, [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 0), _sv("2018-11-09", "2018-10-01", 0)])
    _, div = _F().kalendar()
    assert list(div["record_date"]) == [pd.Timestamp("2018-07-06")]
    assert len(div) < len(sobytiya_e1.load_div())              # полная таблица — 173 выплаты; в ось попадает меньшее число
    assert list(div["announce_date"]) == [pd.Timestamp("2018-05-30")]


def test_zagruzit_peredayot_div_verified_i_svernutyj_div(tmp_path, monkeypatch):
    _svod(tmp_path, monkeypatch, [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 0)])
    ist = iv.zagruzit("train4", [], fajly=_F())
    assert ist.div_verified is True and len(ist.div) == 1
    monkeypatch.setattr(sobytiya_e1, "SVOD_CSV", tmp_path / "net.csv")
    ist0 = iv.zagruzit("train4", [], fajly=_F())
    assert ist0.div_verified is False and len(ist0.div) == len(sobytiya_e1.load_div())      # свода нет: полная таблица, ось закрыта
    assert ist.sha != ist0.sha


def test_sha_zagruzit_menyaetsya_ot_svoda(tmp_path, monkeypatch):
    """Мутация (д) на реальном пути: смена свода (дата или состав) обязана менять ist.sha, из которого строится ключ кэша точки."""
    h = []
    for rows in ([_sv("2018-07-06", "2018-05-30", 1)], [_sv("2018-07-06", "2018-05-31", 1)], [_sv("2018-07-06", "2018-05-30", 1), _sv("2018-09-07", "2018-08-01", 1)]):
        _svod(tmp_path, monkeypatch, rows)
        h.append(iv.zagruzit("train4", [], fajly=_F()).sha)
    assert len(set(h)) == 3


def test_ne_vidno_do_obyavleniya_cherez_realnyj_put(tmp_path, monkeypatch, kal):
    """Мутация (е): дата объявления не используется — событие видно до announce_pit. Реальный путь: svod.csv -> load_div_svod -> sobytiya."""
    cbr = pd.DataFrame({"date": [], "known_from": []})
    _svod(tmp_path, monkeypatch, [_sv("2018-07-06", "2018-07-03", 1)])             # last_cum 04.07; объявлено 03.07 -> блок есть
    _, div = _F().kalendar()
    assert sobytiya_e1.sobytiya(kal, ["AAA"], cbr, div, True)["div"]["AAA"].sum() == 1
    _svod(tmp_path, monkeypatch, [_sv("2018-07-06", "2018-07-05", 1)])             # объявлено после last_cum -> блока нет
    _, div = _F().kalendar()
    assert sobytiya_e1.sobytiya(kal, ["AAA"], cbr, div, True)["div"]["AAA"].sum() == 0
    d = sobytiya_e1.load_div_svod(sobytiya_e1.SVOD_CSV)
    assert d["announce_date"].iloc[0] == pd.Timestamp("2018-07-05")                # announce_date = announce_pit, а не record_date
