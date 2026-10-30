"""Мутации к #893 (тестировщик): дыра индекса сектора закрыта IMOEX с флагом, Д4 для настоящих секторов не снята.
Мутации — обёртки над результатом/константами (monkeypatch); каждая обязана ронять проверку, настоящий код — проходить."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_rel_sektor as rel
from fastbt import rel_ryad as R
from fastbt.eval_v6_osi import rel_td_ind
from test_rel_dyra_sektora import DAYS, ROWS, _bars

T, M, A = _bars(1, 100.0), _bars(4, 3000.0), _bars(2, 50.0, skip=(5,))
DEN = DAYS[5]


def _den(td, d):
    r = td[pd.to_datetime(td["date"]) == d].iloc[0]
    return float(r["close"] / r["open"])


def _proverka(fn):
    """Контракт: день дыры — внутридневное движение тикера к IMOEX, не к сектору и не копия вчера; флаг 'net_barov_indeksa'."""
    got, fl = fn(T, M, {"A": A}, ROWS, "T")
    f = fl[fl["date"] == DEN].iloc[0]
    assert f["flag"] == "net_barov_indeksa" and f["benchmark"] == "IMOEX"
    ref = _den(R.td_rel(T, M, "T"), DEN)
    assert _den(got, DEN) == pytest.approx(ref, rel=1e-12)
    assert _den(got, DEN) != pytest.approx(_den(got, DAYS[4]), rel=1e-9)
    assert len(got) == 10 and (got["close"] > 0).all()


def test_nastoyashchij_kod_prohodit():
    _proverka(R.td_rel_sektor)


def test_mutatsiya_ffill_dyry_lovitsya():
    real = R.td_rel_sektor

    def ffill(*a):
        got, fl = real(*a)
        g = got.copy()
        i = g.index[pd.to_datetime(g["date"]) == DEN][0]
        g.loc[i, ["open", "high", "low", "close"]] = g.loc[i - 1, ["open", "high", "low", "close"]].to_numpy()   # вчера вместо IMOEX
        return g, fl
    with pytest.raises(AssertionError):
        _proverka(ffill)


def test_mutatsiya_flag_ne_stavitsya_lovitsya():
    real = R.td_rel_sektor

    def bez_flaga(*a):
        got, fl = real(*a)
        return got, fl.assign(flag="")
    with pytest.raises(AssertionError):
        _proverka(bez_flaga)


def test_mutatsiya_flag_drugoj_text_lovitsya(monkeypatch):
    monkeypatch.setattr(R, "F_NET_BAROV", "net_sektora")
    with pytest.raises(AssertionError):
        _proverka(R.td_rel_sektor)


def test_mutatsiya_den_po_sektoru_a_ne_imoex_lovitsya():
    real = R.td_rel_sektor

    def sektor(*a):
        got, fl = real(*a)
        return got, fl.assign(benchmark="A")
    with pytest.raises(AssertionError):
        _proverka(sektor)


def test_dyra_i_u_imoex_otkaz_ostayotsya():
    """Нет баров ни у сектора, ни у IMOEX — подстановки нет, оценка отказывает (не молчаливая дыра)."""
    M2 = _bars(4, 3000.0, skip=(5,))
    got, fl = R.td_rel_sektor(T, M2, {"A": A}, ROWS, "T")
    assert len(got) == 9
    td = type("TD", (), {"dates": DAYS.to_numpy(), "ticker": "T"})()
    with pytest.raises(ValueError):
        rel_td_ind(got, td)


def test_hvost_posle_konca_sektora_pomechen():
    """Сектор закончился на 7-й день: хвост — IMOEX с флагом (не тихо)."""
    A2 = _bars(2, 50.0)
    A2 = A2[A2["timestamp"].dt.normalize() <= DAYS[6]]
    _, fl = R.td_rel_sektor(T, M, {"A": A2}, ROWS, "T")
    assert (fl["flag"].iloc[7:] == R.F_NET_BAROV).all() and (fl["flag"].iloc[:7] == "").all()


STAROE = {"T": [(pd.Timestamp("2017-03-20"), "IDX")]}


def _ctx(pit):
    dates = pd.bdate_range("2014-01-01", "2026-02-27")
    s = pd.Series(np.full(len(dates), 50.0), index=dates)
    return rel.Ctx({"IDX": s}, {"T": s}, {"T": s * 0 + 0.01}, pit)


def test_staryj_nastoyashchij_sektor_otkaz():
    with pytest.raises(ValueError, match="Д4"):
        _ctx(STAROE).rel("T", 1)


def test_mutatsiya_proverka_snyata_dlya_vseh_lovitsya(monkeypatch):
    monkeypatch.setattr(rel, "DAWN", pd.Timestamp("2100-01-01"))      # все записи «постоянные» — проверка выключена
    with pytest.raises(pytest.fail.Exception):
        with pytest.raises(ValueError, match="Д4"):
            _ctx(STAROE).rel("T", 1)
        pytest.fail("Д4 не сработала")


def test_dawn_ne_pervaya_zapis_no_poslednyaya_staraya_sektor_otkaz():
    """Запись DAWN в начале + настоящий устаревший сектор позже: проверка к последней записи — отказ."""
    pit = {"T": [(rel.DAWN, "IDX"), (pd.Timestamp("2017-03-20"), "IDX")]}
    with pytest.raises(ValueError, match="Д4"):
        _ctx(pit).rel("T", 1)
