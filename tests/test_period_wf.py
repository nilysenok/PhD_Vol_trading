"""Б8: период — один объект (train4, v2, wf); оценщик отдаёт все кварталы периода; wf закрыт замком. Синтетика, данных и сети нет.
Мутация (граница периода): первый квартал периода выпал — G5."""
import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_eval_v2 as ev, ledger_v2 as lm, period_v2 as pv
from fastbt.metriki_e import sr_dnevnoy


def ryad(d0, d1, seed=0):
    d = np.arange(np.datetime64(d0), np.datetime64(d1) + 1)
    d = d[np.is_busday(d)]
    return d, np.random.default_rng(seed).normal(1e-4, 1e-2, d.size)


def pryamoy(d, x, metka):
    s = pd.PeriodIndex(pd.DatetimeIndex(d), freq="Q").astype(str) == metka
    return float(sr_dnevnoy(x[s]) * np.sqrt(252))


def test_g1_v2_otdayot_2015():
    d, x = ryad("2015-01-01", "2019-12-31")
    kv = ev.kvartaly_perioda("v2")(d, x)
    assert list(kv)[:4] == ["2015Q1", "2015Q2", "2015Q3", "2015Q4"] and len(kv) == 20
    assert not np.isnan(list(kv.values())).any()


def test_g2_train4_prezhniy():
    d, x = ryad("2016-01-01", "2019-12-31", 1)
    assert ev.kvartaly_perioda("train4") is lm.kvartaly
    kv = ev.kvartaly_perioda("train4")(d, x)
    assert len(kv) == 16 and next(iter(kv)) == "2016Q1" and kv == lm.kvartaly(d, x)
    assert lm.gody_perioda("train4") == lm.GODY and lm.gody_perioda(None) == lm.GODY and lm.gody_perioda("v2") == lm.GODY_V2


def test_g3_wf_45_kvartalov_summy_ravny_pryamomu():
    d, x = ryad("2015-01-01", "2026-02-28", 2)
    kv = ev.kvartaly_perioda("wf")(d, x)
    assert len(kv) == 45 and list(kv)[0] == "2015Q1" and list(kv)[-1] == "2026Q1"
    for k, v in kv.items():
        assert v == pytest.approx(pryamoy(d, x, k), rel=1e-12)
    g = lm.gody_perioda("wf")
    assert g == tuple(range(2015, 2027))
    s = lm.summy(d, x, gody=g)
    for j, k in enumerate(pv.kvartaly_spisok("wf")):
        m = pd.PeriodIndex(pd.DatetimeIndex(d), freq="Q").astype(str) == k
        assert s[j, stats_FS()] == pytest.approx(x[m].sum(), rel=1e-12, abs=1e-15)
    assert len(ev.metki_perioda("wf")) == 45 and len(ev.metki_perioda("train4")) == 16


def stats_FS():
    from fastbt import stats
    return stats.F_S


def _bez_otmetki(monkeypatch, tmp_path):
    """Отметка снятого замка теперь лежит в репозитории: «без отметки» моделируется несуществующим путём во всех вызовах
    proverit_chtenie без явного otmetka (значение по умолчанию связано при определении, поэтому подмена OTMETKA не работает)."""
    import functools
    net = tmp_path / "net.json"
    monkeypatch.setattr(pv, "proverit_chtenie", functools.partial(pv.proverit_chtenie, otmetka=net))
    return net


def test_g4_wf_do_snyatiya_zamka_otkaz(tmp_path, monkeypatch):
    otm = _bez_otmetki(monkeypatch, tmp_path)
    with pytest.raises(pv.ZamokVladeltsa, match="замок"):
        pv.proverit_chtenie("wf", otmetka=otm)
    with pytest.raises(pv.ZamokVladeltsa):
        pv.proverit_chtenie("v2", put="data/regen/ch4/IMOEX_2023.parquet", otmetka=otm)
    with pytest.raises(pv.ZamokVladeltsa):
        pv.proverit_chtenie("v2", put="test_locked/x.csv", otmetka=otm)
    with pytest.raises(pv.ZamokVladeltsa):
        ev.EvaluatorV2.evaluate_v2(object(), {}, "wf")            # раньше любого чтения; отметка подменена на несуществующую
    pv.proverit_chtenie("train4", otmetka=otm)                     # ≤ 2021 открыты
    pv.proverit_chtenie("v2", put="data/regen/ch4_dev/sigma_2021.csv", otmetka=otm)
    otm.write_text("{}")
    pv.proverit_chtenie("wf", otmetka=otm)                         # с отметкой владельца — проходит


def test_g4_progon_otkazyvaet_wf(tmp_path, monkeypatch):
    """Проверяет: без отметки proverit_v5 отказывает ZamokVladeltsa (замок раньше проверки periody куба). С отметкой в
    репозитории wf упирается в ValueError «не объявлен в кубе (periody)» — это чинит лист объявления wf в кубах."""
    import fastbt.progon_sobol as ps
    _bez_otmetki(monkeypatch, tmp_path)
    assert "wf" in ps.OKNO and ps.OKNO["wf"] == ("2015-01-01", "2014-06-01")
    assert ps.OKNO["train4"] == ("2016-01-01", "2015-01-01") and ps.OKNO["v2"] == ("2015-01-01", "2014-08-26")
    rz = ps.Rezhim("v5", "wf", 1, False, None)
    c = ps.load_kub("S1", rz)
    with pytest.raises(pv.ZamokVladeltsa):
        ps.proverit_v5(c, rz)


def test_g5_mutaciya_pervyj_kvartal_vypal_lovitsya():
    d, x = ryad("2014-06-02", "2026-02-27", 3)
    d, x = d[d >= np.datetime64("2015-01-01")], x[d >= np.datetime64("2015-01-01")]

    def proverka(kv):
        assert len(kv) == 45 and next(iter(kv)) == "2015Q1"
    proverka(ev.kvartaly_perioda("wf")(d, x))
    mut = {k: v for k, v in list(ev.kvartaly_perioda("wf")(d, x).items())[1:]}      # подложенная ошибка
    with pytest.raises(AssertionError):
        proverka(mut)
    m = (d >= np.datetime64("2016-01-01")) & (d <= np.datetime64("2019-12-31"))
    with pytest.raises(AssertionError):                                              # период без 2015 (train4 вместо v2/wf)
        proverka(ev.kvartaly_perioda("train4")(d[m], x[m]))
