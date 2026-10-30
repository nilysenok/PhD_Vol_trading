"""Итерация 26 Ж-051 «только лонг» (К-26) на синтетике, без данных. Мутации: шорт не отсекается; лонг отсекается вместе с шортом;
плата за шорт берётся и с лонга. Правило выключено = старый путь побайтно."""
import types

import numpy as np
import pytest

from fastbt import etap1_ansambl_eval as AE, etap1_eval as EE, etap1_plus_eval as PE
from test_zh051_plus import FakeInp, FakeParent, FakePlus, N, PT, TICKS, _rows


class InpDiv(FakeInp):
    def __init__(self):
        super().__init__()
        self.div = {t: np.zeros(N) for t in TICKS}


@pytest.fixture(autouse=True)
def _div_i_spy(monkeypatch):
    monkeypatch.setattr("test_zh051_plus.FakeInp", InpDiv)       # Fake() берёт FakeInp из своего модуля; у него нет div (после #370)


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def run(t, days, sig, sigma, p, close=None, w=None, win=None, div=None, dates=None):
        assert dates is not None and len(dates) == len(sig)          # Д5-2: даты строк передаются, как в exec_e.run
        calls.append(dict(t=t, sig=np.array(sig), w=np.array(w), p=p, win=win))
        return types.SimpleNamespace(g=np.zeros(win[1] - win[0]), trades=[])
    for m in (PE, EE, AE):
        monkeypatch.setattr(m, "run_e1", run)
    return calls


D = np.array([0, 1, 1, 0, -1, -1, 0, 1, -1, -1.0])


def _verdict_only_long(fn) -> None:
    """Известный ответ: лонги целы, шорты обнулены."""
    assert fn(D).tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 0, 0]


def test_only_long_otvet_izvesten():
    _verdict_only_long(PE.only_long)
    flip = np.array([1, -1, -1, 0.0])                                 # смена знака без нуля: лонг остался, шорт ушёл
    assert PE.only_long(flip).tolist() == [1, 0, 0, 0]


def test_mutatsiya_short_ne_otsekaetsya_lovitsya():
    with pytest.raises(AssertionError):
        _verdict_only_long(lambda d: PE.side_gate(d, np.ones(len(d), bool), np.ones(len(d), bool)))


def test_mutatsiya_long_otsekaetsya_vmeste_s_shortom_lovitsya():
    with pytest.raises(AssertionError):
        _verdict_only_long(lambda d: PE.side_gate(d, np.zeros(len(d), bool), np.zeros(len(d), bool)))
    with pytest.raises(AssertionError):                                # стороны перепутаны: остаётся шорт
        _verdict_only_long(lambda d: PE.side_gate(d, np.zeros(len(d), bool), np.ones(len(d), bool)))


class Tr(types.SimpleNamespace):
    pass


def _fee_case(charge):
    days = [types.SimpleNamespace(pusto=(i == 3)) for i in range(10)]
    trades = [Tr(side=-1, row0=1, exit_day=5, d_sig=0), Tr(side=1, row0=1, exit_day=5, d_sig=0)]
    g = np.zeros(8)                                                    # окно [1; 9)
    charge(g, trades, days, np.array([2.0] + [0.0] * 9), (1, 9), 0.126)
    return g


def test_plata_za_short_tolko_s_shorta_vo_vse_torgovye_stroki():
    g = _fee_case(PE.charge_short)
    one = 2.0 * 0.126 / 252
    assert g.tolist() == pytest.approx([-one, -one, 0.0, -one, 0, 0, 0, 0])      # строки 1, 2, 4 (3 — пустой день); лонг не платит


def test_mutatsiya_plata_beretsya_i_s_longa_lovitsya():
    def bad(g, trades, days, w, win, plata):
        PE.charge_short(g, [Tr(**{**vars(t), "side": -1}) for t in trades], days, w, win, plata)
    assert not np.allclose(_fee_case(bad), _fee_case(PE.charge_short))


def test_plata_nol_ili_pustaya_lenta_ne_menyaet_g():
    g = np.zeros(8)
    PE.charge_short(g, [], [], np.ones(3), (1, 9), 0.1)
    assert not g.any()
    assert not PE.plus_on(PT) and PE.plus_on({**PT, "storona": "long"}) and PE.plus_on({**PT, "plata_shorta": 0.05})


def test_pravilo_vykluchheno_ravno_staromu_puti_pobaitno(spy, monkeypatch):
    monkeypatch.setattr(PE, "modul", lambda n: (_ for _ in ()).throw(AssertionError("модуль не нужен")))
    _rows(FakeParent, PT)
    ref, spy[:] = list(spy), []
    _rows(FakePlus, {**PT, "storona": "obe", "plata_shorta": 0.0})
    assert len(ref) == len(spy) == 3
    assert all(a["sig"].tobytes() == b["sig"].tobytes() and a["w"].tobytes() == b["w"].tobytes() and a["p"] == b["p"]
               for a, b in zip(ref, spy))


def test_tolko_long_cherez_evaluator_bez_shortov_longi_celi(spy):
    from test_zh051_plus import POS
    _rows(FakePlus, {**PT, "storona": "long"})
    by = {c["t"]: c["sig"] for c in spy}
    assert not by["BBB"].any()                                         # BBB — только шорты
    for t in ("AAA", "CCC"):
        n = len(by[t]) - 1                                             # последний день окна обнуляет сам оценщик
        assert (by[t] >= 0).all() and by[t][:n].tobytes() == np.maximum(POS[t][:n], 0).astype(float).tobytes()
