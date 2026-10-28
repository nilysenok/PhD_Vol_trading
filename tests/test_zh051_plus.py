"""К0-х Ж-051: PlusEvaluator на синтетике (без данных). Golden «оси выключены = родитель», сторона входа, слоты, контроль
рангов, ленивый импорт К3/К4. Мутации: стороны перепутаны, σ из завтра, баллы без перемешивания."""
import types

import numpy as np
import pandas as pd
import pytest

from fastbt.e1_portfel import ball_sily
from fastbt.e1_portfel import ball_sily
from fastbt import etap1_ansambl_eval as AE, etap1_eval as EE, etap1_plus_eval as PE
from fastbt.etap1_eval import YAKOR

TICKS = ["AAA", "BBB", "CCC"]
DATES = pd.bdate_range("2018-01-01", "2020-01-10").to_numpy("datetime64[ns]")
N = len(DATES)


class FakeSk:
    tickers, sig_names = TICKS, ("a",)


class FakeInp:
    def __init__(self):
        rng = np.random.default_rng(1)
        self.meta = {t: dict(s0=0, dates=DATES, i1=N, a18=0) for t in TICKS}
        self.bars = {t: [None] * N for t in TICKS}
        self.close_n6 = {t: 100 + rng.normal(size=N).cumsum() for t in TICKS}
        self.sigma = {"rv22": {t: 0.01 + 0.001 * np.arange(N) / N + 0.002 * i for i, t in enumerate(TICKS)}}
        self.sigma["ewma094"] = self.sigma["rv22"]
        self.tick = {t: 0.01 for t in TICKS}
        self.sobytiya = {"net": {t: np.zeros(N, bool) for t in TICKS}}
        self.rezhim = {}
        self.div = {t: np.zeros(N) for t in TICKS}      # E1Inputs.div (#370): {тикер: D на день гэпа}, без событий


POS = {t: np.where(np.arange(N) % 20 < 6, 1.0 if i != 1 else -1.0, 0.0) for i, t in enumerate(TICKS)}
STR = {t: np.linspace(1, 2, N) * (i + 1) for i, t in enumerate(TICKS)}


class Fake:
    """Подменяет то, что требует данных: позиции, силу сигнала, окно b."""
    def __init__(self, cls_init=None):
        self.sid, self.inp, self.base, self.comm = "S2", FakeInp(), {"a": 1}, 0.0004
        self.sk, self._sigs, self._pos, self._eng = FakeSk(), {}, {}, {}

    def _positions(self, t, sp, q, anchor, gi1):
        return POS[t].astype(np.int8)

    def _strength(self, t, sp, q, s0, b):
        return STR[t][s0:s0 + b + 1]

    def _ctx(self):
        return getattr(self, "ctx", None)

    def _ens_signal(self, t, pts, mode, dates, s0, b):
        return POS[t][:b + 1] * 1.0, np.full(b + 1, 0.5)


class FakePlus(Fake, PE.PlusEvaluator):
    pass


class FakeParent(Fake, AE.AnsamblEvaluator):
    pass


PT = {**YAKOR, "tau": 5, "k_s": 2.0, "R": 2.0, "a": 1, "proverka": "bars5m", "vhod": 0.5, "sigma": "ewma094"}


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def run(t, days, sig, sigma, p, close=None, w=None, win=None, div=None):
        calls.append(dict(t=t, sig=np.array(sig), w=np.array(w), p=p, win=win, div=div))
        return types.SimpleNamespace(g=None)
    monkeypatch.setattr(PE, "run_e1", run)
    monkeypatch.setattr(EE, "run_e1", run)
    monkeypatch.setattr(AE, "run_e1", run)
    return calls


def _rows(cls, pt, ens=None):
    e = cls()
    e.ens = ens
    return e._rows(e.full(pt), "train", False)


def test_golden_vykl_ravno_roditelyu_i_moduli_ne_zvanye(spy, monkeypatch):
    monkeypatch.setattr(PE, "modul", lambda n: (_ for _ in ()).throw(AssertionError(f"вызван {n} при выключенных осях")))
    _rows(FakeParent, PT)
    a, spy[:] = list(spy), []
    _rows(FakePlus, PT)                                   # оси новых модулей не заданы — умолчания PLUS_NET
    assert len(a) == len(spy) == 3
    for x, y in zip(a, spy):
        assert x["sig"].tobytes() == y["sig"].tobytes() and x["w"].tobytes() == y["w"].tobytes() and x["p"] == y["p"]


def test_side_gate_otvet_izvesten_i_mutatsiya_storony():
    d = np.array([0, 1, 1, 0, -1, -1, 0, 1.0])
    lg = np.array([1, 0, 1, 1, 1, 1, 1, 1], bool)         # лонг запрещён в день 1 (старт первой сделки)
    sh = np.ones(8, bool)
    got = PE.side_gate(d, lg, sh)
    assert got.tolist() == [0, 0, 0, 0, -1, -1, 0, 1]
    assert PE.side_gate(d, sh, lg).tolist() != got.tolist()           # перепутанные стороны ловятся
    flip = np.array([1, -1, -1, 0, 0, 0, 0, 0.0])                       # смена знака без нуля: две сделки
    assert PE.side_gate(flip, np.ones(8, bool), np.zeros(8, bool)).tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


def test_sektor_filtruet_vkhod_po_storone(spy, monkeypatch):
    def allow(pt, t, dates, ctx):
        n = len(dates)
        return np.zeros(n, bool), np.ones(n, bool)                 # лонг везде запрещён, шорт разрешён
    monkeypatch.setattr(PE, "modul", lambda n: types.SimpleNamespace(allow=allow))
    _rows(FakePlus, {**PT, "sektor_trend": "po_trendu"})
    by = {c["t"]: c["sig"] for c in spy}
    assert not by["AAA"].any() and not by["CCC"].any()             # лонговые тикеры обнулены
    assert (by["BBB"] <= 0).all() and by["BBB"].any()              # шорт сохранён


def test_top_k_vyzyvaetsya_na_vseh_tickerah_ball_sovpadaet_s_formuloy(spy, monkeypatch):
    seen = {}

    def rasp(sig, ball, k):
        seen.update(sig=sig, ball=ball, k=k)
        out = sig.copy()
        out[:, 1] = 0.0                                            # BBB (колонка 1) вытеснен
        return out
    monkeypatch.setattr(PE, "modul", lambda n: types.SimpleNamespace(raspredelit=rasp, ball_sily=ball_sily))
    _rows(FakePlus, {**PT, "top_k": 5})
    assert seen["sig"].shape[1] == 3 and seen["k"] == 5 and seen["ball"].shape == seen["sig"].shape
    b = int(np.searchsorted(DATES, np.datetime64("2020-01-01")))
    sg = FakeInp().sigma["ewma094"]["AAA"][:b + 1]
    assert np.allclose(seen["ball"][:b + 1, 0], np.abs(STR["AAA"][:b + 1]) / sg)           # σ того же дня d
    assert not np.allclose(seen["ball"][:b, 0], np.abs(STR["AAA"][:b]) / sg[1:])           # σ из завтра — другое число
    assert not {c["t"]: c["sig"] for c in spy}["BBB"].any()        # результат распределителя дошёл до движка


def test_kontrol_rang_peremeshivaet_mezhdu_tikerami_po_dnyam():
    ball = {t: np.arange(10.0) + 100 * i for i, t in enumerate(TICKS)}
    k = PE.PlusEvaluator._kontrol_rang({"kontrol": ("sluchaynyy_rang", 7)}, ball)
    for d in range(10):                                            # набор значений дня сохранён
        assert sorted(k[t][d] for t in TICKS) == sorted(ball[t][d] for t in TICKS)
    assert any(not np.array_equal(k[t], ball[t]) for t in TICKS)    # мутация «не перемешано» ловится
    k2 = PE.PlusEvaluator._kontrol_rang({"kontrol": ("sluchaynyy_rang", 7)}, ball)
    assert all(np.array_equal(k[t], k2[t]) for t in TICKS)          # seed воспроизводим
    assert PE.PlusEvaluator._kontrol_rang({}, ball) is ball


def test_ansambl_ves_umnozhen_na_velichinu(spy, monkeypatch):
    monkeypatch.setattr(PE, "modul", lambda n: types.SimpleNamespace(allow=lambda pt, t, d, c: (np.ones(len(d), bool),) * 2))
    _rows(FakePlus, {**PT, "shirina": "po_trendu"}, ens=([PT], "srednee"))
    assert all(np.allclose(c["w"], 0.5) for c in spy)


def test_modul_ne_slit_dayot_ponyatnuyu_oshibku(monkeypatch):
    monkeypatch.setattr(PE.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError(n)))
    with pytest.raises(NotImplementedError, match="e1_sektor"):
        PE.modul("e1_sektor")


def test_plus_on_i_full_ne_menyayut_roditelskie_klyuchi():
    e = FakePlus()
    f = e.full({"top_k": 3})
    assert PE.plus_on(f) and not PE.plus_on(e.full({})) and f["ma_n"] == 200 and f["sigma"] == "rv22"
