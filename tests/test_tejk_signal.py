"""Л-13: ось tejk_signal (Ж-084), модуль fastbt/tejk_signal.py. Данные <= 2019: FASTBT_BARS1M=…/data/regen/D25/bars1m.
python3 -m pytest tests/test_tejk_signal.py -q --tb=short
Мутации: знак шорта, M «за» ценой, M с d+1, порядок стоп/тейк — в tests/proverka_progon/test_p6_tejk.py; здесь — маршрут, ряды отчёта,
счётчик, якорь, canon, реальная лента."""
import numpy as np
import pytest

from fastbt import bars5m, cache, fast_etap1
from fastbt import tejk_signal as ts
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day, Trade

_MAIN_BARS = cache.MAIN / "data" / "regen" / "D25" / "bars1m"
if not bars5m.SRC.exists() and _MAIN_BARS.exists():
    bars5m.SRC = _MAIN_BARS


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


DAYS = [D([(100, 100, 100, 100)]), D([(100, 105, 99, 100)]), D([(100, 110, 100, 100)]), D([(100, 100, 100, 100)])]


def _run(f, M, tejk="k_srednemu", **kw):
    n = len(DAYS)
    return f("T", DAYS, np.ones(n), np.full(n, 0.02), Params(tick=0.01, check="bary", comm=0.0, tejk=tejk, **kw),
             tejk_M=None if M is None else np.full(n, M))


@pytest.mark.parametrize("M,take", [(104.0, True), (98.0, False), (float("nan"), False)])
def test_n_bez_tejka_i_marshrut(M, take):
    a, b = _run(run, M), _run(fast_etap1.run, M)                     # fast.run при tejk ≠ net — медленный путь, тот же Result
    assert a.trades == b.trades and a.n_bez_tejka == b.n_bez_tejka == (0 if take else 1)
    assert (a.trades[0].reason == "take") is take


def test_tejk_trebuet_M_i_nesovmestim_s_R_a():
    with pytest.raises(ValueError):
        _run(run, None)
    with pytest.raises(ValueError):
        _run(run, 104.0, a=1.0)
    with pytest.raises(ValueError):
        _run(run, 104.0, b=1.0, R=2.0, tau=3)


def test_pokryto_ne_pokryvaet_tejk():
    assert fast_etap1.pokryto({"tejk_signal": "net"}) and fast_etap1.pokryto({})
    assert not fast_etap1.pokryto({"tejk_signal": "pol_puti"}) and not fast_etap1.pokryto({"tejk_signal": "k_srednemu"})


def _tr(entry, exit_, reason, side=1, day=0, fee=0.0):
    return Trade("T", 0, 0, 0, entry, day, 0, exit_, reason, side, fee, fee)


def test_otchet_ryady():
    dates = np.array(["2016-03-01", "2017-03-01"], "datetime64[D]")
    sd = [_tr(100, 110, "take", day=0), _tr(100, 95, "stop", day=0), _tr(100, 90, "stop", side=-1, day=0)]   # шорт 100→90 — прибыль
    o = ts.otchet([(sd, dates)], [2016, 2017])
    assert o["dolya_tejk_y"]["2016"] == pytest.approx(1 / 3) and o["dolya_pribyl_y"]["2016"] == pytest.approx(2 / 3)
    assert o["vyigrysh_sredn_y"]["2016"] == pytest.approx((np.log(1.1) + np.log(100 / 90)) / 2)
    assert o["proigrysh_sredn_y"]["2016"] == pytest.approx(np.log(0.95))
    assert o["top5_pnl_y"]["2016"] == pytest.approx(np.log(100 / 90) / (np.log(1.1) + np.log(0.95) + np.log(100 / 90)))   # k = 1 из 3
    assert all(np.isnan(v["2017"]) for v in o.values())                                       # год без сделок — nan, не 0
    assert list(o["top5_pnl_y"]) == ["2016", "2017"]


def test_otchet_top5_nan_pri_otricatelnoy_summe_i_otkrytye_ne_idut():
    dates = np.array(["2016-03-01"], "datetime64[D]")
    o = ts.otchet([([_tr(100, 90, "stop"), _tr(100, 99, None)], dates)], [2016])               # вторая сделка открыта
    assert np.isnan(o["top5_pnl_y"]["2016"]) and o["dolya_pribyl_y"]["2016"] == 0.0


def test_komissiya_v_dohodnosti():
    assert ts.dohodnost(_tr(100, 100, "stop", fee=0.001)) == pytest.approx(-0.002)


def test_uroven_M_sma_i_net_osi_u_S3():
    dn = {"close": np.arange(1.0, 21.0)[None], "high": np.arange(2.0, 22.0)[None], "low": np.arange(0.0, 20.0)[None],
          "volume": np.ones((1, 20)), "dates": np.datetime64("2018-01-01") + np.arange(20)}
    assert ts.uroven_M("S1", {"ma_window": 5}, dn)[0, 9] == pytest.approx(8.0)               # mean(5..9)... close 6..10 = 8
    for s in ("S3", "S4"):
        with pytest.raises(ValueError):
            ts.uroven_M(s, {}, dn)


def test_canon_tejk_i_kluch():
    from fastbt.ledger_v2 import canon_v2
    q = canon_v2("S6", {"k_s": 2.0, "R": 3.0, "k_x": 1.0, "tejk_signal": "pol_puti"})
    assert q["R"] == "net" and q["k_x"] == "net" and q["k_s"] == 2.0


def test_yakor_snimaetsya_lyuboy_novoy_osyu():
    from fastbt.etap1_eval import YAKOR
    from fastbt.etap1_eval_v2 import NOVYE, EvaluatorV2
    got = {}
    ev = object.__new__(EvaluatorV2)
    ev.full = lambda p: {**YAKOR, **p}
    ev._rows = lambda p, per, anchor: (_ for _ in ()).throw(KeyError(anchor))
    for k, v in (("tejk_signal", "pol_puti"), ("k_x", 1.0), ("t_vyhod", "open2"), (None, None)):
        ev._ctx = dict(nov={**NOVYE, **({k: v} if k else {})}, vozm={}, default=False)
        with pytest.raises(KeyError) as e:
            ev.excess({}, "train4")
        got[k] = e.value.args[0]
    assert got == {"tejk_signal": False, "k_x": False, "t_vyhod": False, None: True}


@pytest.fixture(scope="module")
def ev():
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    return EvaluatorV2("S1", etap1.build_inputs("S1", period="train4"), etap1.base_point("S1"), 0.0004)


def test_stroka_ledger_polya_na_realnoy_lente(ev):
    pt = dict(vhod=0.5, proverka="bars5m")
    r0 = ev.evaluate_v2(pt, "train4")[0]
    r1 = ev.evaluate_v2({**pt, "tejk_signal": "k_srednemu"}, "train4")[0]
    for r in (r0, r1):
        assert list(r["top5_pnl_y"]) == ["2016", "2017", "2018", "2019"] and "n_bez_tejka" in r
    assert r0["n_bez_tejka"] == 0 and all(v == 0 for v in r0["dolya_tejk_y"].values())
    assert any(v > 0 for v in r1["dolya_tejk_y"].values()) and r1["n_bez_tejka"] >= 0


def test_S3_s_osyu_ne_schitaetsya_molcha():
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    e3 = EvaluatorV2("S3", etap1.build_inputs("S3", period="train4"), etap1.base_point("S3"), 0.0004)
    with pytest.raises(ValueError):
        e3.evaluate_v2(dict(vhod=0.5, proverka="bars5m", tejk_signal="pol_puti"), "train4")
