"""V6-12d, реальные данные train4 (≤ 2021; FASTBT_V6_REAL=1, 3 тикера, ≤ 2 процессов): «выкл.» ВСЕХ новых осей = v5 побайтно на 20 точках каждого Sk;
каждая новая подключённая ось (пивот S5, VWAP S6, pol_puti, piv_buf, оси тренда на стоп-заявке) меняет результат; мутация «пивот со строки d+1» ловится."""
import os

import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, eval_v5, eval_v6_piv as piv6
from fastbt.eval_v5_osi import yakor
from fastbt.eval_v6 import EvaluatorV6
from fastbt.exec_e_piv import PivUrovni

SIDS = canon_v5.SIDS
real = pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1" or not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(),
                          reason="тяжёлый реальный тест v6 (хук ≤ 300 с): FASTBT_V6_REAL=1 и дневной ряд 18:50")
_KESH = {}
SP = {"S1": dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5),
      "S2": dict(ma_tip="sma", bb_window=20, bb_std=0.75, podtv="none"), "S3": dict(okno=5, podtv="none"),
      "S4": dict(period=5, mnozh=0.5), "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
      "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net")}
TR = ("S3", "S4")
# (Sk, ось точки, значения): пивот, VWAP, pol_puti, piv_buf (при стопе piv:next), оси тренда на стоп-заявке, значения V6-11
OSI = ([("S5", "stop_nach", ["piv:next", "piv:half"]), ("S5", "tejk", ["piv:R", "piv:half", "pol_puti"]), ("S6", "stop_nach", ["vwap:1.5", "vwap:3"]),
        ("S1", "tejk", ["pol_puti"]), ("S2", "tejk", ["pol_puti"]), ("S6", "tejk", ["pol_puti"]), ("S5", "piv_buf", [0.2])]
       + [(s, a, v) for s in TR for a, v in (("bezubytok", ["atr:1.0", "rang:1"]), ("stop_v_plyus", ["2:1"]), ("pir", [0.5]), ("reentry", ["da"]),
                                             ("stop_nach", ["pct:0.01", "atr:2.0"]), ("tejk", ["pct:0.02", "atr:3.0"]), ("trail", ["step:0.5"]))])


def _para(sid):
    if sid not in _KESH:
        from fastbt import etap1
        inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
        inp.sk.tickers = list(inp.sk.tickers)[:3]
        _KESH[sid] = (eval_v5.EvaluatorV5(sid, inp, base), EvaluatorV6(sid, inp, base))
    return _KESH[sid]


def _p(sid):
    return {**yakor(sid), **SP[sid], **({"trail": "atr:1.0"} if sid in TR else {"k_e": 1.0})}


def _vykl(sid):
    return {n: v["setka"][0] for n, v in canon_v6.novye(canon_v6._kub(sid)).items()}


@real
@pytest.mark.parametrize("sid", SIDS)
def test_vykl_vseh_novyh_osej_ravno_v5_pobaytno(sid):
    e5, e6 = _para(sid)
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 60)
    for _, p in pts[:20]:
        r5, x5, t5 = e5.evaluate_v2(p, "train4")
        r6, x6, t6 = e6.evaluate_v2({**p, **_vykl(sid)}, "train4")
        r5.pop("t_sek", None), r6.pop("t_sek", None)
        b = lambda x: None if x is None else x.tobytes()      # noqa: E731
        assert repr(r5) == repr(r6) and b(x5) == b(x6) and b(t5) == b(t6)


def _baza(sid, ax):
    p = {**_p(sid), **_vykl(sid)}
    return {**p, "stop_nach": "piv:next"} if ax == "piv_buf" else p


@real
@pytest.mark.parametrize("sid,ax,vals", OSI)
def test_kazhdaya_novaya_os_menyaet_rezultat(sid, ax, vals):
    _, e6 = _para(sid)
    prich = set()
    for ke in ((None,) if sid in TR else (1.0, 0.5)):
        p = {**_baza(sid, ax), **({} if ke is None else {"k_e": ke})}
        r0, x0, _ = e6.evaluate_v2(p, "train4")
        assert x0 is not None and np.abs(x0).sum() > 0, f"{sid}.{ax}: база без сделок ({r0.get('prichina')})"
        for v in vals:
            r, x, _ = e6.evaluate_v2({**p, ax: v}, "train4")
            if x is None:
                prich.add(r["prichina"])
            elif not np.array_equal(x, x0):
                break
        else:
            continue
        return
    pytest.fail(f"{sid}.{ax}: не меняет результат; ne_schitana: {prich}")


@real
def test_mutatsiya_pivot_s_d_plyus_1_menyaet_rezultat(monkeypatch):
    _, e6 = _para("S5")
    p = {**_p("S5"), **_vykl("S5"), "stop_nach": "piv:next"}
    _, x, _ = e6.evaluate_v2(p, "train4")

    def sdvig(evr, ext, t, s0, b):
        a = piv6._extra(evr, t)
        return PivUrovni(**{k: a[k][s0 + 1:s0 + b + 2] for k in ("P", "S1", "S2", "R1", "R2")})
    monkeypatch.setattr(piv6, "tm", sdvig)
    _, xm, _ = e6.evaluate_v2(p, "train4")
    assert not np.array_equal(x, xm), "мутация «уровни со строки d+1» не меняет результат — тест её не поймает"
