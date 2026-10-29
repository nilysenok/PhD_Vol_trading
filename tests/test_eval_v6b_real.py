"""V6-12b, реальные данные train4 (skip без дневного ряда 18:50): «выкл.» = v5 побайтно на 20 точках каждого Sk; каждая новая ось
меняет результат (мягкая точка, условия оси); тень паузы = excess тикера ledger; вход в день паузы запрещён; мутация «пауза не доходит»."""
import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, eval_v5, eval_v6_osi as osi6
from fastbt.eval_v5_osi import yakor
from fastbt.eval_v6 import EvaluatorV6

SIDS, B_OSI = canon_v5.SIDS, osi6.DVIG + osi6.PAUZA
real = pytest.mark.skipif(not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(), reason="нет дневного ряда 18:50")
_KESH = {}
SP = {"S1": dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5),      # мягкие сигналы: на 3 тикерах есть сделки
      "S2": dict(ma_tip="sma", bb_window=20, bb_std=0.75, podtv="none"), "S3": dict(okno=5, podtv="none"),
      "S4": dict(period=5, mnozh=0.5), "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
      "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net")}
DOP = {"pauza_porog": {"pauza_okno": 42}, "time_decay": {"tau": 10, "tejk": "pct:0.05"}, "ed_lim": {"k_e": 1.0}, "zhizn": {"k_e": 1.0}}
TD_TREND = {"tau": 10, "tejk": "rr:1.0"}                                                    # у тренда тейк — rr (pct — значение V6-11)
TREND = {"trail": "atr:1.0"}          # без трейла стоп-заявочный тренд v5 в окне не торгует: excess не зависит от stop_nach (baseline v5)
TD_S5 = {"tau": 10, "tejk": "pct:0.5", "smena": "net"}                                      # у S5 иначе позиции закрываются раньше tau//2


def _para(sid):
    if sid not in _KESH:
        from fastbt import etap1
        inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
        inp.sk.tickers = list(inp.sk.tickers)[:3]
        _KESH[sid] = (eval_v5.EvaluatorV5(sid, inp, base), EvaluatorV6(sid, inp, base))
    return _KESH[sid]


def _vykl(sid):
    nov = canon_v6.novye(canon_v6._kub(sid))
    return {n: nov[n]["setka"][0] for n in B_OSI if n in nov}


@real
@pytest.mark.parametrize("sid", SIDS)
def test_vykl_12b_ravno_v5_pobaytno(sid):
    e5, e6 = _para(sid)
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 60)
    for _, p in pts[:20]:
        r5, x5, t5 = e5.evaluate_v2(p, "train4")
        r6, x6, t6 = e6.evaluate_v2({**p, **_vykl(sid)}, "train4")
        r5.pop("t_sek", None), r6.pop("t_sek", None)
        b = lambda x: None if x is None else x.tobytes()      # noqa: E731
        assert repr(r5) == repr(r6) and b(x5) == b(x6) and b(t5) == b(t6)


def _meniaet(sid, ax):
    """(menyaet, ne_schitana): хоть одно значение оси меняет excess точки (мягкий сигнал, нужные условия оси); иначе — причины."""
    _, e6 = _para(sid)
    trend = sid in ("S3", "S4")
    dop = TD_S5 if (sid, ax) == ("S5", "time_decay") else TD_TREND if ax == "time_decay" and trend else DOP.get(ax, {})
    p, prich = {**yakor(sid), **SP[sid], **(TREND if trend else {}), **dop}, set()
    _, x0, _ = e6.evaluate_v2(p, "train4")
    assert x0 is not None and np.abs(x0).sum() > 0, f"{sid}: точка без сделок — тест ничего не доказывает"
    for v in canon_v6.novye(canon_v6._kub(sid))[ax]["setka"][1:]:
        r, x, _ = e6.evaluate_v2({**p, ax: v}, "train4")
        if x is None:
            prich.add(r["prichina"])
        elif not np.array_equal(x, x0):
            return True, prich
    return False, prich


@real
@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("ax", ["time_decay", "sigma_ist", "ed_lim", "zhizn", "pauza_porog"])
def test_kazhdaya_os_menyaet_rezultat(sid, ax):
    if ax not in canon_v6.novye(canon_v6._kub(sid)):
        pytest.skip("оси нет в кубе этого Sk")
    m, prich = _meniaet(sid, ax)
    assert m, f"{sid}.{ax}: не меняет результат; причины ne_schitana: {prich}"


def _b0(sid="S1"):
    return {**yakor(sid), **SP[sid], "k_e": 1.0}


@real
def test_time_decay_bez_tau_ne_schitana():
    _, e6 = _para("S1")
    p = {**yakor("S1"), **SP["S1"]}
    r, x, _ = e6.evaluate_v2({**p, "time_decay": "da"}, "train4")
    assert r["status"] == "ne_schitana" and "time_decay" in r["prichina"] and x is None
    r, x, _ = e6.evaluate_v2({**p, "time_decay": "da", "tau": 10, "tejk": "pct:0.05"}, "train4")
    assert r["status"] == "ok"


@real
def test_ten_ravna_excess_tikera_ledger():
    _, e6 = _para("S1")
    e6.evaluate_v2(_b0(), "train4")
    rows = [(ti, r, 0, 0) for ti, r in enumerate(e6._last[4])]
    ten = osi6.ten(e6, rows, "train4", e6._zctx[1])
    ser = e6._series("train4", e6.comm)
    tik, dts = ser[1], ser[2]
    assert len(ten) == tik.shape[0] == 3
    for ti, t in enumerate(e6.sk.tickers):
        s = ten[t].groupby(level=0).sum().reindex(dts)
        assert np.allclose(s.to_numpy(), tik[ti], equal_nan=True, atol=1e-12) and np.isfinite(tik[ti]).any()


@real
def test_pauza_zapreshchaet_vhody_v_dni_pauzy(monkeypatch):
    _, e6 = _para("S1")
    r0, x0, _ = e6.evaluate_v2(_b0(), "train4")
    r, x, _ = e6.evaluate_v2({**_b0(), "pauza_porog": 0.5, "pauza_okno": 42}, "train4")
    blok = 0
    for ti, t in enumerate(e6.sk.tickers):
        b = e6._b(t, "train4")
        dts = e6.inp.meta[t]["dates"][:b + 1]
        al = osi6.razreshit(e6.v6, t, dts, e6._ten)
        blok += int((~al).sum())
        assert all(al[tr.d_sig] for tr in e6._last[4][ti].trades), t               # вход в день паузы — нет
    assert blok > 50 and not np.array_equal(x, x0)
    monkeypatch.setattr(EvaluatorV6, "_allow", eval_v5.EvaluatorV5._allow)          # мутация «пауза не доходит до входа»
    _, xm, _ = e6.evaluate_v2({**_b0(), "pauza_porog": 0.5, "pauza_okno": 42}, "train4")
    assert np.array_equal(xm, x0)
