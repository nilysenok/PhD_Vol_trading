"""Д-7 (Ж-145): `vozm={"sig_fn": f}` у EvaluatorV5/V6 — подмена сигнала до слоя исполнения (к2, плацебо); прочие ключи — ValueError.

Путь сигнала v5/v6: `_rows` → `_start` → `_sig5` (свой, не `_sig2` EvaluatorV2); sig_fn применяется сразу после `_sig5`,
до allow/ранга/размера — как `_sig2` у v4. Реальные данные train4 (≤ 2019), 3 тикера; нет ряда 18:50 — skip.
Мутации: «sig_fn игнорируется» (G7.4), «лаг 0 ≠ исходный» (№ 9, G7.1).
python3 -m pytest tests/test_eval_vozm_sig_fn.py -q
"""
import json

import numpy as np
import pytest

from fastbt import eval_v5, eval_v6
from fastbt import progon_kontrol as pk
from fastbt.eval_v5_osi import yakor

SP = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)


def _est() -> bool:
    from fastbt import daily_1850
    return daily_1850.out_path().exists()


real = pytest.mark.skipif(not _est(), reason="нет дневного ряда 18:50 в FASTBT_MAIN")
_EV: dict = {}


def _ev(cls):
    if cls not in _EV:
        from fastbt import etap1
        inp, base = etap1.build_inputs("S1", period="train4"), etap1.base_point("S1")
        inp.sk.tickers = list(inp.sk.tickers)[:3]
        _EV[cls] = (eval_v5.EvaluatorV5 if cls == "v5" else eval_v6.EvaluatorV6)("S1", inp, base)
    return _EV[cls]


def _pt():
    return {**yakor("S1"), **SP}


def _poln(ev, vozm=None):
    r, x, t = ev.evaluate_v2(_pt(), "train4", vozm)
    r = {k: v for k, v in r.items() if k not in ("t_sek", "vozm")}
    return repr(r), x.tobytes(), t.tobytes()


def _sdelok(ev, vozm):
    return sum(ev.evaluate_v2(_pt(), "train4", vozm)[0]["sdelok_y"].values())


@real
@pytest.mark.parametrize("cls", ["v5", "v6"])
def test_g71_tozhdestvo_i_sdvig_nol_pobaytno(cls):
    ev, base = _ev(cls), _poln(_ev(cls))
    assert _poln(ev, {"sig_fn": lambda t, s, d: s}) == base
    assert _poln(ev, {"sig_fn": pk.k2_sig_fn(0)}) == base


@real
@pytest.mark.parametrize("cls", ["v5", "v6"])
def test_g72_placebo_sohranyaet_chislo_i_dliny_otrezkov(cls):
    ev, seen = _ev(cls), []
    fn = pk.placebo_sig_fn(1)

    def spy(t, s, d):
        out = fn(t, s, d)
        seen.append((np.asarray(s, float).copy(), out))
        return out
    ev.evaluate_v2(_pt(), "train4", {"sig_fn": spy})
    assert seen and any(pk.runs_of(a) for a, _ in seen)
    for a, b in seen:
        ra, rb = pk.runs_of(a), pk.runs_of(b)
        assert sorted((ln, v) for _, ln, v in ra) == sorted((ln, v) for _, ln, v in rb)
    assert any(not np.array_equal(a, b) for a, b in seen)           # сигнал действительно перемешан


@real
@pytest.mark.parametrize("cls", ["v5", "v6"])
def test_sig_fn_dohodit_do_ispolneniya(cls):
    ev, base = _ev(cls), _poln(_ev(cls))
    assert _sdelok(ev, {"sig_fn": lambda t, s, d: np.zeros(len(s))}) == 0
    assert _poln(ev, {"sig_fn": pk.k2_sig_fn(100)}) != base
    assert _poln(ev) == base                                      # возмущение не залипает на оценщике


@real
@pytest.mark.parametrize("cls", ["v5", "v6"])
@pytest.mark.parametrize("vozm", [{"komissiya": 0.0005}, {"sigma_mult": 2.0}, {"sig_fn": lambda t, s, d: s, "komissiya": 0.0}, {"sig_fn": 1}])
def test_g73_prochie_klyuchi_valueerror(cls, vozm):
    with pytest.raises(ValueError, match="vozm"):
        _ev(cls).evaluate_v2(_pt(), "train4", vozm)


@real
@pytest.mark.parametrize("cls,mod", [("v5", [eval_v5]), ("v6", [eval_v5, eval_v6])])
def test_g74_mutaciya_sig_fn_ignoriruetsya_lovitsya(cls, mod, monkeypatch):
    zero = {"sig_fn": lambda t, s, d: np.zeros(len(s))}
    ev = _ev(cls)
    assert _sdelok(ev, zero) == 0
    for m in mod:
        monkeypatch.setattr(m, "proverit_vozm", lambda v, s: {})   # подложено: sig_fn отброшен до оценщика
    assert _sdelok(ev, zero) > 0                                   # тест G7.x «сделок 0» на мутации падает


@real
def test_g76_sig_fn_ne_popadaet_v_boevoy_ledger(tmp_path):
    r, _, _ = _ev("v5").evaluate_v2(_pt(), "train4", {"sig_fn": lambda t, s, d: s})
    assert r["vozm"] is not None                                   # строка помечена: не отличима от боевой быть не может
    with pytest.raises(TypeError):
        json.dumps(r)                                              # и Pisatel.close (json.dumps) её не запишет
    assert _ev("v5").evaluate_v2(_pt(), "train4")[0]["vozm"] is None
    with pytest.raises(ValueError, match="kontrol"):               # контрольные строки — только в kontrol/
        pk.kontrol_pisatel(tmp_path / "ledger", "S1", 1, dict(T=20, kod="x", vhody="y"))
    assert not (tmp_path / "ledger").exists()
