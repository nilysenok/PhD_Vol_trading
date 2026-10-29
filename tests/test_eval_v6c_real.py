"""V6-12c, реальные данные train4 (≤ 2021; skip без дневного ряда 18:50; тяжёлые — SDF_TYAZHELYE=1, 3 тикера, ≤ 2 процессов):
«выкл.» новых осей исполнения = v5 побайтно на 20 точках каждого Sk; каждая ось меняет результат там, где есть в кубе;
мутация «лестница без деления веса на 2» и «ось исполнения не доходит до оценщика» ловятся."""
import os

import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, eval_v5, eval_v6_ispoln as isp
from fastbt.eval_v5_osi import yakor
from fastbt.eval_v6 import EvaluatorV6

SIDS = canon_v5.SIDS
real = pytest.mark.skipif(os.environ.get("FASTBT_V6_REAL") != "1" or not __import__("fastbt.daily_1850", fromlist=["x"]).out_path().exists(),
                          reason="тяжёлый реальный тест v6 (хук ≤ 300 с): FASTBT_V6_REAL=1 и дневной ряд 18:50")
tyazh = real
_KESH = {}
SP = {"S1": dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5),
      "S2": dict(ma_tip="sma", bb_window=20, bb_std=0.75, podtv="none"), "S3": dict(okno=5, podtv="none"),
      "S4": dict(period=5, mnozh=0.5), "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
      "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net")}
TREND = {"trail": "atr:1.0"}
# условия оси: vyhod_lim несовместим с t_vyhod ≠ close (якорь куба — open2), поэтому база с close; у тренда выход по времени (tau) — иначе t_vyhod нечему менять
DOP = {("vyhod_lim", False): {"t_vyhod": "close"}, ("t_vyhod", True): {"tau": 10}}


def _para(sid):
    if sid not in _KESH:
        from fastbt import etap1
        inp, base = etap1.build_inputs(sid, period="train4"), etap1.base_point(sid)
        inp.sk.tickers = list(inp.sk.tickers)[:3]
        _KESH[sid] = (eval_v5.EvaluatorV5(sid, inp, base), EvaluatorV6(sid, inp, base))
    return _KESH[sid]


def _vykl(sid):
    nov = canon_v6.novye(canon_v6._kub(sid))
    return {n: nov[n]["setka"][0] for n in isp.ISPOLN if n in nov}


def _p(sid):
    return {**yakor(sid), **SP[sid], **(TREND if sid in ("S3", "S4") else {"k_e": 1.0})}


@real
@tyazh
@pytest.mark.parametrize("sid", SIDS)
def test_vykl_12c_ravno_v5_pobaytno(sid):
    e5, e6 = _para(sid)
    pts, _ = canon_v5.tochki_v5(canon_v5.load_kub(sid), 0, 60)
    for _, p in pts[:20]:
        r5, x5, t5 = e5.evaluate_v2(p, "train4")
        r6, x6, t6 = e6.evaluate_v2({**p, **_vykl(sid)}, "train4")
        r5.pop("t_sek", None), r6.pop("t_sek", None)
        b = lambda x: None if x is None else x.tobytes()      # noqa: E731
        assert repr(r5) == repr(r6) and b(x5) == b(x6) and b(t5) == b(t6)


@real
@tyazh
@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("ax", isp.ISPOLN)
def test_kazhdaya_os_ispolneniya_menyaet_rezultat(sid, ax):
    if ax not in canon_v6.novye(canon_v6._kub(sid)):
        pytest.skip("оси нет в кубе этого Sk")
    _, e6 = _para(sid)
    prich = set()
    for ke in ((None,) if sid in ("S3", "S4") else (1.0, 0.5, 2.0)):         # глубина лимита: гэп через уровень бывает не при каждой
        p = {**_p(sid), **({} if ke is None else {"k_e": ke}), **DOP.get((ax, sid in ("S3", "S4")), {})}
        _, x0, _ = e6.evaluate_v2(p, "train4")
        assert x0 is not None and np.abs(x0).sum() > 0, f"{sid}: точка без сделок"
        for v in canon_v6.novye(canon_v6._kub(sid))[ax]["setka"][1:]:
            r, x, _ = e6.evaluate_v2({**p, ax: v}, "train4")
            if x is None:
                prich.add(r["prichina"])
            elif not np.array_equal(x, x0):
                return
    pytest.fail(f"{sid}.{ax}: не меняет результат; ne_schitana: {prich}")


@real
@tyazh
def test_lestnica_mutatsiya_ves_ne_delitsya_lovitsya(monkeypatch):
    _, e6 = _para("S1")
    p = _p("S1")
    _, x0, _ = e6.evaluate_v2(p, "train4")
    _, x, _ = e6.evaluate_v2({**p, "lestnica": 0.5}, "train4")
    assert not np.array_equal(x, x0)
    monkeypatch.setattr(isp.IspolnMixin, "_ves", lambda self, sg, rg: super(isp.IspolnMixin, self)._ves(sg, rg))   # мутация: вес ноги не /2
    with pytest.raises(ValueError, match="t0d"):                    # занятость обеих ног = 2 → ФДР t0d отказывает (fdr.zanyatost)
        e6.evaluate_v2({**p, "lestnica": 0.5}, "train4")


@real
@tyazh
def test_mutatsiya_os_ne_dohodit_do_ocenshchika_lovitsya(monkeypatch):
    _, e6 = _para("S1")
    p = _p("S1")
    _, x0, _ = e6.evaluate_v2(p, "train4")
    monkeypatch.setattr(isp, "primenit", lambda r, v6: None)
    _, xm, _ = e6.evaluate_v2({**p, "gap": "propusk", "t_vyhod": "1500", "f_neisp": "zakrytie_d1"}, "train4")
    assert np.array_equal(xm, x0)                                  # без primenit оси не действуют → тесты «меняет результат» падают
    monkeypatch.undo()
    _, x, _ = e6.evaluate_v2({**p, "gap": "propusk", "t_vyhod": "1500", "f_neisp": "zakrytie_d1"}, "train4")
    assert not np.array_equal(x, x0)
