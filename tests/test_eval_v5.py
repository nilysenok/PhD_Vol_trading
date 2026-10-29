"""Л9: оценщик точки куба v5 (fastbt/eval_v5.py, fastbt/eval_v5_osi.py).

Гейт «v5 на якорях = v4»: точка v5 со всеми новыми осями на yakor (сигнал семьи v4, semya="v4") даёт ряды, равные
EvaluatorV2 (ФДР t0d) для соответствующей точки v4, ≤ 1e-12, ≥ 50 точек × 3 тикера train4 на каждую Sk.
Мутации: «новая ось игнорируется» (разбор и оценщик), «фильтр на дне d+1» — тесты-проверки обязаны их найти.
Реальные данные — train4 (≤ 2019) через fastbt.data/etap1 (FASTBT_MAIN — своя копия); нет файла — skip.
"""
import copy
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import canon_v5, eval_v5, eval_v5_osi
from fastbt.eval_v5_osi import chisla, razbor, yakor

SIDS = canon_v5.SIDS
TOL = 1e-12


def _sp5(sid):
    k = canon_v5.load_kub(sid)
    return {n: v["setka"][len(v["setka"]) // 2] for n, v in k["signal"].items()}


def osi_ignoriruyutsya(sid) -> list:
    """(ось, значение), у которых разбор совпадает с якорем, хотя канон точки ≠ якорь."""
    k, sp, ya = canon_v5.load_kub(sid), _sp5(sid), yakor(sid)
    base = asdict(razbor(sid, {**sp, **ya}))
    out = []
    for sec in ("filtr", "vhod", "vyhod"):
        for n, v in k[sec].items():
            for x in v["setka"]:
                st = {"stop_nach": "atr:2.0"} if str(x).startswith(("atr", "rang")) and n == "bezubytok" else {}
                c = canon_v5.canon_v5(sid, {**sp, **ya, **st, n: x})
                c0 = canon_v5.canon_v5(sid, {**sp, **ya, **st})
                if c is None or c == c0:
                    continue
                if asdict(razbor(sid, c)) == (base if not st else asdict(razbor(sid, c0))):
                    out.append((n, x))
    return out


def test_chisla():
    assert chisla("atr:2.0") == (2.0,) and chisla("sila:0.5:1") == (0.5, 1.0) and chisla("1/3:1.5") == (1 / 3, 1.5)
    assert chisla("1.5:0.5") == (1.5, 0.5) and chisla("parab:0.02:0.1") == (0.02, 0.1)


@pytest.mark.parametrize("sid", SIDS)
def test_neizvestnaya_os_otkaz(sid):
    with pytest.raises(ValueError, match="не знает"):
        razbor(sid, {**_sp5(sid), **yakor(sid), "hurst_th": 0.5})


@pytest.mark.parametrize("sid", SIDS)
def test_kazhdaya_os_menyaet_razbor(sid):
    assert osi_ignoriruyutsya(sid) == []


def test_mutaciya_os_ignoriruetsya_v_razbore_lovitsya(monkeypatch):
    monkeypatch.setattr(eval_v5_osi, "_trail", lambda r, v: None)
    assert ("trail", "atr:2.0") in osi_ignoriruyutsya("S1")


# ---------------- реальные данные train4 ----------------

def _est_dannye() -> bool:
    from fastbt import daily_1850
    return daily_1850.out_path().exists()


real = pytest.mark.skipif(not _est_dannye(), reason="нет дневного ряда 18:50 в FASTBT_MAIN")
_INP: dict = {}


def _inp(sid):
    if sid not in _INP:
        from fastbt import etap1
        _INP[sid] = (etap1.build_inputs(sid, period="train4"), etap1.base_point(sid))
    return _INP[sid]


def _ocenshchiki(sid, n_tik=3, semya="v4"):
    from fastbt.etap1_eval_v2 import EvaluatorV2
    inp, base = _inp(sid)
    sk = copy.copy(inp.sk)
    sk.tickers = list(inp.sk.tickers)[:n_tik]
    e2, e5 = EvaluatorV2.novaya_setka(sid, inp, base), eval_v5.EvaluatorV5(sid, inp, base, semya=semya)
    e2.sk = e5.sk = sk
    return e2, e5


def _pary(sid, n, seed=20261029):
    """[(точка v5, точка v4)]: сигнал v4 случайный из сетки v4; оси v5 с аналогом v4 (стоп σ, трейл σ, τ) — случайные."""
    from fastbt.etap1_poisk import load_setka
    k, ax4, rng = canon_v5.load_kub(sid), load_setka(sid)["axes"], random.Random(seed + int(sid[1]))
    trend = sid in canon_v5.TREND
    stop = [v for v in k["vyhod"]["stop_nach"]["setka"] if str(v).startswith("sigma")]
    trail = [v for v in k["vyhod"]["trail"]["setka"] if v == "net" or str(v).startswith("sigma")]
    out = []
    for i in range(n):
        sp4 = {a: rng.choice(list(g)) for a, g in ax4.items() if a in _inp(sid)[0].sk.sig_names}
        ya = yakor(sid)
        if i >= 10:                                          # первые 10 — чистый якорь, дальше — оси с аналогом v4
            ya.update(stop_nach=rng.choice(stop), trail=rng.choice(trail), tau=rng.choice(k["vyhod"]["tau"]["setka"]))
        pt4 = {**eval_v5_osi.PT4, **sp4, "k_s": chisla(ya["stop_nach"])[0], "tau": ya["tau"],
               "k_tr": "net" if ya["trail"] == "net" else chisla(ya["trail"])[0]}
        pt4.update(vhod_tip="stop_zayavka") if trend else pt4.update(vhod=0.0)
        out.append(({**ya, **sp4}, pt4))
    return out


@real
@pytest.mark.skipif(__import__("os").environ.get("FASTBT_V6_REAL") != "1", reason="тяжёлый (~230 с на 6 Sk, хук ≤ 300 с): FASTBT_V6_REAL=1")
@pytest.mark.parametrize("sid", SIDS)
def test_yakor_v5_ravno_v4(sid):
    e2, e5 = _ocenshchiki(sid)
    n_ok, n_sdelok, worst = 0, 0, 0.0
    for p5, p4 in _pary(sid, 50):
        # К-wf5 (Ж-137 п. 6): якорь v5 теперь t_vyhod=open2 (сознательная смена, было close); v4-сторона получает тот же t_vyhod
        r4, x4, t4 = e2.evaluate_v2({**p4, "t_vyhod": canon_v5.load_kub(sid)["yakorya_progon"]["t_vyhod"]}, "train4")
        r5, x5, t5 = e5.evaluate_v2(p5, "train4")
        assert r4["status"] == r5["status"] == "ok", (r4.get("prichina"), r5.get("prichina"))
        worst = max(worst, float(np.max(np.abs(x4 - x5))), float(np.nanmax(np.abs(t4 - t5))))
        assert np.array_equal(np.isnan(t4), np.isnan(t5))
        n_ok += 1
        n_sdelok += sum(len(r.trades) for r in e5._last[4])
    print(f"{sid}: точек {n_ok}, сделок {n_sdelok}, max|Δ| {worst:.3e}")
    assert n_ok >= 50 and worst <= TOL and n_sdelok > 0


IZM = dict(k_e=0.5, T=3, reentry="da", stop_nach="atr:2.0", tejk="atr:2.0", trail="atr:2.0", tau=10, chast="1/2:1",
           pir=0.5, smena="net", stop_v_plyus="2:1", razmer_tip="risk_sigma", m_sily="lin", adx_th=25, vol_th=1.0,
           atr_lo=10, rsi_th=30, mom_n=3, dow=1, squeeze_pct=50)
SP_S1 = dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=1.5, z_exit=0.5)


def ne_menyayut(e5, sid="S1") -> list:
    """Оси, значение которых ≠ yakor, но ряд портфеля совпадает с якорем побайтно (оценщик ось молча не взял)."""
    base = e5.evaluate_v2({**yakor(sid), **SP_S1}, "train4")[1]
    out = []
    for n, v in IZM.items():
        r, x, _ = e5.evaluate_v2({**yakor(sid), **SP_S1, n: v}, "train4")
        if r["status"] != "ok" or np.array_equal(x, base):
            out.append(n)
    return out


@real
def test_novaya_os_menyaet_ryad():
    _, e5 = _ocenshchiki("S1", n_tik=4, semya="v5")
    assert ne_menyayut(e5) == []


@real
def test_mutaciya_os_ignoriruetsya_v_ocenshchike_lovitsya(monkeypatch):
    _, e5 = _ocenshchiki("S1", n_tik=4, semya="v5")
    monkeypatch.setattr(eval_v5_osi, "_tejk", lambda r, sid, v: None)
    assert "tejk" in ne_menyayut(e5)


def narusheniya_prichinnosti(td, filtr, sig, tochki=(300, 520, 777, 1001, 1250)) -> list:
    """Дни d, где маска фильтров v5 по данным ≤ d ≠ маске по полному ряду (фильтр смотрит в d+1 и дальше)."""
    full = eval_v5.maska(td, "kontr", filtr, sig)
    out = []
    for d in tochki:
        cut = SimpleNamespace(**{k: np.asarray(getattr(td, k))[:d + 1] for k in ("high", "low", "close", "volume", "dates")})
        if eval_v5.maska(cut, "kontr", filtr, sig[:d + 1])[d] != full[d]:
            out.append(d)
    return out


def _td_sig():
    inp, _ = _inp("S1")
    t = inp.sk.tickers[0]
    td, ind = inp.sk._td[t]
    from fastbt.signals.interface import get_family
    eval_v5.EvaluatorV5("S1", inp, _inp("S1")[1])                      # регистрация семей v5
    return td, get_family("S1", setka="v5")(td, ind, [SP_S1]).sig[0]


FILTR = dict(adx_th=25, atr_lo=10, vol_th=0.7, rsi_th=30, mom_n=3, dow=1, squeeze_pct=40)


@real
def test_filtr_na_dne_d():
    td, sig = _td_sig()
    assert narusheniya_prichinnosti(td, FILTR, sig) == []
    m = eval_v5.maska(td, "kontr", FILTR, sig)
    assert 0 < m.sum() < len(m)                                         # фильтр что-то режет и что-то пропускает


@real
def test_mutaciya_filtr_na_dne_d_plus_1_lovitsya(monkeypatch):
    td, sig = _td_sig()
    orig = eval_v5.primenit
    monkeypatch.setattr(eval_v5, "primenit", lambda df, tp, f, sig=None: np.r_[orig(df, tp, f, sig=sig)[1:], True])
    assert narusheniya_prichinnosti(td, FILTR, sig) != []


@real
def test_run_rows_rang_kopiya_run_e1():
    """Копия сборки строк с rang = etap1_run.run_e1, когда ранг не используется Params (a_rang, b0_c нет)."""
    from fastbt import etap1_run
    from fastbt.eval_v5_osi import razbor as rz
    _, e5 = _ocenshchiki("S1", n_tik=2, semya="v5")
    e5.evaluate_v2({**yakor("S1"), **SP_S1, "tejk": "atr:2.0"}, "train4")
    e5.r5 = rz("S1", {**yakor("S1"), **SP_S1, "tejk": "atr:2.0"})
    t, inp = e5.sk.tickers[0], e5.inp
    s0, b, a, dates, sig, sg, al = e5._start("train4")[t]
    days = e5._days5[0]
    from dataclasses import replace
    p = replace(e5._params(e5.r5.pt4, t, 5.0, False, e5.r5.nov, {}), **e5.r5.ext)
    kw = dict(close=inp.close_n6[t][:b + 1], win=(a, b), div=inp.div[t][:b + 1], fdr=e5.fdr)
    r0 = etap1_run.run_e1(t, days, sig, sg, p, **kw)
    r1 = eval_v5.run_rows(t, days, sig, sg, p, rang=np.zeros(len(days)), **kw)
    assert len(r0.trades) > 0
    for f in ("g", "sides", "expo", "cnt", "noch", "short"):
        assert np.array_equal(getattr(r0, f), getattr(r1, f)), f
