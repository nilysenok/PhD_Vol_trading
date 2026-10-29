"""Ф-сл3d: оси #653 (b0, b0_c, sp, step_s, a_rang + вход rang) и #660 (pir) включены в `fast_etap1.pokryto` по флагу YADRO_OSI3_GOTOV.
1) pokryto: точки с этими осями покрыты при флаге и НЕ покрыты без него; мутация «ось принята при ядре без неё» ловится;
2) `fast_etap1.run` с rang делегирует ядру при флаге и эталону без него (побайтно, синтетика);
3) оси, добавленные в медленный путь позже (time_decay, piv_sl/piv_tp/vwap_m, оси v5 на стоп-заявке, t_vyhod) — по-прежнему ne_znaet / эталон;
4) быстрый = медленный побайтно на путях train4 (≤ 2021), по FASTBT_V5_N точек на каждый Sk v5, половина — с осями #653/#660;
5) замер с/точку на 20 точках (малая проверка, 1 процесс; потолок).
NUMBA_CACHE_DIR=<свежий> FASTBT_MAIN=$PWD FASTBT_V5_N=200 python3 -m pytest tests/test_fast_v5_osi3_pokryto.py -q"""
import dataclasses
import os

import numpy as np
import pytest

from fastbt import exec_e, fast_etap1 as fe, fast_exec as fx, progon_sobol as ps
from test_fast_etap1_smoke import _same
from test_fast_v5_osi3 import rang_of, tochka as tochka3
from test_fast_v5_pir import tochka as tochka_pir
from test_fast_v5_s34 import lenta

RZ = ps.Rezhim("v5", "train4", "seed", False, "t0d")
SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
OSI3 = {"tejk": "sila:1.0", "bezubytok": "atr:1.0", "stop_v_plyus": "1.0:0.25", "trail": "step:0.5", "pir": 0.5}


def tochki(sid, n):
    return [p for _, p in ps.tochki(ps.load_kub(sid, RZ), 0, n, RZ)]


def proverka_flaga(fn, monkeypatch) -> None:
    """Контракт: с флагом точка с осью #653/#660 принята, без флага — отклонена; точка без осей принята всегда."""
    base = {"stop_nach": "sigma:2.0"}
    for flag in (True, False):
        monkeypatch.setattr(fx, "YADRO_OSI3_GOTOV", flag)
        for k, v in OSI3.items():
            assert fn({**base, k: v}) is flag, f"{k} при флаге {flag}"
        assert fn(base), "точка без осей #653/#660 покрыта всегда"
    monkeypatch.setattr(fx, "YADRO_OSI3_GOTOV", True)


def test_pokryto_po_flagu(monkeypatch):
    proverka_flaga(fe.pokryto, monkeypatch)
    assert not fe.pokryto({"stop_nach": "sigma:2.0", "pir": 0.5, "rho": 0.3})        # прочие непокрытые оси по-прежнему режут
    assert not fe.os_osi3({"pir": 0.5})                                              # не точка v5 — не ось v5


def test_mutatsiya_os_prinyata_pri_yadre_bez_neyo_lovitsya(monkeypatch):
    """Мутация: pokryto не смотрит флаг ядра (ось принята при ядре без неё) — проверка флага обязана упасть."""
    mut = lambda pt: fe.pokryto(pt) or (fx.YADRO_OSI3_GOTOV is False and fe.os_osi3(pt))    # noqa: E731
    with pytest.raises(AssertionError):
        proverka_flaga(mut, monkeypatch)


def test_run_s_rang_delegiruet_yadru_i_ravno_etalonu(monkeypatch):
    calls = []
    orig = fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda *a, **k: calls.append(1) or orig(*a, **k))
    n = 0
    for i in range(120):
        p = (tochka3 if i % 2 else tochka_pir)(i)
        days, sig, sigma, close, div, _ = lenta(40_000 + i)
        kw = dict(close=close, div=div, rang=rang_of(i, len(days)) if (p.a_rang or p.b0_c) else None)
        try:
            a = exec_e.run("T", days, sig, sigma, p, **kw)
        except ValueError:
            continue
        _same(a, fe.run("T", days, sig, sigma, p, **kw))
        n += 1
    assert n > 80 and len(calls) >= n - 5
    monkeypatch.setattr(fx, "YADRO_OSI3_GOTOV", False)        # без флага — эталон, ядро не вызвано
    calls.clear()
    for i in range(12):
        p = tochka3(i)
        days, sig, sigma, close, div, _ = lenta(40_000 + i)
        fe.run("T", days, sig, sigma, p, close=close, div=div, rang=rang_of(i, len(days)))
    assert not calls


def test_pozdnie_osi_medlennogo_puti_ostayutsya_v_etalone(monkeypatch):
    """time_decay (#668), piv_sl/piv_tp/vwap_m (#677), оси v5 на стоп-заявке тренда (#666/#667/#671) — ne_znaet; ядро их не считает молча."""
    base = tochka3(0)
    for kw in (dict(time_decay=True), dict(piv_sl="next"), dict(piv_tp="P"), dict(piv_sl="vwap", vwap_m=2.0)):
        p = dataclasses.replace(base, **kw)
        assert fx.ne_znaet(p), kw
        with pytest.raises(NotImplementedError):
            days, sig, sigma, close, div, _ = lenta(41_000)
            fx.run_v5("T", days, sig, sigma, p, close=close, div=div, rang=None)
    for kw in (dict(b0=1.0), dict(step_s=0.5), dict(pir=0.5), dict(sp=(1.0, 0.25)), dict(ed_s="atr"), dict(chand=2.0)):
        assert fx.ne_znaet(dataclasses.replace(base, vhod_tip="stop_zayavka", b=None, **kw)), kw
    assert fe.pokryto({"stop_nach": "sigma:2.0", "t_vyhod": "open"}) is False       # t_vyhod ≠ close — эталон
    monkeypatch.setattr(fx, "YADRO_OSI3_GOTOV", False)
    assert fx.ne_znaet(dataclasses.replace(base, b0=1.0)) == ["b0"]                   # без флага оси #653 снова неизвестны ядру


def _vybor(sid, n):
    """n точек: половина с осями #653/#660 (если есть в кубе), остальные — по порядку."""
    pts = ps.tochki(ps.load_kub(sid, RZ), 0, 20 * n, RZ)
    osi = [x for x in pts if fe.os_osi3(x[1])][: n // 2]
    return osi + [x for x in pts if x not in osi][: n - len(osi)], len(osi)


class Shpion:
    """Подмена exec_e.run на время оценки точки: каждый вызов оценщика (реальные бары ≤ 2021, σ, ранг, Params точки куба) считается
    эталоном и быстрым `fast_etap1.run` на тех же входах; результат побайтно равен — иначе AssertionError. Оценщик получает эталон.
    Якорь куба v5 — t_vyhod = open2 (#680/#682): ядро его не знает, поэтому в сверке на его месте стоит close (остальные оси точки — свои);
    через оценщик быстрый слой куб v5 не берёт вовсе (pokryto ложно на open2) — потолок эффекта, см. отчёт."""

    def __init__(self):
        self.t = [0.0, 0.0]
        self.n = self.nyadro = self.nrang = 0

    def __call__(self, ticker, days, sig, sigma, p, **kw):
        import time as _t
        pc, vals = dataclasses.replace(p, t_vyhod="close"), []
        for run in (fe._SLOW_RUN, fe.run):
            t0 = _t.perf_counter()
            try:
                vals.append(run(ticker, days, sig, sigma, pc, **kw))
            except ValueError as e:
                vals.append(str(e))
            self.t[run is fe.run] += _t.perf_counter() - t0
        if isinstance(vals[0], str) or isinstance(vals[1], str):
            assert vals[0] == vals[1], (ticker, vals)
        else:
            _same(*vals)
        self.n += 1
        self.nrang += kw.get("rang") is not None
        return fe._SLOW_RUN(ticker, days, sig, sigma, p, **kw)


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="малая проверка на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", SIDS)
def test_ekviv_v5_osi3(sid, monkeypatch):
    """Быстрый = медленный побайтно (Shpion: каждый run оценщика сверяется на реальных барах) по FASTBT_V5_N точкам Sk."""
    calls = []
    orig = fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda *a, **k: calls.append(k.get("rang") is not None) or orig(*a, **k))
    sr = ps.fabrika_real(sid, RZ)
    pts, nosi = _vybor(sid, int(os.environ["FASTBT_V5_N"]))
    sh = Shpion()
    monkeypatch.setattr(exec_e, "run", sh)
    for i, pt in pts:
        ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
    # тренд S3/S4: на стоп-заявке открытия (vhod_open, #666/#671) все вызовы — ne_znaet, ядро не вызвано и не должно (эталон)
    assert pts and sh.n and (bool(calls) is (sid not in ("S3", "S4"))), "ядро вызвано не там / не вызвано"
    print(f"\n{sid}: точек {len(pts)}, с осями #653/#660 {nosi}, run сверено {sh.n}, вызовов ядра {len(calls)}, с rang {sum(calls)}")


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="замер на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", SIDS)
def test_zamer_osi3(sid, monkeypatch, capsys):
    """Малая проверка, 1 процесс, 20 точек: время run за точку (17 тикеров) — эталон (до) / быстрый слой с флагом (после).
    Только `exec_e.run`: оценщик целиком быстрый слой на кубе v5 не берёт (t_vyhod = open2) — это потолок ускорения run, не точки."""
    sr = ps.fabrika_real(sid, RZ)
    pts, nosi = _vybor(sid, 20)
    sh0 = Shpion()
    monkeypatch.setattr(exec_e, "run", sh0)
    ps.schitat_tochku(sr, sid, pts[0][0], pts[0][1], "t", False, False, rz=RZ)        # прогрев: компиляция numba, кэши входов
    sh = Shpion()
    monkeypatch.setattr(exec_e, "run", sh)
    for i, pt in pts:
        ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
    with capsys.disabled():
        print(f"\nзамер {sid}: до {sh.t[0] / len(pts):.3f} с/точку, после {sh.t[1] / len(pts):.3f} с/точку "
              f"(run, 20 точек, из них с осями {nosi}, 1 процесс)")
