"""Ф-сл2b: ядро S3/S4 (`fast_exec.run_v5`) подключено в `fast_etap1.run` и включено в `pokryto` по флагу YADRO_S34_GOTOV.
1) pokryto: точки S3/S4 (и S1 с T/reentry/chast/trail atr) покрыты при флаге и НЕ покрыты без него; мутация «ось принята
   при ядре без неё» (проверка флага снята) ловится;
2) run делегирует ядру при флаге и эталону без него; результат тот же побайтно (синтетика);
3) замечание проверяющего #631: выход "part" с ценой входа проверяется как в эталоне (_v_bare строгий);
4) быстрый = медленный побайтно на реальных путях train4 (≤ 2021), N точек S3 и S4 (FASTBT_V5_N, гейт — 50);
5) замер с/точку на 20 точках (малая проверка, 1 процесс; потолок).
FASTBT_MAIN=$PWD FASTBT_V5_N=50 python3 -m pytest tests/test_fast_v5_s34_pokryto.py -q"""
import json
import os
import time

import numpy as np
import pytest

from fastbt import exec_e, fast_etap1 as fe, fast_exec as fx, progon_sobol as ps
from fastbt.exec_e_types import Day, Params, Trade
from test_fast_etap1_smoke import _same
from test_fast_v5_pokryto import _eq
from test_fast_v5_s34 import N_SZ, lenta, tochka

RZ = ps.Rezhim("v5", "train4", "seed", False, "t0d")


def tochki(sid, n=300):
    return [p for _, p in ps.tochki(ps.load_kub(sid, RZ), 0, n, RZ)]


def proverka_flaga(fn, monkeypatch) -> None:
    """Контракт: с флагом ядра S3/S4 точка принята, без флага — отклонена; точка без осей S3/S4 принята всегда."""
    s3, s1 = tochki("S3", 5)[0], {k: v for k, v in tochki("S1", 5)[0].items() if k not in ("T", "reentry", "chast")}
    s1.update(T=1, reentry="net", chast="net", trail="sigma:2.0")
    for flag in (True, False):
        monkeypatch.setattr(fx, "YADRO_S34_GOTOV", flag)
        assert fn(s3) is flag, f"S3 при флаге {flag}"
        assert fn(s1), "точка без осей S3/S4 покрыта всегда"
    monkeypatch.setattr(fx, "YADRO_S34_GOTOV", True)


def test_pokryto_po_flagu(monkeypatch):
    proverka_flaga(fe.pokryto, monkeypatch)
    for sid in ("S3", "S4", "S1"):
        pts = tochki(sid)
        assert sum(fe.os_s34(p) for p in pts) > 50 and all(fe.pokryto(p) for p in pts), sid
        monkeypatch.setattr(fx, "YADRO_S34_GOTOV", False)
        assert not any(fe.pokryto(p) for p in pts if fe.os_s34(p)), sid
        monkeypatch.setattr(fx, "YADRO_S34_GOTOV", True)


def test_mutatsiya_os_prinyata_pri_yadre_bez_neyo_lovitsya(monkeypatch):
    """Мутация: pokryto не смотрит флаг (S3/S4 принимается при ядре без неё) — проверка флага обязана упасть."""
    proverka_flaga(fe.pokryto, monkeypatch)
    mut = lambda pt: fe.pokryto(pt) or (fx.YADRO_S34_GOTOV is False and fe.os_s34(pt))    # noqa: E731
    with pytest.raises(AssertionError):
        proverka_flaga(mut, monkeypatch)


def test_run_delegiruet_yadru_i_ravno_etalonu(monkeypatch):
    calls = []
    orig = fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda *a, **k: calls.append(1) or orig(*a, **k))
    n = 0
    for i in list(range(0, 40)) + list(range(N_SZ, N_SZ + 40)):
        p = tochka(i)
        days, sig, sigma, close, div, urov = lenta(10_000 + i)
        kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
        try:
            a = exec_e.run("T", days, sig, sigma, p, **kw)
        except ValueError:
            continue
        _same(a, fe.run("T", days, sig, sigma, p, **kw))
        n += 1
    assert n > 40 and len(calls) >= n - 5                     # ядро реально считало (лента без осей S3/S4 идёт прежним ядром)
    monkeypatch.setattr(fx, "YADRO_S34_GOTOV", False)         # без флага — эталон, ядро не вызвано
    calls.clear()
    p = tochka(0)
    days, sig, sigma, close, div, urov = lenta(10_000)
    fe.run("T", days, sig, sigma, p, urovni=urov, close=close, div=div)
    assert not calls


def test_part_s_cenoy_vhoda_proveryaetsya_kak_v_etalone():
    """#631: выход "part" с ценой, равной цене входа, в другом баре дня входа вне [low; high] — отказ (до правки пропускался)."""
    d = Day(np.array([100.0, 106.0]), np.array([101.0, 110.0]), np.array([99.0, 105.0]), np.array([100.5, 106.0]), False)
    t = Trade("T", 0, 0, 0, 100.0, 0, 1, 100.0, "part", 1, 0.0, 0.0, 0.0, 0.5, None)
    with pytest.raises(ValueError, match="исполнение вне"):
        fx._v_bare("T", [d], None, Params(), [t])
    t2 = Trade("T", 0, 0, 0, 100.0, 0, 1, 107.0, "part", 1, 0.0, 0.0, 0.0, 0.5, None)
    fx._v_bare("T", [d], None, Params(), [t2])               # цена в баре — без отказа


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="малая проверка на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_ekviv_v5_s34(sid, monkeypatch):
    calls = []
    orig = fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda *a, **k: calls.append(1) or orig(*a, **k))
    sr = ps.fabrika_real(sid, RZ)
    pts = ps.tochki(ps.load_kub(sid, RZ), 0, int(os.environ["FASTBT_V5_N"]), RZ)
    nfast = 0
    for i, pt in pts:
        rs, ps_, ts = ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
        rf, pf, tf = ps.schitat_tochku(sr, sid, i, pt, "t", False, True, rz=RZ)
        nfast += rf["sloy"] == "fast"
        assert rf["sloy"] == "fast" or rf["status"] != "ok", (sid, i)
        j = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("t_sek", "sloy")}, sort_keys=True, default=repr)  # noqa: E731
        assert j(rs) == j(rf), (sid, i)
        assert _eq(ps_, pf) and _eq(ts, tf), (sid, i)
    assert pts and nfast > 0 and calls, "ядро S3/S4 не вызывалось"


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="замер на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_zamer_s34(sid, monkeypatch, capsys):
    """G4 (малая проверка, 1 процесс, 20 точек): с/точку эталон (флаг снят) / ядро; потолок — одна машина, оценщик целиком."""
    sr = ps.fabrika_real(sid, RZ)
    pts = ps.tochki(ps.load_kub(sid, RZ), 0, 20, RZ)
    ps.schitat_tochku(sr, sid, pts[0][0], pts[0][1], "t", False, True, rz=RZ)        # прогрев: компиляция, кэши входов
    t = []
    for flag in (False, True):
        monkeypatch.setattr(fx, "YADRO_S34_GOTOV", flag)
        t0 = time.perf_counter()
        for i, pt in pts:
            ps.schitat_tochku(sr, sid, i, pt, "t", False, True, rz=RZ)
        t.append((time.perf_counter() - t0) / len(pts))
    with capsys.disabled():
        print(f"\nзамер {sid}: до {t[0]:.3f} с/точку, после {t[1]:.3f} с/точку (20 точек, 1 процесс)")
