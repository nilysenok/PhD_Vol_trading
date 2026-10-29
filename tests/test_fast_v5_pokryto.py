"""Ф-сл1: быстрый слой v5 — `pokryto` пускает точки v5 (ось rang_tip), `digest` входов — один раз на набор входов.
Находка: из точек куба v5 `pokryto` отсекал только rang_tip ≠ tikernyj (оценщик считает ранг сам; в `run` идёт rang ≠ None,
а такой вызов `run` уводит в эталон по `_slow_only`). Остальные оси v5 `pokryto` и так пропускал.
1) pokryto без данных: v5-точка с rang_tip ∈ RANG_TIPY покрыта, чужое значение и непокрытые оси (rho, tejk_signal, lestnica,
   m_tip, …) — нет; точка не-v5 с rang_tip ≠ tikernyj — как раньше, эталон;
2) мутация «непокрытая ось принята быстрым путём» ловится проверкой `proverka_pokryto`;
3) digest: второй вызов на тех же входах не пересчитывает, подмена листа — пересчитывает (Д5);
4) быстрый = медленный побайтно на реальных путях train4 (≤ 2021), по N точек каждого Sk (только если задан FASTBT_V5_N;
   гейт Ф-сл1 — 200). Без переменной — пропуск.
FASTBT_MAIN=$PWD FASTBT_V5_N=200 python3 -m pytest tests/test_fast_v5_pokryto.py -q
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import fast_etap1 as fe
from fastbt import progon_sobol as ps
from fastbt.rangi_v6 import RANG_TIPY

RZ = ps.Rezhim("v5", "train4", "seed", False, "t0d")
NEPOKRYTO = ({"rho": 0.3}, {"tejk_signal": "pol_puti"}, {"lestnica": "dva"}, {"m_tip": "lin"}, {"ed_lim": "mediana"},
             {"storona": "long"}, {"gap": "zakr"}, {"filtr": "x"}, {"rang_tip": "foo"})


def tochki(sid, n=300):
    return ps.tochki(ps.load_kub(sid, RZ), 0, n, RZ)


def proverka_pokryto(fn) -> None:
    """Контракт: `fn` отклоняет точку v5 с любой непокрытой осью и принимает все значения rang_tip."""
    pt = tochki("S1", 5)[0][1]
    for t in RANG_TIPY:
        assert fn({**pt, "rang_tip": t}), f"rang_tip={t} должен быть покрыт"
    for bad in NEPOKRYTO:
        assert not fn({**pt, **bad}), f"ось {bad} принята быстрым путём"


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_pokryto_v5_tolko_rang_tip(sid):
    pts = [p for _, p in tochki(sid)]
    assert len(pts) > 50
    assert all(fe.pokryto(p) for p in pts)                                   # все оси куба v5 покрыты
    assert {p["rang_tip"] for p in pts} == set(RANG_TIPY)                    # включая rang_tip ≠ tikernyj


def test_pokryto_nepokrytye_osi_i_ne_v5():
    proverka_pokryto(fe.pokryto)
    pt = tochki("S1", 5)[0][1]
    ne5 = {k: v for k, v in pt.items() if k != "stop_nach"}
    assert fe.pokryto({**ne5, "rang_tip": "tikernyj"}) and not fe.pokryto({**ne5, "rang_tip": "obshchiy"})


def test_mutatsiya_nepokrytaya_os_lovitsya():
    def mut(pt):                                       # «принимает всё»: проверка umolch снята
        return True

    def mut2(pt):                                      # ось rho не смотрится
        return fe.pokryto({k: v for k, v in pt.items() if k != "rho"})

    proverka_pokryto(fe.pokryto)
    for m in (mut, mut2):
        with pytest.raises(AssertionError):
            proverka_pokryto(m)


def test_digest_odin_raz_na_nabor_vkhodov(monkeypatch):
    calls = []
    inp = SimpleNamespace(sk=[1], sigma={"a": [1.0, 2.0]}, bars={"a": [1, 2]}, close_n6={}, rezhim={}, sobytiya={}, tick={},
                          rate=[0.1], meta={}, div={}, commission=0.0004)

    def orig(i):
        calls.append(1)
        return f"d{len(calls)}"

    monkeypatch.setitem(fe._ORIG, "digest", orig)
    fe._MEM.pop(("digest", id(inp)), None)
    a = fe._digest_fast(inp)
    assert [fe._digest_fast(inp) for _ in range(5)] == [a] * 5 and len(calls) == 1
    inp.sigma = {"a": [1.0, 3.0]}                      # подмена листа — новый хэш (Д5)
    assert fe._digest_fast(inp) != a and len(calls) == 2


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is b
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="малая проверка на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_ekviv_v5(sid):
    n = int(os.environ["FASTBT_V5_N"])
    sr = ps.fabrika_real(sid, RZ)
    pts = ps.tochki(ps.load_kub(sid, RZ), 0, n, RZ)
    nfast = 0
    for i, pt in pts:
        rs, ps_, ts = ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
        rf, pf, tf = ps.schitat_tochku(sr, sid, i, pt, "t", False, True, rz=RZ)
        nfast += rf["sloy"] == "fast"
        assert rf["sloy"] == "fast" or rf["status"] != "ok", (sid, i)       # посчитанная точка шла быстрым путём
        drop = ("t_sek", "sloy")
        j = lambda r: json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True, default=repr)  # noqa: E731
        assert j(rs) == j(rf), (sid, i)
        assert _eq(ps_, pf) and _eq(ts, tf), (sid, i)
    assert pts and nfast > 0
