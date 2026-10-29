"""Ф-сл3c (лист 3 из П-0): ось pir (пирамидинг, ≤ 2 добавок, риск ≤ 2ρ) в fast_exec.kernel_v5. Стоит поверх #653 (оси 3a/3b).
Эталон — `exec_e.run` (Run + V6.v6_pir); `fast_exec.run_v5` обязан совпасть ПОБАЙТНО (допуск 0, proverka.md п. 3), включая ноги "pir".
1) синтетика: 1200 путей (ранг, chast, оси 3a, дивиденды, тик, дни pusto); 2) 50 точек на барах train4 (AFLT, ≤ 2019);
3) крайние ленты; 4) мутации эталона: порог 2ρ → 3ρ, добавка в баре входа, комиссия выхода на w вместо w − Σadds — сверка обязана разойтись.
NUMBA_CACHE_DIR=<свежий каталог> FASTBT_NO_CACHE=1 python3 -m pytest tests/test_fast_v5_pir.py -q"""
import dataclasses

import numpy as np
import pytest

from fastbt import exec_e, exec_e_v6, fast_exec as fx
from fastbt.exec_e_types import Day, Params
from test_fast_etap1_smoke import TICK, _same
from test_fast_v5_osi3 import rang_of, sverit
from test_fast_v5_s34 import ch, lenta

N = 1200


def tochka(i: int) -> Params:
    """Лимитный вход, b, bary, pir ∈ {0,2; 0,5; 1}; поверх — chast, sp, step_s, b0, a_rang, be, k_tr (ноги и стоп меняются)."""
    rng = np.random.default_rng(20261031 + i)
    kw = {}
    if rng.random() < 0.35:
        kw["chast"] = ch(rng, (0.5, 1.0), (1 / 3, 1.5))
    if rng.random() < 0.3:
        kw["sp"] = ch(rng, (1.0, 0.25), (1.5, 0.5))
    if rng.random() < 0.3:
        kw["step_s"] = ch(rng, 0.5, 1.0)
    if rng.random() < 0.25:
        kw["b0"] = ch(rng, 0.75, 1.0)
    elif rng.random() < 0.2:
        kw["be"] = True
    if rng.random() < 0.3:
        kw["a_rang"] = (1.5, 1.0)
    return Params(b=ch(rng, 1.0, 2.0, 3.0), pir=ch(rng, 0.2, 0.5, 1.0), k=ch(rng, 0.25, 0.5, 1.0),
                  f=ch(rng, "propusk", "zakrytie_d1", "povtor"), tau=ch(rng, None, 5, 8), s=ch(rng, "fiks", "ezhedn"), comm=0.0004,
                  side=ch(rng, 1, -1), tick=ch(rng, None, 0.0, TICK), k_tr=ch(rng, None, 1.5), exit_signal=ch(rng, True, False),
                  stop_touch=ch(rng, True, False), reentry=ch(rng, True, False), **kw)


def test_sintetika_yadro_ravno_etalonu_na_1200_putyah():
    plokho, cnt = [], dict(pir=0, dve=0, urezano=0, zakryta=0)
    for i in range(N):
        p = tochka(i)
        assert not fx.ne_znaet(p), p
        days, sig, sigma, close, div, _ = lenta(40_000 + i)
        rang = rang_of(i, len(days))
        if not sverit(days, sig, sigma, p, close, div, rang):
            plokho.append(i)
            continue
        try:
            r = exec_e.run("T", days, sig, sigma, p, close=close, div=div, rang=rang)
        except ValueError:                                  # тот же отказ у ядра проверил sverit
            continue
        legs = [t for t in r.trades if t.reason == "pir"]
        cnt["pir"] += bool(legs)
        cnt["dve"] += any(sum(t.parent == k for t in legs) == 2 for k in {t.parent for t in legs})
        cnt["urezano"] += any(t.frac < p.pir - 1e-12 for t in legs)
        cnt["zakryta"] += any(t.exit_day is not None for t in legs)
    assert not plokho, plokho[:10]
    assert cnt["pir"] >= 200 and min(cnt["dve"], cnt["urezano"], cnt["zakryta"]) >= 20, cnt     # ось реально работает


def test_krajnie_lenty():
    e, one = np.empty(0), Day(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), False)
    pust = Day(e, e, e, e, True)
    for days in ([], [one], [one, one], [pust, pust, pust], [one, pust, one, one]):
        n = len(days)
        for sigma in (np.full(n, 0.02), np.zeros(n), np.full(n, float("nan"))):
            for i in range(6):
                assert sverit(days, np.ones(n), sigma, tochka(i), None, None, np.full(n, 0.5)), (n, i)


def test_pir_vne_granic_otkaz_tot_zhe():
    days, sig, sigma, close, div, _ = lenta(41_000)
    for p in (dataclasses.replace(tochka(0), pir=1.5), dataclasses.replace(tochka(0), pir=-0.1), dataclasses.replace(tochka(0), k=None)):
        assert sverit(days, sig, sigma, p, close, div, rang_of(1, len(days))), p
    assert fx.ne_znaet(dataclasses.replace(tochka(0), tejk="k_srednemu")) == ["tejk"]       # режим вне ядра по-прежнему отдаётся эталону


@pytest.fixture(scope="module")
def aflt():
    from test_fast_v5_ekviv import _realnye
    return _realnye("AFLT")


def test_realnye_50_tochek_pobajtno(aflt):
    days, sigma = aflt
    n, ok, s_pir = len(days), [], 0
    for i in range(50):
        rng = np.random.default_rng(i)
        p = dataclasses.replace(tochka(i), tau=ch(rng, None, 10, 20), s="fiks")
        sig = np.repeat(rng.choice([-1, 0, 1], n // 4 + 1), 4)[:n].astype(float)
        rang = rang_of(i, n)
        div = np.where(rng.random(n) < 0.004, rng.uniform(0.5, 3.0, n), 0.0)
        ok.append(sverit(days, sig, sigma, p, None, div, rang))
        s_pir += any(t.reason == "pir" for t in fx.run_v5("T", days, sig, sigma, p, div=div, rang=rang).trades)
    assert all(ok), [i for i, x in enumerate(ok) if not x]
    assert s_pir >= 20, s_pir


# --- мутации эталона: без неё ядро = эталон, с ней — расходятся (и мутант действует) ---
def _risk_3(mp):                           # порог риска 3ρ вместо 2ρ
    mp.setattr(exec_e_v6, "RISK_MAX", 3.0)


def _v_bare_vhoda(mp):                     # добавка разрешена и в баре входа (скан сам начинается с бара после входа — проверяем его после входа)
    from fastbt import exec_e_run
    mp.setattr(exec_e_v6, "part_allowed", lambda day, bar, ent: (day, bar) >= ent)
    orig = exec_e_run.Run._stop_vhoda

    def f(s, i, j, d, b, px, S):
        hit = orig(s, i, j, d, b, px, S)
        sg = s._sg(j)
        if not hit and s.p.pir and S is not None and sg is not None:
            s.v6_pir(i, j, b, d.o[b], d.h[b], d.l[b], S, sg)
        return hit
    mp.setattr(exec_e_run.Run, "_stop_vhoda", f)


def _fee_na_w(mp):                         # комиссия выхода родителя на весь w, а не на w − Σadds
    orig = exec_e_v6.V6.v6_exit

    def f(s, i, b, px):
        orig(s, i, b, px)
        return 0.0
    mp.setattr(exec_e_v6.V6, "v6_exit", f)


@pytest.mark.parametrize("name", ["risk_3", "bar_vhoda", "fee_na_w"])
def test_mutatsiya_etalona_lovitsya(name, monkeypatch):
    mut = {"risk_3": _risk_3, "bar_vhoda": _v_bare_vhoda, "fee_na_w": _fee_na_w}[name]
    chislo = 0
    for i in range(N):
        p = tochka(i)
        days, sig, sigma, close, div, _ = lenta(42_000 + i)
        rang = rang_of(i, len(days))
        kw = dict(close=close, div=div, rang=rang)
        try:
            ok = fx.run_v5("T", days, sig, sigma, p, **kw)
        except ValueError:
            continue
        _same(exec_e.run("T", days, sig, sigma, p, **kw), ok)                           # без мутации — равны
        with monkeypatch.context() as m:
            mut(m)
            try:
                bad = exec_e.run("T", days, sig, sigma, p, **kw)
            except ValueError:
                chislo += 1
                continue
        try:
            _same(bad, ok)
        except AssertionError:
            chislo += 1
    assert chislo >= 5, f"мутация {name} не поймана: расхождений {chislo}"
