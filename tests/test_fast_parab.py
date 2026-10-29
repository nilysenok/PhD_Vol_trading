"""Ф-parab: параболический трейлинг тренда (П3b, Params.parab = (ps, pm)) в быстром ядре fast_exec / fast_exec_parab.
Эталон — `exec_e.run` (Run, fastbt/exec_e_parab.py); `fast_exec.run_v5` обязан совпасть ПОБАЙТНО (допуск 0, proverka.md п. 3).
1) 1200 синтетических путей (лонг/шорт, лимит и стоп-заявка, тейк, тик, дивиденды, close Н6, дни pusto);
2) сценарии: гэп через параболический стоп, тейк и стоп в одном баре, стоп бара u по состоянию u−1;
3) мутации эталона (mutations № 3, № 4 и заглядывание) — ядро с ними расходится, сверка обязана падать;
4) покрытие по флагу YADRO_PARAB_GOTOV; 5) 200 точек S3 и S4 куба v5 на уровне exec_e.run (t_vyhod = close) — при FASTBT_V5_N=200,
ядро берёт их, когда ему известен vhod_open (#699); без него все вызовы S3/S4 — ne_znaet, тест пропущен.
NUMBA_CACHE_DIR=<свежий> FASTBT_BARS1M=<основная копия>/data/regen/D25/bars1m FASTBT_MAIN=$PWD FASTBT_V5_N=200 python3 -m pytest tests/test_fast_parab.py -q"""
import dataclasses
import os

import numpy as np
import pytest

from fastbt import exec_e, exec_e_run, fast_etap1 as fe, fast_exec as fx, progon_sobol as ps
from fastbt.exec_e_types import Day, Params
from fastbt.vyhod_parabolic import NIVELI_V4
from test_fast_etap1_smoke import TICK, _same
from test_fast_v5_s34 import ch, lenta

N = 1200
SG = 0.02


def tochka(i: int) -> Params:
    """Параболический трейлинг с b и bary: лимитный вход (k, T, reentry_den, f) либо стоп-заявка (каждый 4-й); ps ≤ pm."""
    rng = np.random.default_rng(20261031 + i)
    ps_, pm_ = ch(rng, *NIVELI_V4, (0.02, 0.2), (0.05, 0.05), (0.01, 0.3))
    kw = dict(vhod_tip="stop_zayavka", exit_signal=False, stop_touch=True, k_e=ch(rng, 0.0, 0.0, -0.5, 0.25)) if i % 4 == 3 else dict(
        k=ch(rng, None, 0.25, 0.5, 1.0), f=ch(rng, "propusk", "zakrytie_d1", "povtor"), exit_signal=ch(rng, True, False),
        stop_touch=ch(rng, True, False), reentry=ch(rng, True, False), T=ch(rng, 1, 2), reentry_den=bool(i % 2),
        vyhod_lim=ch(rng, False, False, True))
    if "k" in kw and kw["k"] is None:
        kw.update(T=1, reentry_den=False, vyhod_lim=False)
    return Params(b=ch(rng, 1.0, 2.0, 3.0), a=ch(rng, None, 1.0, 2.0), tau=ch(rng, None, 3, 5), comm=0.0004, side=ch(rng, 1, -1),
                  tick=ch(rng, None, 0.0, TICK), parab=(ps_, pm_), **kw)


def sverit(days, sig, sigma, p, close=None, div=None, urov=None) -> bool:
    kw = dict(close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
    try:
        a = exec_e.run("T", days, sig, sigma, p, **kw)
    except ValueError as e:
        try:
            fx.run_v5("T", days, sig, sigma, p, **kw)
        except ValueError as e2:
            return str(e2) == str(e)
        return False                                    # эталон отказал, ядро — нет
    try:
        _same(a, fx.run_v5("T", days, sig, sigma, p, **kw))
    except AssertionError:
        return False
    return True


def sintetika(n=N, start=0):
    return [(tochka(i), lenta(50_000 + i)) for i in range(start, start + n)]


def test_sintetika_yadro_ravno_etalonu_na_1200_putyah():
    plokho, trail = [], 0
    for i, (p, (days, sig, sigma, close, div, urov)) in enumerate(sintetika()):
        assert not fx.ne_znaet(p), p
        if not sverit(days, sig, sigma, p, close, div, urov):
            plokho.append(i)
            continue
        r = exec_e.run("T", days, sig, sigma, p, close=close, div=div, urovni=urov if p.vhod_tip == "stop_zayavka" else None)
        trail += sum(t.reason == "trail" for t in r.trades)
    assert not plokho, plokho[:10]
    assert trail >= 200, trail                          # параболический стоп реально срабатывает, а не «ничего не делает»


def test_open2_sintetika_400_putey():
    """parab при t_vyhod = open2 (Ф-open2: скан дня до бара 10:00, выход по open): лента с временем баров из test_fast_open2."""
    from test_fast_open2 import lenta as lenta_o2
    plokho, trail = [], 0
    for i in range(400):
        p = dataclasses.replace(tochka(4 * i), t_vyhod="open2", vyhod_lim=False)
        days, sig, sigma, close, div = lenta_o2(60_000 + i)
        if not sverit(days, sig, sigma, p, close, div):
            plokho.append(i)
            continue
        trail += sum(t.reason == "trail" for t in exec_e.run("T", days, sig, sigma, p, close=close, div=div).trades)
    assert not plokho, plokho[:10]
    assert trail >= 30, trail


def _dni(o, h, lo, c):
    return [Day(np.array([a], float), np.array([b], float), np.array([x], float), np.array([y], float), False) for a, b, x, y in zip(o, h, lo, c)]


def stsenarij(o, h, lo, c, sig, side=1, a=None):
    """Вход по close дня 0 (k = None), σ = 0,02, b = 2: начальный стоп 96 (лонг); ядро и эталон — побайтно, сделка — первая."""
    days, sig, sigma = _dni(o, h, lo, c), np.array(sig, float), np.full(len(c), SG)
    p = Params(b=2.0, a=a, parab=(0.01, 0.10), stop_touch=True, side=side)
    assert sverit(days, sig, sigma, p)
    return exec_e.run("T", days, sig, sigma, p).trades[0]


def test_gep_cherez_parabolicheskiy_stop_po_otkrytiyu():
    t = stsenarij([100, 100, 90], [100, 101, 91], [100, 99, 89], [100, 100, 90], [1, 1, 1])
    assert (t.exit_day, t.reason, t.exit_px) == (2, "trail", 90.0)                  # лонг: гэп вниз — по open, не по уровню 96,1392
    t = stsenarij([100, 100, 110], [100, 101, 111], [100, 99, 109], [100, 100, 110], [1, 1, 1], side=-1)
    assert (t.exit_day, t.reason, t.exit_px) == (2, "trail", 110.0)                 # шорт зеркально


def test_tejk_i_stop_v_odnom_bare_stop_pervyy():
    t = stsenarij([100, 100], [100, 103], [100, 95], [100, 97], [1, 1], a=1.0)         # T = 102 и S = 96,04 в одном баре
    assert (t.reason, round(t.exit_px, 4)) == ("trail", 96.04)
    t = stsenarij([100, 100], [100, 105], [100, 97], [100, 100], [1, 1], side=-1, a=1.0)   # шорт: S = 103,96, T = 98
    assert (t.reason, round(t.exit_px, 4)) == ("trail", 103.96)


def test_stop_bara_po_sostoyaniyu_predydushchego():
    """День 1: стоп 96 + 0,01·(100 − 96) = 96,04 (ext бара входа 100, step 0,01) — бар u=1 с low 96,03 пробивает; low 96,05 — нет."""
    assert stsenarij([100, 100], [100, 100], [100, 96.03], [100, 97], [1, 1]).reason == "trail"
    t = stsenarij([100, 100, 100], [100, 100, 100], [100, 96.05, 96.05], [100, 97, 97], [1, 1, 1])
    assert t.exit_day is None or t.exit_day == 2                                   # день 1 цел; на день 2 стоп вырос (≥ 96,0792)


def test_krajnie_lenty():
    e, one = np.empty(0), Day(np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([100.0]), False)
    pust = Day(e, e, e, e, True)
    for days in ([], [one], [one, one], [pust, pust, pust], [one, pust, one, one]):
        n = len(days)
        for sigma in (np.full(n, 0.02), np.zeros(n), np.full(n, float("nan"))):
            for i in range(8):
                assert sverit(days, np.ones(n), sigma, tochka(i)), (n, i)


# --- мутации эталона: ядро с ошибкой = ядро против изменённого эталона; сверка обязана падать ---
def _est_rashozhdenie(mp, nabor=400) -> bool:
    return any(not sverit(days, sig, sigma, p, close, div, urov) for p, (days, sig, sigma, close, div, urov) in sintetika(nabor))


def test_mutatsiya_af_ne_rastyot_pri_novom_ekstremume_lovitsya(monkeypatch):
    def pb_after(sd, st, p, hi, lo):
        st[0] = exec_e_run.pb_level(sd, st)                       # ext обновляется, шаг AF — нет
        if (hi > st[1]) if sd > 0 else (lo < st[1]):
            st[1] = hi if sd > 0 else lo
    monkeypatch.setattr(exec_e_run, "pb_after", pb_after)
    assert _est_rashozhdenie(monkeypatch)


def test_mutatsiya_stop_bara_po_baru_u_zaglyadyvanie_lovitsya(monkeypatch):
    orig = exec_e_run.pb_after

    def pb_after(sd, st, p, hi, lo):                              # сначала экстремум и шаг бара u, потом стоп: стоп следующего бара по u
        st2 = list(st)
        orig(sd, st2, p, hi, lo)
        st[:] = [exec_e_run.pb_level(sd, st2), st2[1], st2[2]]
    monkeypatch.setattr(exec_e_run, "pb_after", pb_after)
    assert _est_rashozhdenie(monkeypatch)


def test_mutatsiya_gep_po_urovnyu_lovitsya(monkeypatch):
    orig = exec_e_run.stop_hit
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, lo, S, touch: (lambda r: (r[0], S if r[0] else r[1]))(orig(sd, o, h, lo, S, touch)))
    assert _est_rashozhdenie(monkeypatch)


# --- покрытие и точки куба ---
def test_pokryto_po_flagu(monkeypatch):
    pt = {"stop_nach": "sigma:2.0", "trail": "parab:0.02:0.15"}
    assert fe.os_parab(pt) and not fe.os_parab({"stop_nach": "sigma:2.0"}) and not fe.os_parab({"trail": "parab:0.02:0.15"})
    assert fe.pokryto(pt)
    monkeypatch.setattr(fx, "YADRO_PARAB_GOTOV", False)
    assert not fe.pokryto(pt) and fe.pokryto({"stop_nach": "sigma:2.0"})
    assert fx.ne_znaet(tochka(0)) == ["parab"]                    # без флага ядро ось не знает — эталон
    with pytest.raises(NotImplementedError):
        days, sig, sigma, close, div, _ = lenta(50_000)
        fx.run_v5("T", days, sig, sigma, tochka(0), close=close, div=div)


def test_parab_s_drugim_trailingom_otkaz_te_zhe():
    days, sig, sigma, close, div, urov = lenta(51_000)
    for kw in (dict(k_tr=1.5), dict(be=True), dict(atr_m=2.0), dict(b=None), dict(parab=(0.1, 0.05)), dict(check="zakrytie")):
        assert sverit(days, sig, sigma, dataclasses.replace(tochka(0), **kw), close, div, urov), kw


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="малая проверка на данных train4: задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", ("S3", "S4"))
def test_kub_v5_parab_pobajtno(sid, monkeypatch):
    """Каждый вызов run точки куба (бары ≤ 2021, t_vyhod подменён на close): эталон = быстрый слой побайтно; parab-вызовов ядра > 0."""
    from test_fast_v5_osi3_pokryto import RZ, Shpion
    probe = dataclasses.replace(tochka(3), vhod_open=True)
    if fx.ne_znaet(probe):
        pytest.skip(f"ядро не знает {fx.ne_znaet(probe)}: все вызовы S3/S4 идут эталоном (ждёт #699)")
    calls, orig = [], fx.run_v5
    monkeypatch.setattr(fx, "run_v5", lambda tk, d, s, sg, p, **k: calls.append(p.parab is not None) or orig(tk, d, s, sg, p, **k))
    sr, n = ps.fabrika_real(sid, RZ), int(os.environ["FASTBT_V5_N"])
    pts = [x for x in ps.tochki(ps.load_kub(sid, RZ), 0, 20 * n, RZ)][:n]
    sh = Shpion()
    monkeypatch.setattr(exec_e, "run", sh)
    for i, pt in pts:
        ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
    print(f"\n{sid}: точек {len(pts)}, run сверено {sh.n}, вызовов ядра {len(calls)}, из них parab {sum(calls)}")
    assert sh.n and sum(calls) > 0, "ядро не вызвано на parab"
