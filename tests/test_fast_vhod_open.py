"""Ф-vhod_open: ядро `fast_exec` берёт точки S3/S4 с `vhod_open` и осями стоп-заявки (#666/#667/#671: ed_s / ed_t / chand,
b0 / sp / step_s / pir, reentry_den на стоп-заявке; прежде — эталон через `ne_znaet`). Эталон — `exec_e.run` (RunStop), допуск 0.
1) 1200 синтетических путей стоп-заявки с этими осями и vhod_open (побайтно, `sverit`); оси реально работают (счётчики);
2) `ne_znaet`: vhod_open на стоп-заявке — ядро знает, на лимитном пути — по-прежнему нет;
3) мутации эталона расходятся с ядром: гэп по уровню вместо open (№ 4), reentry в баре выхода, пара дня d вместо d−1.
Сверка идёт на уровне `exec_e.run` с t_vyhod = close (open2 — ядро не знает, лист Ф-open2).
FASTBT_MAIN=<копия> python3 -m pytest tests/test_fast_vhod_open.py -q"""
import dataclasses
import os
import time

import numpy as np
import pytest

from fastbt import exec_e, exec_e_stop, fast_exec as fx, progon_sobol as ps
from fastbt.exec_e_stop import RunStop
from fastbt.exec_e_types import Params
from fastbt.exec_e_zayavki import INTRA
from test_fast_etap1_smoke import TICK
from test_fast_v5_s34 import ch, lenta, sverit

N = 1200


def tochka(i: int) -> Params:
    """Стоп-заявка S3/S4 с vhod_open и осями Л5 / V6-11b / V6-11c; недопустимые сочетания эталон отклоняет — `sverit` сверяет и отказ."""
    rng = np.random.default_rng(20261030 + i)
    es, et = ch(rng, "sigma", "sigma", "atr", "pct"), ch(rng, "sigma", "sigma", "atr", "pct")
    b = ch(rng, 0.01, 0.02) if es == "pct" else ch(rng, 1.0, 2.0, 3.0)
    a = ch(rng, None, 0.03) if et == "pct" else ch(rng, None, 1.0, 2.0)
    kw = dict(ed_s=es, ed_t=et, chand=ch(rng, None, None, 1.5, 3.0),
              b0=ch(rng, None, None, 1.0), sp=ch(rng, None, None, (1.0, 0.5)), step_s=ch(rng, None, None, 1.0),
              pir=ch(rng, 0.0, 0.0, 0.5), reentry_den=rng.random() < 0.7)
    return Params(vhod_tip="stop_zayavka", exit_signal=False, stop_touch=True, vhod_open=True, k_e=ch(rng, 0.0, 0.0, -0.5, 0.25),
                  b=b, a=a, tau=ch(rng, None, 3, 5), s=ch(rng, "fiks", "ezhedn"), comm=0.0004, side=ch(rng, 1, -1),
                  tick=ch(rng, None, 0.0, TICK), atr_m=ch(rng, None, None, 2.0), **kw)


def sluchai(i: int, p=None) -> tuple:
    return (p or tochka(i)), lenta(30_000 + i)


def test_sintetika_1200_putej_yadro_ravno_etalonu():
    plokho, cnt = [], dict(valid=0, povtor=0, kanal=0, bu=0, pir=0, trail=0)
    for i in range(N):
        p, (days, sig, sigma, close, div, urov) = sluchai(i)
        assert fx.ne_znaet(p) == [], p
        if not sverit(days, sig, sigma, p, close, div, urov):
            plokho.append(i)
            continue
        try:
            r = exec_e.run("T", days, sig, sigma, p, close=close, div=div, urovni=urov)
        except ValueError:
            continue
        rs = [t.reason for t in r.trades]
        cnt["valid"] += 1
        cnt["kanal"] += "kanal" in rs
        cnt["bu"] += "bu" in rs
        cnt["pir"] += "pir" in rs
        cnt["trail"] += "trail" in rs
        cnt["povtor"] += any(a.exit_day == b.entry_day and a.exit_bar is not None and b.entry_bar > a.exit_bar
                             for a, b in zip(r.trades, r.trades[1:]))
    assert not plokho, plokho[:10]
    assert cnt["valid"] >= 300 and min(cnt["povtor"], cnt["kanal"], cnt["bu"], cnt["trail"]) >= 10, cnt   # оси реально работают


def test_ne_znaet_vhod_open_tolko_na_stop_zayavke():
    p = Params(vhod_tip="stop_zayavka", exit_signal=False, stop_touch=True, vhod_open=True, b=2.0)
    assert fx.ne_znaet(p) == []
    assert fx.ne_znaet(dataclasses.replace(p, bar_plus=1)) == ["bar_plus"]                # parab в ядре с #703, zhizn — с Ф-v6-osi; bar_plus ядро не знает
    assert "vhod_open" in fx.ne_znaet(Params(vhod_open=True, check="bary"))               # рынок по open 10:00 — лимитный путь, эталон


def _rashozhdenij(mut, n=N) -> int:
    """Число путей с reentry_den (или любых для гэпа), на которых мутированный эталон ≠ ядро."""
    bad = 0
    for i in range(n):
        p, (days, sig, sigma, close, div, urov) = sluchai(i)
        if mut.get("only_re") and not p.reentry_den:
            continue
        kw = dict(close=close, div=div, urovni=urov)
        try:
            ya = fx.run_v5("T", days, sig, sigma, p, **kw)
        except (ValueError, NotImplementedError):
            continue
        with mut["ctx"]():
            try:
                ref = exec_e.run("T", days, sig, sigma, p, **kw)
            except ValueError:
                continue
        try:
            from test_fast_etap1_smoke import _same
            _same(ref, ya)
        except AssertionError:
            bad += 1
    return bad


class _Patch:
    def __init__(s, obj, name, val):
        s.o, s.n, s.v = obj, name, val

    def __enter__(s):
        s.old = getattr(s.o, s.n)
        setattr(s.o, s.n, s.v)

    def __exit__(s, *a):
        setattr(s.o, s.n, s.old)


def _reenter_mut(kind):
    """Копия RunStop._reenter с одной подложенной ошибкой: bar — вход с бара выхода; den — пара дня d (не d−1)."""
    def f(s, i, j, d):
        x = s.last_x
        if (not s.p.reentry_den or s.tr is not None or s.order is not None or j == 0 or x is None or x.exit_day != i
                or x.reason not in INTRA or s.done_re == i or x.exit_bar + 1 >= len(d.c)):
            return
        s.done_re = i
        ip = i if kind == "den" else s.ix[j - 1]
        pr = s._pair(ip) if np.isfinite(s.sigma[ip]) else None
        if pr is None:
            return
        b0 = x.exit_bar if kind == "bar" else x.exit_bar + 1
        r = exec_e_stop.stop_vhod(d.o[b0:], d.h[b0:], d.l[b0:], *pr)
        if r is not None:
            s.n_orders, s.n_fills = s.n_orders + 1, s.n_fills + 1
            s._open(i, j, d, (r[0], r[1] + b0, r[2]), ip)
    return f


def test_mutatsiya_reentry_v_bare_vyhoda_lovitsya():
    assert _rashozhdenij(dict(only_re=True, ctx=lambda: _Patch(RunStop, "_reenter", _reenter_mut("bar")))) > 0


def test_mutatsiya_para_dnya_d_vmesto_d_minus_1_lovitsya():
    assert _rashozhdenij(dict(only_re=True, ctx=lambda: _Patch(RunStop, "_reenter", _reenter_mut("den")))) > 0


def test_mutatsiya_gep_po_urovnyu_vmesto_open_lovitsya():
    """№ 4: вход стоп-заявкой при гэпе через уровень исполнен по уровню, а не по open."""
    orig = exec_e_stop.stop_vhod

    def gep_po_urovnyu(o, h, l, up, dn):
        r = orig(o, h, l, up, dn)
        return None if r is None else (r[0], r[1], up if r[0] > 0 else dn)
    assert _rashozhdenij(dict(ctx=lambda: _Patch(exec_e_stop, "stop_vhod", gep_po_urovnyu))) > 0


@pytest.mark.skipif(not os.environ.get("FASTBT_V5_N"), reason="малая проверка на данных train4 (≤ 2021): задать FASTBT_V5_N")
@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_real_bary_yadro_ravno_etalonu_na_urovne_run(sid, monkeypatch, capsys):
    """Каждый вызов exec_e.run точкой куба v5 (t_vyhod = open2, ядро его не знает) сверяется как t_vyhod = close:
    ядро `run_v5` против эталона, побайтно; parab (П3b) ядру не отдаётся и в счёт не входит."""
    from test_fast_etap1_smoke import _same
    from test_fast_v5_s34_pokryto import RZ
    cnt = dict(sverka=0, parab=0, t_etalon=0.0, t_yadro=0.0)       # время — суммарное по вызовам run (замер, потолок: одна машина)
    orig = exec_e.run

    def sverka(ticker, days, sig, sigma, p, **kw):
        q = dataclasses.replace(p, t_vyhod="close")
        if fx.ne_znaet(q):
            cnt["parab"] += fx.ne_znaet(q) == ["parab"]
            assert fx.ne_znaet(q) == ["parab"], fx.ne_znaet(q)
        else:
            try:
                t0 = time.perf_counter()
                ref = orig(ticker, days, sig, sigma, q, **kw)
                cnt["t_etalon"] += time.perf_counter() - t0
            except ValueError as e:
                with pytest.raises(ValueError, match=str(e)[:30].replace("(", r"\(")):
                    fx.run_v5(ticker, days, sig, sigma, q, **kw)
            else:
                t0 = time.perf_counter()
                ya = fx.run_v5(ticker, days, sig, sigma, q, **kw)
                cnt["t_yadro"] += time.perf_counter() - t0
                _same(ref, ya)
                cnt["sverka"] += 1
        return orig(ticker, days, sig, sigma, p, **kw)
    monkeypatch.setattr(exec_e, "run", sverka)
    sr = ps.fabrika_real(sid, RZ)
    for i, pt in ps.tochki(ps.load_kub(sid, RZ), 0, int(os.environ["FASTBT_V5_N"]), RZ):
        ps.schitat_tochku(sr, sid, i, pt, "t", False, False, rz=RZ)
    assert cnt["sverka"] > 100, cnt
    with capsys.disabled():
        print(f"\n{sid}: {cnt['sverka']} вызовов run, эталон {cnt['t_etalon']:.1f} с, ядро {cnt['t_yadro']:.1f} с; parab (эталон): {cnt['parab']}")
