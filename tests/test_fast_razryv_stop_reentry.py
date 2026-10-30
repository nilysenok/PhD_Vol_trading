"""PR #834 (тестировщик): пробел покрытия — точки со стоп-заявкой И `reentry_den` на лентах с разрывами календаря 1…30 дней.
`tests/test_fast_razryv.py` берёт `reentry_den` только на лимите (tochka: i ≥ N_SZ), на стоп-заявке его нет, поэтому
`RunStop._reenter` без проверки `j in s.stale` проходил 1200 путей. Здесь: 1) 600 путей стоп-заявка + reentry_den, разрывы 1…30
(в т. ч. 5 и 6, два разрыва подряд), `fx.run_v5` и `fast_etap1.run` = `exec_e.run` побайтно; 2) то же для лимита + reentry_den;
3) граница `>` / `>=` (разрыв ровно 5: повторный вход есть в обоих путях, 6 — нет); 4) мутации: правка снята, `>=` в одном / обоих путях.
FASTBT_MAIN=<копия> NUMBA_NUM_THREADS=1 python3 -m pytest tests/test_fast_razryv_stop_reentry.py -q"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, fast_etap1, fast_exec as fx
from fastbt.exec_e_stop import RunStop
from test_fast_etap1_smoke import _same
from test_fast_razryv import sverit
from test_fast_v5_s34 import lenta, tochka
from test_fast_v6_razryv_kalendarya import P, oba, vhody_dnya

N = 600
GAPS = tuple(range(1, 31))


def daty(seed: int, n: int) -> pd.DatetimeIndex:
    """Разрыв 1…30 дней с вероятностью 0,25 на шаг; seed % 3 == 0 — два разрыва подряд; разрывы 5 и 6 берутся чаще."""
    rng = np.random.default_rng(seed)
    gap = np.where(rng.random(n) < 0.3, rng.choice((5, 6), n), rng.choice(GAPS, n))
    step = np.where(rng.random(n) < 0.25, gap, 1)
    if seed % 3 == 0:
        k = int(rng.integers(2, n - 2))
        step[k], step[k + 1] = int(rng.choice((5, 6, 7))), int(rng.choice((5, 6, 9, 30)))
    return pd.DatetimeIndex(pd.Timestamp("2019-03-01") + pd.to_timedelta(np.cumsum(step), unit="D"))


def tochka_re(i: int, stop: bool):
    """Точка S3/S4 (стоп-заявка) или лимитная — в обоих случаях с `reentry_den`."""
    p = tochka(i if stop else 600 + i)
    return dataclasses.replace(p, reentry_den=True)


def progon(stop: bool = True, n: int = N):
    plokho, cnt = [], dict(valid=0, snyata=0, povtor=0, cherez=0)
    for i in range(n):
        p, (days, sig, sigma, close, div, urov) = tochka_re(i, stop), lenta(90_000 + i)
        assert fx.ne_znaet(p) == [], p
        dates = daty(i, len(days))
        for f in (fx.run_v5, fast_etap1.run):
            if not sverit(days, sig, sigma, p, close, div, urov, dates, f):
                plokho.append(i)
                break
        else:
            kw = dict(close=close, div=div, urovni=urov if stop else None)
            try:
                a = exec_e.run("T", days, sig, sigma, p, dates=dates, **kw)
                b = exec_e.run("T", days, sig, sigma, dataclasses.replace(p, razryv=None), dates=dates, **kw)
            except ValueError:
                continue
            cnt["valid"] += 1
            cnt["snyata"] += [dataclasses.astuple(t) for t in a.trades] != [dataclasses.astuple(t) for t in b.trades]
            cnt["povtor"] += any(x.exit_day == y.entry_day for x, y in zip(a.trades, a.trades[1:]))
    return plokho, cnt


def lovitsya(**kw) -> bool:
    try:
        return bool(progon(**kw)[0])
    except (ValueError, pytest.fail.Exception):
        return True


def test_stop_zayavka_s_reentry_den_600_putej_razryvy_1_30_pobajtno():
    plokho, cnt = progon(True)
    assert plokho == [], plokho[:20]
    assert cnt["valid"] > 400 and cnt["snyata"] > 30 and cnt["povtor"] > 30, cnt     # правило бьёт; повторные входы есть


def test_limit_s_reentry_den_600_putej_razryvy_1_30_pobajtno():
    """Лимит защищён: Run.step проверяет stale до своего reenter."""
    plokho, cnt = progon(False)
    assert plokho == [], plokho[:20]
    assert cnt["valid"] > 400 and cnt["povtor"] > 30, cnt


@pytest.mark.parametrize("gap", [5, 6])
def test_granica_razryva_pyat_vhod_est_shest_net_v_oboih_putyah(gap):
    e, b = oba(gap)
    _same(e, b)
    assert bool(vhody_dnya(e, 2)) is (gap == 5) and bool(vhody_dnya(b, 2)) is (gap == 5)


def test_dva_razryva_podryad_pobajtno():
    """Позиция через разрыв 4 (жива), затем стоп по гэпу и разрыв 7 перед днём повторного входа: пары вечера 25.02 нет."""
    for g in (4, 5, 6, 7, 12):
        e, b = oba(g)
        _same(e, b)
    from test_fast_v6_razryv_kalendarya import sluchaj
    days, sig, sigma, urovni, dates = sluchaj(3)
    dates = pd.DatetimeIndex([dates[0], dates[1], dates[2] + pd.Timedelta(days=9), dates[3] + pd.Timedelta(days=9)])   # 2 → 12 дней
    _same(exec_e.run("T", days, sig, sigma, P, urovni=urovni, dates=dates), fast_etap1.run("T", days, sig, sigma, P, urovni=urovni, dates=dates))


def test_mut_pravka_snyata(monkeypatch):
    orig = RunStop._reenter

    def bez_stale(s, i, j, d):
        was = s.stale
        s.stale = frozenset()
        try:
            return orig(s, i, j, d)
        finally:
            s.stale = was
    monkeypatch.setattr(RunStop, "_reenter", bez_stale)
    assert lovitsya(stop=True)
    assert vhody_dnya(oba(6)[0], 2) != []


def _ge_stale_steps(dates, ix, razryv):
    d = pd.DatetimeIndex(dates)
    return frozenset(j for j in range(1, len(ix)) if (d[ix[j]] - d[ix[j - 1]]).days >= razryv)


def _ge_mask(dates, pusto, razryv):
    ix = np.flatnonzero(~pusto)
    if razryv is None or dates is None or len(ix) < 2:
        return np.empty(0, np.bool_)
    d = pd.DatetimeIndex(dates)[ix]
    return np.r_[False, np.asarray((d[1:] - d[:-1]).days >= razryv)].astype(np.bool_)


def test_mut_ge_v_oboih_putyah_ravny_no_granica_lovitsya(monkeypatch):
    """`>=` в обоих путях: пути друг другу равны, но граница «ровно 5 — вход есть» ломается — ловит тест границы."""
    monkeypatch.setattr(exec_e, "stale_steps", _ge_stale_steps)
    monkeypatch.setattr(fx, "stale_mask", _ge_mask)
    assert vhody_dnya(oba(5)[0], 2) == []


def test_mut_ge_tolko_v_yadre_lovitsya(monkeypatch):
    monkeypatch.setattr(fx, "stale_mask", _ge_mask)
    assert lovitsya(stop=True)


def test_mut_ge_tolko_v_etalone_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e, "stale_steps", _ge_stale_steps)
    assert lovitsya(stop=True)
