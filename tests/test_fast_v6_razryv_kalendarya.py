"""Расследование (без починки): быстрое ядро v6 ≠ эталонный путь на дне после разрыва календаря > razryv (wf, 25.02 → 24.03.2022).

Причина (воспроизведена здесь на синтетике): правило Д5-2 (Params.razryv, умолчание 5) снимает на шаге j после разрыва заявку,
поставленную вечером j−1, и запрещает повторный вход в этот день («повторный вход — вечером j−1 устарел», `Run.step`). Эталон на
стоп-заявках (`RunStop`, ось vhod_tip = stop_zayavka) вызывает свой `_reenter` прямо из `_scan` / `_open` — мимо проверки
`j not in s.stale` в `Run.step`; он ставит OCO-пару вечера j−1 (до разрыва) после внутридневного выхода в день j. Ядро
(`fast_exec.kernel_v5`) проверку делает (`pi[P_RED] and not stale[j]`). Верно ядро: пара вечера перед остановкой торгов устарела.

Случай: вход buy-stop 25.02, позиция держится через разрыв, в день после разрыва — стоп по open (гэп), потом бар выше up[25.02].
Разрыв ≤ 5 дней (выходные) — оба пути: повторный вход есть и совпадают. Разрыв 6 (как 30.12.2014 → 05.01.2015) и 27 дней —
эталон входит повторно, ядро нет: xfail(strict) до починки эталона. Тест `..._pochinka_...` подставляет проверку stale в
`RunStop._reenter` и показывает, что её одной хватает (это и есть причина, а не иное расхождение).
FASTBT_MAIN=<копия> python3 -m pytest tests/test_fast_v6_razryv_kalendarya.py -q"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, fast_etap1, fast_exec as fx
from fastbt.exec_e_stop import RunStop
from fastbt.exec_e_types import Day, Params
from test_fast_etap1_smoke import _same

NAN = np.nan
P = Params(vhod_tip="stop_zayavka", exit_signal=False, stop_touch=True, b=2.0, reentry_den=True, comm=0.0, razryv=5)


def _d(rows) -> Day:
    a = np.array(rows, float)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def sluchaj(gap: int) -> tuple:
    """4 торговых дня: 24.02, 25.02 (вход buy-stop 101), день через gap календарных дней (стоп по open 95, затем бар до 104 —
    пара вечера 25.02 up = 103 исполнилась бы), следующий день. σ = 2 % — стоп 101·(1 − 0,04) = 96,96."""
    days = [_d([(100, 100, 100, 100)] * 2),
            _d([(100, 101.5, 100, 101.2), (101.2, 101.4, 101.0, 101.2)]),
            _d([(95, 95.5, 94.5, 95), (100, 104, 99.5, 103.5), (103.5, 103.6, 103.4, 103.5)]),
            _d([(103.5, 103.6, 103.4, 103.5)] * 2)]
    d1 = pd.Timestamp("2022-02-25")
    dates = pd.DatetimeIndex([d1 - pd.Timedelta(days=1), d1, d1 + pd.Timedelta(days=gap), d1 + pd.Timedelta(days=gap + 1)])
    urovni = dict(up=np.array([101.0, 103.0, NAN, NAN]), dn=np.array([99.0, 97.0, NAN, NAN]),
                  xl=np.full(4, NAN), xs=np.full(4, NAN))
    return days, np.ones(4), np.full(4, 0.02), urovni, dates


def oba(gap: int, p: Params = P) -> tuple:
    days, sig, sigma, urovni, dates = sluchaj(gap)
    e = exec_e.run("T", days, sig, sigma, p, urovni=urovni, dates=dates)
    b = fast_etap1.run("T", days, sig, sigma, p, urovni=urovni, dates=dates)
    return e, b


def vhody_dnya(r, den: int) -> list:
    return [(t.entry_bar, t.entry_px, t.side) for t in r.trades if t.entry_day == den]


def test_sluchaj_idet_v_yadro_i_stop_po_gepu_v_den_posle_razryva():
    assert fx.ne_znaet(P) == []
    e, b = oba(27)
    for r in (e, b):
        t0 = r.trades[0]
        assert (t0.entry_day, t0.entry_px, t0.exit_day, t0.exit_bar, t0.exit_px, t0.reason) == (1, 101.0, 2, 0, 95.0, "stop")


@pytest.mark.parametrize("gap", [1, 3, 5])
def test_razryv_do_pyati_dnej_povtornyj_vhod_est_i_puti_ravny(gap):
    e, b = oba(gap)
    assert vhody_dnya(e, 2) == [(1, 103.0, 1)]          # пара вечера 25.02 жива: повторный вход по up = 103
    _same(e, b)


@pytest.mark.parametrize("gap", [6, 27])
def test_yadro_posle_razryva_povtornogo_vhoda_net(gap):
    _, b = oba(gap)
    assert vhody_dnya(b, 2) == []                        # Д5-2: пара вечера перед разрывом снята


@pytest.mark.parametrize("gap", [6, 27])
def test_etalon_posle_razryva_povtornogo_vhoda_net(gap):
    e, _ = oba(gap)
    assert vhody_dnya(e, 2) == []


@pytest.mark.parametrize("gap", [6, 27])
def test_posle_razryva_etalon_raven_yadru(gap):
    _same(*oba(gap))


@pytest.mark.parametrize("gap", [1, 3, 5, 6, 27])
def test_pochinka_proverkoj_stale_v_reenter_ustranyaet_raskhozhdenie(gap, monkeypatch):
    """Подстановка (только в тесте): `RunStop._reenter` не действует на шаге j ∈ s.stale. Расхождение исчезает на всех разрывах —
    причина одна. Мутация обратная: без подстановки при gap ≥ 6 пути расходятся (xfail выше)."""
    orig = RunStop._reenter

    def s_proverkoj(s, i, j, d):
        if j in s.stale:
            return None
        return orig(s, i, j, d)
    monkeypatch.setattr(RunStop, "_reenter", s_proverkoj)
    e, b = oba(gap)
    _same(e, b)
    assert bool(vhody_dnya(e, 2)) is (gap <= 5)


def test_bez_pravila_razryva_oba_puti_vhodyat():
    """razryv = None (правило выключено): пара не снимается, оба пути входят повторно и равны — расхождение только от Д5-2."""
    e, b = oba(27, dataclasses.replace(P, razryv=None))
    assert vhody_dnya(e, 2) == [(1, 103.0, 1)]
    _same(e, b)
