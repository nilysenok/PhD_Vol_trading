"""Э1-в п. 5: `gamma_hold` доходит до расчёта и укорачивает срок при σ входа выше медианной (уровневый движок, kind=2).
Нулевой размах в торнадо — не дефект параметра: при базовом max_hold=20 временной стоп не успевает сработать (сделки закрываются
по z_exit/стопу/тейку за ≤ 15 баров), см. отчёт. Тест фиксирует, что при тесном max_hold параметр работает; мутация — gamma в ядро не передаётся."""
import numpy as np

from fastbt import rm_sqrt_rv as rms
from fastbt.batch import trades_one
from test_rm_sqrt_rv import E, make_unit, rm_of, rv_step

MED = 0.02                     # σ̄ синтетики: σ на 120 барах 2018 → до 2018 истории нет, медиана подставляется вручную


def hold(gamma, max_hold=20):
    u = make_unit(False, {})
    rms.apply(u, rm_of("S1", max_hold=max_hold, gamma_hold=gamma), rv_step(j_big=E - 2, big=64e-4))
    u.med3 = np.array([MED])
    te, tx, _ = trades_one(u, 0, u.rmP[0], u.trend, end=u.n)
    return int(tx[0] - te[0])


def test_gamma_ukorachivaet_srok_pri_vysokoi_sigme():
    assert hold(None) == 20                       # без gamma — max_hold как есть
    assert 14 <= hold(1.0) <= 15                  # σ входа 0,0267 против σ̄ 0,02: 20·0,02/0,0267 ≈ 15
    assert hold(0.5) < 20 and hold(0.5) > hold(1.0)


def test_mut_gamma_ne_dohodit_do_yadra(monkeypatch):
    orig = rms._rows
    monkeypatch.setattr(rms, "_rows", lambda rm, trend, adx_on: orig({**rm, "gamma_hold": None}, trend, adx_on))
    assert hold(1.0) == 20                        # мутация: gamma обнулена — тест выше (14 ≤ hold ≤ 15) упал бы
