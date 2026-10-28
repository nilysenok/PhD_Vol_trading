"""Ж-035: подложенные ошибки (monkeypatch) в модуле `fastbt/sweep_drivers.py` и в сквозной цепочке. Модуль не правится.
Детектор на правильном коде — True (тесты в `test_sweep_modul.py`), на мутации — False."""
import math
import types

import pytest

from fastbt import sweep_drivers as sd
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_engine import Tiny
from test_sweep_modul import (GRID, BASE, make_eval, ok_commission_rows, ok_dsr_n1_grows, ok_morris_mu_star,
                              ok_objective_penalty, ok_param_moves_result, ok_plateau_not_peak, ok_validation_not_in_train,
                              pbo_random_mean, pbo_skill)


def test_mutm_01_param_ignoriruetsya_ocenshchikom(monkeypatch):
    assert ok_param_moves_result()
    orig = SynthEvaluator.signals
    monkeypatch.setattr(SynthEvaluator, "signals", lambda self, sp: orig(self, {"a": 5}))
    orig_pnl = SynthEvaluator.pnl
    monkeypatch.setattr(SynthEvaluator, "pnl", lambda self, s, p, c, e: orig_pnl(self, s, {**p, "w": 1.0}, c, e))
    assert not ok_param_moves_result()


def test_mutm_02_2020_v_train_cherez_period(monkeypatch):
    assert ok_validation_not_in_train()
    import test_sweep_modul as tm
    real = tm.make_eval
    monkeypatch.setattr(tm, "make_eval", lambda ev, years_of=None, comm=0.0004: real(ev, {"train": (2018, 2019, 2020)}, comm))
    assert not ok_validation_not_in_train()


def test_mutm_03_komissiya_na_vkhode_tolko(monkeypatch):
    assert ok_commission_rows()
    orig = Tiny.pnl
    monkeypatch.setattr(Tiny, "pnl", lambda self, s, p, c, e: orig(self, s, p, c / 2, e))
    assert not ok_commission_rows()


@pytest.mark.v11
def test_mutm_04_pbo_znak_logita_iskazhen(monkeypatch):
    assert abs(pbo_random_mean() - 0.5) <= 0.10
    fake = types.SimpleNamespace(**{k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
    fake.log = lambda x: abs(math.log(x)) + 0.1          # логит > 0 всегда -> PBO ≡ 0
    monkeypatch.setattr(sd, "math", fake)
    assert abs(pbo_random_mean() - 0.5) > 0.10


@pytest.mark.v11
def test_mutm_05_pbo_luchshaya_naoborot(monkeypatch):
    assert pbo_skill() < 0.2
    orig = sd.pbo_cscv
    monkeypatch.setattr(sd, "pbo_cscv", lambda m, n_blocks=8, higher_is_better=True: orig(m, n_blocks, False))
    assert pbo_skill() >= 0.2


def test_mutm_06_N_v_dsr_podmenen_na_1(monkeypatch):
    assert ok_dsr_n1_grows()
    orig = sd.expected_max_sharpe
    monkeypatch.setattr(sd, "expected_max_sharpe", lambda n, v: orig(1, v))
    assert not ok_dsr_n1_grows()


def test_mutm_07_plato_zamenyeno_pikom(monkeypatch):
    assert ok_plateau_not_peak()
    monkeypatch.setattr(sd, "plateau_score", lambda rows, keys, grid, point, radius=1, col="objective":
                        (next(r[col] for r in rows if all(r[k] == point[k] for k in keys)), 1))
    assert not ok_plateau_not_peak()


def test_mutm_08_oat_stanovitsya_faktornym(monkeypatch):
    def n_oat():
        return sd.oat_sweep(make_eval(SynthEvaluator()), BASE, GRID, "train")["n_trials"]
    assert n_oat() == 7
    monkeypatch.setattr(sd, "oat_sweep", lambda ev, base, grid, period, **k: sd.factorial_sweep(ev, base, grid, period))
    assert n_oat() != 7


def test_mutm_09_lambda_nulevaya(monkeypatch):
    assert ok_objective_penalty()
    monkeypatch.setattr(sd.objective_stable, "__defaults__", (0.0,))
    assert not ok_objective_penalty()


@pytest.mark.v11
def test_mutm_10_morris_bez_modulya(monkeypatch):
    assert ok_morris_mu_star()
    monkeypatch.setattr(sd, "abs", lambda x: x, raising=False)       # mu* = средний знаковый эффект
    assert not ok_morris_mu_star()


@pytest.mark.v11
def test_mutm_11_optuna_minimiziruet(monkeypatch):
    optuna = pytest.importorskip("optuna")
    ev = make_eval(SynthEvaluator())

    def best_is_max(**kw):
        r = sd.optuna_search(ev, BASE, GRID, "train", n_trials=15, seed=7)
        return r["best_objective"] == max(x["objective"] for x in r["rows"])
    assert best_is_max()
    orig = optuna.create_study
    monkeypatch.setattr(optuna, "create_study", lambda **k: orig(**{**k, "direction": "minimize"}))
    assert not best_is_max()
