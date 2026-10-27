"""Э1-а (Ж-031): PBO по CSCV и DSR(N). Допуск PBO на случайных данных объявлен заранее: |среднее − 0,5| ≤ 0,08
по 30 независимым матрицам 30 точек × 400 дней (одна матрица шумит сильнее: 70 комбинаций сильно зависимы)."""
import numpy as np
import pytest

from fastbt import metriki_e as me
from fastbt import sweep_stat as ss

TOL = 0.08


def test_pbo_random_is_half():
    p = [ss.pbo_cscv(np.random.default_rng(s).normal(0, 1, (400, 30))) for s in range(30)]
    assert abs(np.mean(p) - 0.5) <= TOL


def test_pbo_detects_real_skill():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 1, (400, 30))
    m[:, 7] += 0.4                                     # одна точка с настоящим преимуществом
    assert ss.pbo_cscv(m) < 0.1


def test_pbo_exact_extremes():
    """Детерминированно: у A плюс в блоках 0–3, минус в 4–7; у B наоборот. IS-лучшая всегда проигрывает в OOS -> PBO = 1.
    Если A всегда лучше B — PBO = 0. Мутация argmax→argmin или перепутанный ранг меняет ответ."""
    up, dn = np.tile([1.5, 0.5], 4), np.tile([-0.5, -1.5], 4)       # блок из 2 строк; 8 блоков = 16 строк
    a = np.concatenate([up, dn]),
    A = np.concatenate([np.tile([1.5, 0.5], 8), np.tile([-0.5, -1.5], 8)])[:, None]   # 16 + 16 строк -> по 4 блока
    A = np.concatenate([np.tile([1.5, 0.5], 8), np.tile([-0.5, -1.5], 8)])[:, None]
    assert ss.pbo_cscv(np.hstack([A, A[::-1]])) == 1.0
    assert ss.pbo_cscv(np.hstack([np.tile([1.5, 0.5], 16)[:, None], np.tile([-0.5, -1.5], 16)[:, None]])) == 0.0


def test_pbo_degenerate_is_nan():
    assert np.isnan(ss.pbo_cscv(np.zeros((400, 1))))
    assert np.isnan(ss.pbo_cscv(np.zeros((10, 5))))


def test_dsr_grows_when_n_is_one(monkeypatch):
    rng = np.random.default_rng(3)
    r = rng.normal(0.0008, 0.01, 500)
    srs = [me.sr_dnevnoy(rng.normal(0, 0.01, 500)) for _ in range(39)] + [me.sr_dnevnoy(r)]
    one, many = ss.dsr_n(r, [me.sr_dnevnoy(r)]), ss.dsr_n(r, srs)
    assert one > many
    assert ss.dsr_n(r, srs, mult=2) < many             # повторный взгляд: N удвоен -> DSR ниже
    monkeypatch.setattr(me, "sr_zvezda", lambda n, v: 0.0)
    assert ss.dsr_n(r, srs) == pytest.approx(one)      # мутация: SR* не растёт с N -> разницы нет, проверка выше упала бы


def test_objective_mean_minus_lambda_sigma():
    assert ss.objective({2018: 1.0, 2019: 0.0}) == pytest.approx(0.5 - 0.5 * np.std([1.0, 0.0], ddof=1))
    assert ss.objective({2018: 0.7}) == 0.7
    assert np.isnan(ss.objective({2018: float("nan"), 2019: 1.0}))
