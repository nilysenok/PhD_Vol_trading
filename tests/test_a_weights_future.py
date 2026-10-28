"""Ансамбль A по годам (решение владельца 26.10, 10:30): веса года Y — только из данных ≤ Y−1.

Веса A года Y — состав топ-10 (сигнал × RM × исполнение) и равное голосование (fastbt.approach_a,
ApproachA.year / ensemble; chapter4_a и final_test берут A отсюда). Тест «будущее не влияет»:
цены и объём со ПЕРВОГО бара года Y и позже заменяются шумом (случайное блуждание, OHLC масштабируются
одним множителем, объём — случайно), доходности пересчитываются → топ-10 года Y обязан совпасть
полностью. Контроль чувствительности: та же подмена с бара на год раньше топ-10 года Y меняет.
Кэш сигналов на диск отключён (FASTBT_NO_CACHE=1): зашумлённые ряды не попадают в общий кэш.
python3 -m pytest tests/test_a_weights_future.py -q
"""
import dataclasses

import numpy as np
import pytest

from fastbt import dev
from fastbt.data import year_bounds

TICKER = "SBER"
SIDS = ("S1", "S3", "S5")            # контртренд, тренд, диапазон (S5 — V41)
YEARS = (2019, 2023)                 # разработка и тест


def _noisy(td, k, seed):
    rng = np.random.RandomState(seed)
    n = td.n
    f = np.ones(n)
    f[k:] = np.exp(np.cumsum(rng.normal(0.0, 0.03, n - k)))
    o, h, l, c = (np.asarray(a, float) * f for a in (td.open, td.high, td.low, td.close))
    v = np.asarray(td.volume, float).copy()
    v[k:] = v[k:] * np.exp(rng.normal(0.0, 0.5, n - k))
    lr = np.zeros(n)
    lr[:-1] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    return dataclasses.replace(td, open=o, high=h, low=l, close=c, volume=v, log_ret=lr)


@pytest.fixture(scope="module")
def base():
    import os
    os.environ["FASTBT_NO_CACHE"] = "1"
    try:
        from fastbt.data import load_ticker
        td = load_ticker(TICKER, True)
    except Exception as e:                                   # нет общих данных
        pytest.skip(f"нет данных: {e}")
    d, _ = dev.load_design(dev.ROOT / "fastbt" / "designs" / "final" / "main.yaml")
    cfg = dev.make_config(d, "test")
    yield td, cfg
    os.environ.pop("FASTBT_NO_CACHE", None)
    from fastbt import design as dsg
    dsg.apply(())                                            # сетки — к умолчанию (для test_core)


def _top10(td, sid, cfg, Y):
    from fastbt import design as dsg
    from fastbt.approach_a import ApproachA
    from fastbt.unit import make_unit
    dsg.apply(cfg.design)
    a = ApproachA(make_unit(TICKER, sid, td), cfg)
    top = a.year(Y)
    return top, (a.ensemble(top) if top else None)


@pytest.mark.parametrize("sid", SIDS)
@pytest.mark.parametrize("Y", YEARS)
def test_a_weights_use_only_past(base, sid, Y):
    td, cfg = base
    ys = year_bounds(td.dates, Y)[0]
    top0, ens0 = _top10(td, sid, cfg, Y)
    assert top0, "нет топ-10"
    top1, ens1 = _top10(_noisy(td, ys, 7), sid, cfg, Y)
    assert top1 == top0                                       # веса года Y не изменились
    assert np.array_equal(ens1[:ys], ens0[:ys])               # и позиции до начала года Y


def test_a_weights_sensitive_to_past(base):
    """Контроль: шум с бара за год до Y меняет топ-10 хотя бы у одной стратегии (тест не пустой)."""
    td, cfg = base
    Y = 2023
    k = year_bounds(td.dates, Y - 1)[0]
    changed = [sid for sid in SIDS
               if _top10(_noisy(td, k, 7), sid, cfg, Y)[0] != _top10(td, sid, cfg, Y)[0]]
    assert changed
