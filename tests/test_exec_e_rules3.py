"""Ж-047, К2 (ч. 3): ЯКОРЬ. Все σ-выходы выключены, рыночный вход → exec_e даёт позиции Sk без изменений (побайтно):
pos = sig, gross = sig·ln(C[t+1]/C[t]), cost = comm·|Δpos| (+ выход на конце окна, Н5). Включены все новые режимы
(tick, stop_touch, reentry=False, check, close Н6); для лонг-только sig результат совпадает с E без правил."""
import math

import numpy as np
import pytest

from fastbt import exec_e_run
from fastbt.exec_e import Params, run
from test_exec_e import D

C = 0.0004


def anchor_case(seed, sig_vals):
    rng = np.random.default_rng(seed)
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    return [D([(c, c * 1.01, c * 0.99, c)]) for c in cl], cl, rng.choice(sig_vals, 60)


def check_anchor(seed, chk):
    new = Params(tick=0.01, stop_touch=True, reentry=False, check=chk, comm=C)
    days, cl, sig = anchor_case(seed, [-1, 0, 1])
    r = run("T", days, sig, np.full(60, 0.02), new, close=cl)
    lr = np.r_[[math.log(cl[t + 1] / cl[t]) for t in range(59)], 0.0]
    cost = C * np.abs(np.diff(np.r_[0.0, sig.astype(float)]))
    cost[-1] += C * abs(sig[-1])                                                # Н5
    assert np.array_equal(r.pos, sig.astype(float))
    assert np.array_equal(r.gross, sig * lr) and np.array_equal(r.cost, cost)
    days, cl, sig = anchor_case(seed, [0, 1])                                   # лонг-только: побайтно как E без правил
    a = run("T", days, sig, np.full(60, 0.02), Params(comm=C))
    b = run("T", days, sig, np.full(60, 0.02), new, close=cl)
    for f in ("pos", "gross", "cost", "free"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


@pytest.mark.parametrize("chk", ["bary", "zakrytie"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_anchor_sk_positions_unchanged(seed, chk):
    check_anchor(seed, chk)


def test_anchor_mutation_short_side_lost_caught(monkeypatch):
    orig = exec_e_run.Run._enter
    monkeypatch.setattr(exec_e_run.Run, "_enter", lambda s, i, j, b, px, i0, sg, d, sd=1: orig(s, i, j, b, px, i0, sg, d, 1))
    with pytest.raises(AssertionError):
        check_anchor(1, "bary")


def test_anchor_mutation_exit_left_on_caught(monkeypatch):
    """Включённый σ-выход ломает якорь: стоп 0,01σ срабатывает на любом шуме."""
    days, cl, sig = anchor_case(1, [0, 1])
    r = run("T", days, sig, np.full(60, 0.02), Params(b=0.01, stop_touch=True, comm=C), close=cl)
    assert any(t.reason == "stop" for t in r.trades)                            # выход виден; с выключенным — ни одного
    off = run("T", days, sig, np.full(60, 0.02), Params(comm=C), close=cl)
    assert not any(t.reason == "stop" for t in off.trades)
