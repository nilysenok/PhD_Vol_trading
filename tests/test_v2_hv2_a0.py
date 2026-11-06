"""Ж-214, A0: HAR-J(б) восстановлена и воспроизводит замороженную колонку HAR побитно; PIT (пары t+1 <= d); мутация «будущие пары»;
подгонка по известному ответу. Тесты на реальных данных пропускаются, если локальных файлов нет."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s9 import hv2_a0 as A0, hv2_paths as P


def synth_panel(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 0.1 + X @ np.array([0.4, 0.3, 0.2, -0.05])          # ln RV_{t+1} = f(X_t) точно
    lnrv = np.r_[0.0, y[:-1]]
    dates = pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]")
    return H.Panel(dates, np.zeros(n, int), np.exp(lnrv), X)


def test_fit_recovers_known_coefficients():
    p = synth_panel()
    b, s2, n = A0.fit_at(p, 150)
    assert np.allclose(b, [0.1, 0.4, 0.3, 0.2, -0.05], atol=1e-9) and s2 < 1e-18 and n == 150


def test_pit_future_rows_do_not_change_fit_and_mutation_is_caught():
    p = synth_panel()
    i = 120
    b0, s0, _ = A0.fit_at(p, i)
    q = H.Panel(p.dates, p.seg, p.rv.copy(), p.X.copy())
    q.rv[i + 1:] *= 5.0
    q.X[i + 1:] += 3.0
    b1, s1, _ = A0.fit_at(q, i)
    assert np.array_equal(b0, b1) and s0 == s1
    bm, _, nm = A0.fit_at(q, i, extra=1)                      # мутант: пара (i, i+1) с испорченной целью
    assert nm == 121 and not np.allclose(bm, b0)


def test_segments_not_glued():
    p = synth_panel(60)
    seg = np.r_[np.zeros(30, int), np.ones(30, int)]
    q = H.Panel(p.dates, seg, p.rv, p.X)
    assert A0.fit_at(q, 50)[2] == 49                          # t = 0..49 минус пара (29, 30) через разрыв


@pytest.mark.skipif(not (P.SBER.exists() and P.COMP.exists()), reason="локальные данные")
def test_reproduces_frozen_har_bitwise_on_sample():
    bars = pd.read_parquet(P.SBER)
    _, p, el = A0.panel_grid(bars)
    comp = pd.read_parquet(P.COMP)
    idx = np.r_[0:40, 1000:1010, len(el) - 20:len(el)]
    s = A0.harj_series(p, el[idx])
    assert (s["d"].to_numpy() == comp["d"].to_numpy()[idx]).all()
    assert np.array_equal(s["sigma2_b"].to_numpy(), comp["HAR"].to_numpy()[idx])
    assert not np.allclose(s["sigma2_a"].to_numpy(), comp["HAR"].to_numpy()[idx])   # вариант (а) без s²/2 — не контроль
    sm = A0.harj_series(p, el[idx], extra=1)
    assert not np.allclose(sm["sigma2_b"].to_numpy(), comp["HAR"].to_numpy()[idx])  # мутант «будущие пары» ловится
