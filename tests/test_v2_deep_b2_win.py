"""G5 B2: окна, веса, частота, CUSUM — определения и мутации (№ 1, 15 и мутации потока)."""
import numpy as np
import pytest

from models_v2.s4 import fit, runner
from models_v2.s5.b2 import cusum, engine, loyo, synth_b2

T = 600


@pytest.fixture(scope="module")
def w():
    return synth_b2.World(*synth_b2.pair(700, 31))


def test_rolling_takes_exactly_n_last_pairs_not_future(w):
    dz = w.dz0
    full = runner.pairs_upto(dz, w.F, T)
    for n in (252, 504):
        idx = engine.train_pairs(dz, w.F, engine.Cfg("r", window=n), T)
        assert len(idx) == n and (idx == full[-n:]).all() and (idx + 1 <= T).all()
        assert len(full[-(n + 1):]) != n                      # мутант N+1 ловится
        assert (runner.pairs_upto(dz, w.F, T + 1)[-n:] + 1 <= T).all() is np.False_   # мутант «будущая пара»: t+1 = T+1 > T


def test_rolling_short_history_equals_expanding(w):
    full = runner.pairs_upto(w.dz0, w.F, 300)
    assert (engine.train_pairs(w.dz0, w.F, engine.Cfg("r", window=1008), 300) == full).all()


def test_weights_decrease_with_age_and_halve_at_half_life():
    idx = np.arange(0, 100)
    wt = fit.weights(idx, 99, 25)
    assert (np.diff(wt) > 0).all() and wt[-1] == 1.0 and wt[99 - 25] == pytest.approx(0.5)
    assert not (np.diff(wt[::-1]) > 0).all()                  # инверсия возраста (мутант) нарушила бы монотонность


def test_loyo_nearest_and_distance_weights():
    tr, val = np.arange(0, 200), np.arange(200, 250)
    k, dist = loyo.nearest(tr, val, n=30)
    assert len(k) == 30 and k.min() == 170 and (dist == 200 - k).all()          # ближайшие к блоку
    k2, d2 = loyo.nearest(np.r_[np.arange(0, 20), np.arange(300, 320)], val, n=10)
    assert set(k2) <= set(np.r_[np.arange(10, 20), np.arange(300, 310)]) and len(k2) == 10  # обе стороны


@pytest.mark.parametrize("freq,keyf", [("w", lambda d: d.isocalendar()[:2]), ("m", lambda d: (d.year, d.month))])
def test_blocks_train_at_first_date_and_cover_grid(w, freq, keyf):
    import pandas as pd
    grid = np.arange(300, 600)
    bl = engine.blocks(w.F, grid, freq)
    assert sorted(x for ds in bl.values() for x in ds) == list(grid)
    for Tb, ds in bl.items():
        assert Tb == min(ds)
        assert len({tuple(keyf(pd.Timestamp(w.F.dates[x]))) for x in ds}) == 1
    assert len(engine.blocks(w.F, grid, "d")) == len(grid)


def test_block_model_is_trained_at_first_date_mutant_last_date_caught(w):
    import pandas as pd
    from models_v2.s5.b2 import synth_b2 as sb
    grid = np.arange(300, 600)
    bl = engine.blocks(w.F, grid, "w")
    T0, ds = next((t, d) for t, d in bl.items() if len(d) >= 4 and t > 400)
    d_first = ds[0]
    cfg = engine.Cfg("w", freq="w")
    day0 = pd.Timestamp(w.F.dates[d_first])
    a, b, c = synth_b2.pair(700, 31)
    dmg = sb.World(sb.damage(a, day0), sb.damage(b, day0), sb.damage(c, day0))
    ok1, ok2 = w.forecast(cfg, T0, d_first), dmg.forecast(cfg, T0, d_first)
    assert ok1.tobytes() == ok2.tobytes()
    bad1, bad2 = w.forecast(cfg, ds[-1], d_first), dmg.forecast(cfg, ds[-1], d_first)   # мутант: T = последняя дата блока
    assert bad1.tobytes() != bad2.tobytes()


def _shift_series(seed=1, n=400, at=250, delta=3.0):
    e = np.random.default_rng(seed).standard_normal(n)
    e[at:] += delta
    return e


def test_cusum_detects_shift_within_10_rows_and_none_before():
    e = _shift_series()
    _, _, al = cusum.run(e, 5.0)
    hits = np.flatnonzero(al)
    assert not al[:250].any() and 250 <= hits[0] <= 259


def test_cusum_states_before_u_do_not_depend_on_later_errors_and_mutants_caught():
    e = _shift_series()
    e2 = e.copy()
    e2[300:] = 50.0
    s1, s2 = cusum.run(e, 5.0)[:2], cusum.run(e2, 5.0)[:2]
    assert all((a[:300] == b[:300]).all() for a, b in zip(s1, s2))
    # мутант: стандартизация по всей выборке — состояния до u зависят от поздних ошибок
    def full(x):
        return ((x - np.nanmean(x)) / np.nanstd(x, ddof=1))[:300]
    assert not (full(e) == full(e2)).all()
    # мутант: e_{d+1} вместо e_d (сдвиг на строку вперёд) — реакция на шок опережает его
    _, _, al_m = cusum.run(np.r_[e[1:], np.nan], 5.0)
    assert np.flatnonzero(al_m)[0] < np.flatnonzero(cusum.run(e, 5.0)[2])[0]


def test_cusum_nan_keeps_state_and_arrivals_shifted_by_one():
    e = _shift_series()
    e[260] = np.nan
    sp, sm, _ = cusum.run(e, 50.0)
    assert sp[260] == sp[259] and sm[260] == sm[259]
    y = np.arange(5.0)
    assert np.isnan(cusum.arrivals(y)[0]) and (cusum.arrivals(y)[1:] == y[:-1]).all()


def test_cusum_restart_window_has_at_least_252_pairs_and_uses_only_past_alarms():
    idx = np.arange(0, 1000)
    al = np.zeros(1000, bool)
    al[[100, 900]] = True
    assert (cusum.window_pairs(idx[:951], al, 950) == idx[:951][-252:]).all()      # 51 пар < 252 -> последние 252
    assert (cusum.window_pairs(idx[:700], al, 699) == idx[100:700]).all()
    al2 = al.copy()
    al2[800:] = True                                   # будущие срабатывания относительно T=699 не влияют
    assert (cusum.window_pairs(idx[:700], al2, 699) == cusum.window_pairs(idx[:700], al, 699)).all()
    assert (cusum.window_pairs(idx[:50], np.zeros(1000, bool), 49) == idx[:50]).all()
