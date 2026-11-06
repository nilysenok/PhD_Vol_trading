"""G3 (Ж-206, волна D, поток XGB): мутации темы «окна / веса / частота / режимы / критерий / учёт». Синтетика, без сети. Каждая мутация эмулируется в тесте и обязана ловиться проверкой."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import fit
from models_v2.s5 import pit
from models_v2.s5.b1 import designs, jobs
from models_v2.s5.b2 import feat, synth_b2
from models_v2.s6.xgb import c_space, cfgx, d_comp, d_fit, d_regime, d_rows, d_space

D = 560
HP = c_space.to_hpc(synth_b2.HP)


@pytest.fixture(scope="module")
def world():
    s = synth_b2.World(*synth_b2.pair(700, 41))
    return {"F": s.F, "dz0": s.dz0, "feats": s.feats, "acc": {g: list(c) for g, c in feat.GROUPS.items()}, "eps": pit.eps_floor(s.F.rv)}


def assert_window(m, n, T):
    """Окно: ровно n последних пар, все t+1 <= T (ни одной будущей пары)."""
    t = m["train_idx"]
    assert len(t) == n and t.max() + 1 <= T and (np.diff(t) > 0).all()


def test_window_exact_n_pairs_and_no_future_pair_mutation_n_plus_one(world):
    spec = d_space.spec_for(world)
    idx = jobs.pairs(world["F"], spec, D)
    m = d_fit.fit_d(spec.X, spec.y, idx, HP, 300, 0, D)
    assert_window(m, 300, D)
    bad = d_fit.fit_d(spec.X, spec.y, idx, HP, 301, 0, D)                    # мутация: окно N+1
    with pytest.raises(AssertionError):
        assert_window(bad, 300, D)
    with pytest.raises(AssertionError):                                       # мутация: будущая пара в окне
        assert_window({"train_idx": np.append(m["train_idx"][1:], D)}, 300, D)


def test_weights_match_independent_formula_and_inversion_is_caught(world):
    spec = d_space.spec_for(world)
    idx = jobs.pairs(world["F"], spec, D)
    got = d_fit.fit_d(spec.X, spec.y, idx, HP, 0, 252, D)
    w = 0.5 ** ((D - np.arange(len(spec.y))) / 252)
    ref = fit.fit_i1(spec.X, spec.y, idx, HP, w, None)                       # независимая сборка весов
    assert got["k"] == ref["k"]
    inv = fit.fit_i1(spec.X, spec.y, idx, HP, 0.5 ** (-(D - np.arange(len(spec.y))) / 252), None)   # мутация: инверсия весов (старое тяжелее)
    x = spec.X[[D]]
    assert np.asarray(got["predict"](x)).tobytes() == np.asarray(ref["predict"](x)).tobytes()
    assert np.asarray(inv["predict"](x)).tobytes() != np.asarray(got["predict"](x)).tobytes()


def test_zero_window_zero_weights_equals_c_fit_bitwise(world):
    from models_v2.s6.xgb import c_fit
    spec = d_space.spec_for(world)
    idx = jobs.pairs(world["F"], spec, D)
    a = d_fit.fit_d(spec.X, spec.y, idx, HP, 0, 0, D)
    b = c_fit.fit_c(spec.X, spec.y, idx, HP, "l2", fit.SEED)
    x = spec.X[[D]]
    assert np.asarray(a["predict"](x)).tobytes() == np.asarray(b["predict"](x)).tobytes() and a["k"] == b["k"] and a["s2_es"] == b["s2_es"]


def assert_blocks(bl, grid):
    allr = sorted(d for ds in bl.values() for d in ds)
    assert allr == sorted(int(g) for g in grid)
    assert all(T == min(ds) for T, ds in bl.items())                          # модель блока обучена на первой дате, не на последней


@pytest.mark.parametrize("freq", ["d", "w", "m"])
def test_blocks_train_on_first_date_mutation_last_date(world, freq):
    F = world["F"]
    grid = np.arange(300, 600)
    bl = d_space.blocks(F, grid, freq)
    assert_blocks(bl, grid)
    assert len(bl) == 300 if freq == "d" else len(bl) < 300
    bad = {max(ds): ds for ds in bl.values()}                                   # мутация: T = последняя дата блока
    if freq != "d":
        with pytest.raises(AssertionError):
            assert_blocks(bad, grid)


def test_freq_threshold_declared_and_mutation_zero_threshold_caught():
    assert d_space.FREQ_THRESHOLD == 0.0005
    lim = {"w": 0.2000, "m": 0.2000}
    assert d_comp.freq_choice({**lim, "d": 0.1996}) == "m"                      # выигрыш 0,0004 < порога: дешёвое остаётся
    assert d_comp.freq_choice({**lim, "d": 0.1994}) == "d"
    assert d_comp.freq_choice({**lim, "d": 0.1996}, thr=0.0) == "d"             # мутация: нулевой порог выбрал бы ежедневное
    assert d_comp.freq_choice({"d": 0.2000, "w": 0.1998, "m": 0.2000}) == "m"   # w лучше m на 0,0002 < порога


def test_freq_chain_daily_vs_chosen_cheaper():
    assert d_comp.freq_choice({"d": 0.1990, "w": 0.1998, "m": 0.2000}) == "d"
    assert d_comp.freq_choice({"d": 0.2001, "w": 0.1990, "m": 0.2000}) == "w"


def test_stress_state_uses_only_prior_252_values_mutation_current_day(world):
    F = world["F"]
    st = d_comp.stress_state(F)
    rvw = np.exp(F.X[:, 1])
    ks = [k for k in range(300, 640) if np.isfinite(rvw[k - 252:k]).sum() >= 242 and np.isfinite(rvw[k])]
    good = [bool(rvw[k] > np.nanquantile(rvw[k - 252:k], 2 / 3)) for k in ks]
    mut = [bool(rvw[k] > np.nanquantile(rvw[k - 251:k + 1], 2 / 3)) for k in ks]   # мутация № 22: квантиль с текущим днём
    assert len(ks) > 200 and [bool(st[k]) for k in ks] == good
    assert mut != good                                                              # тест не вакуумен: мутация меняет состояние


def test_compose_raw_and_folds_use_state_at_t_mutation_t_plus_one():
    n = 10
    a = pd.DataFrame({"row": np.arange(n), "f": 1.0, "s2": 1.0, "k": 1, "T": np.arange(n), "secs": 1.0})
    b = a.assign(f=2.0)
    state = np.array([True, False] * 5)
    out = d_comp.compose_raw(a, b, state)
    assert (out["f"].to_numpy() == np.where(state, 1.0, 2.0)).all()
    assert (d_comp.compose_raw(a, b, np.roll(state, -1))["f"].to_numpy() != out["f"].to_numpy()).all()    # мутация: состояние d+1
    la = pd.DataFrame({"Y": 2018, "y": 2017, "row": np.arange(n), "q": np.arange(n, dtype=float)})
    lb = la.assign(q=100.0 + (np.arange(n) * 7) % 11)             # разность q_B − q_A не постоянна: перестановка состояния меняет среднее
    q = d_comp.compose_folds(la, lb, state)["q"].iloc[0]
    assert q == pytest.approx(np.where(state, la["q"], lb["q"]).mean())
    st2 = np.array([True, True, False, True, False, False, False, True, True, False])
    q2 = d_comp.compose_folds(la, lb, st2)["q"].iloc[0]
    assert q2 == pytest.approx(np.where(st2, la["q"], lb["q"]).mean())
    assert d_comp.compose_folds(la, lb, np.roll(st2, -1))["q"].iloc[0] != pytest.approx(q2)   # мутация: состояние строки t+1
    with pytest.raises(ValueError):
        d_comp.compose_folds(la, lb.iloc[:-1], state)


def test_annual_raw_picks_window_of_target_year_mutation_next_year():
    dates = np.array(pd.date_range("2017-12-20", "2019-01-15", freq="B"), dtype="datetime64[ns]")
    F = SimpleNamespace(dates=dates)
    rows_ = np.arange(len(dates) - 1)
    mk = lambda v: pd.DataFrame({"row": rows_, "f": v, "s2": v, "k": 1, "T": rows_, "secs": 1.0})  # noqa: E731
    raws = {"expanding": mk(0.0), "roll504": mk(1.0)}
    out = d_comp.annual_raw(F, raws, {2017: None, 2018: "roll504", 2019: None})
    yr = dates[rows_ + 1].astype("datetime64[Y]").astype(int) + 1970
    assert (out["f"].to_numpy() == np.where(yr == 2018, 1.0, 0.0)).all()
    bad = d_comp.annual_raw(F, raws, {2017: "roll504", 2018: None, 2019: None})   # мутация: выбор года Y применён к другому году
    assert (bad["f"].to_numpy() != out["f"].to_numpy()).any()


def test_crit_wf_uses_only_past_years_mutation_includes_year_Y():
    td = pd.Series(pd.to_datetime(["2017-06-01", "2018-06-01", "2018-07-02", "2019-06-03"]))
    q = np.array([1.0, 2.0, 4.0, 10.0])
    got = d_comp.crit_wf(q, td)
    # Y=2018: прошлое {1.0} → 1,0; Y=2019: прошлое {1.0, 3.0}: mean 2, max 3 → 2,25. Среднее = 1,625
    assert got == pytest.approx(1.625)
    assert d_comp.crit_wf(q[::-1].copy(), td[::-1].reset_index(drop=True)) == pytest.approx(got)
    assert d_comp.crit_wf(np.array([1.0, 2.0, 4.0, 1e6]), td) == pytest.approx(got)   # будущее (2019) в свой критерий не входит; 2019 — только как цель


def test_switches_delay_and_transitions_on_crafted_series():
    s = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], bool)
    assert d_regime.n_switches(s) == 3
    m = d_regime.transition_mask(s)
    assert m[2] and m[9] and not m[10] and m[11] and not m[1]
    st = np.zeros(40, bool)
    st[12:] = True
    out = d_regime.reaction_delay(st, np.array([10, 20]))
    assert out["n_shocks"] == 2 and out["median_delay"] == 1.0 and out["share_same_day"] == 0.5 and out["share_flagged"] == 1.0


def test_budget_limit_and_no_double_count(tmp_path, monkeypatch):
    trials, ck = tmp_path / "t.csv", tmp_path / "c.json"
    ck.write_text(json.dumps({"used": {"temporal_regime": 0}}))
    monkeypatch.setattr(d_rows, "TRIALS", trials)
    monkeypatch.setattr(d_rows, "CKPT", ck)
    rec = lambda i: {"trial_id": "t%d" % i, "wave": "D-xgb", "direction": "x", "config": "c", "metric_inner": "m", "seed": 42, "time": "t"}  # noqa: E731
    assert d_rows.register([rec(i) for i in range(30)]) == 30
    assert d_rows.register([rec(0), rec(1)]) == 0                                   # повтор — не trial
    assert json.loads(ck.read_text())["used"]["temporal_regime"] == 30
    with pytest.raises(ValueError):
        d_rows.register([rec(30)])                                                  # 31-й — отказ до записи
    assert len(pd.read_csv(trials)) == 30
    assert cfgx.TRIAL_HEAD[0] == "trial_id"


def test_trial_id_depends_on_every_axis():
    base = d_space.DCfg("f4")
    ids = {d_space.trial_id(c) for c in (base, d_space.DCfg("j"), d_space.DCfg("f4", "roll504"), d_space.DCfg("f4", "exp756"), d_space.DCfg("f4", freq="w"),
                                          d_space.DCfg("f4", r1="qw"), d_space.DCfg("f4", r1="all"))}
    assert len(ids) == 7 and len(d_space.default_cfgs()) == 12
    assert len({d_space.trial_id(c) for c in d_space.default_cfgs()}) == 12
