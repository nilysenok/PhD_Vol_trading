"""Ж-044: PBO по CSCV (sweep_drivers.pbo_cscv, sweep_pbo) и T в PSR/DSR. Синтетика."""
import inspect
import math
import sys
import textwrap

import numpy as np
import pytest

sys.path[:0] = [".", "scripts"]
from fastbt import sweep_drivers as sd                  # noqa: E402
from fastbt import sweep_pbo                            # noqa: E402
from fastbt import sweep_val_stat as svs                # noqa: E402

T_DAYS, N_COLS = 400, 20


def _mutated_pbo(old, new):
    src = textwrap.dedent(inspect.getsource(sd.pbo_cscv))
    assert old in src
    ns = dict(vars(sd))
    exec(src.replace(old, new), ns)
    return ns["pbo_cscv"]


def _noise(seed):
    return np.random.default_rng(seed).normal(0, 1, (T_DAYS, N_COLS)).tolist()


def _signal(seed):
    m = np.random.default_rng(seed).normal(0, 1, (T_DAYS, N_COLS))
    m[:, 3] += 0.4                                       # единственный настоящий сигнал
    return m.tolist()


def test_pbo_noise_about_half():
    v = [sd.pbo_cscv(_noise(s), 8)["pbo"] for s in range(12)]
    assert 0.35 < float(np.mean(v)) < 0.65


def test_pbo_one_real_signal_below_02():
    v = [sd.pbo_cscv(_signal(s), 8)["pbo"] for s in range(12)]
    assert float(np.mean(v)) < 0.2 and max(v) < 0.2


def test_pbo_mutation_logit_sign_caught():
    bad = _mutated_pbo("math.log(w / (1 - w))", "math.log((1 - w) / w)")
    assert bad(_signal(0), 8)["pbo"] > 0.8 > sd.pbo_cscv(_signal(0), 8)["pbo"]


def test_pbo_eight_blocks_give_70_splits_and_default_is_8():
    r = sd.pbo_cscv(_noise(0), 8)
    assert r["n_splits"] == math.comb(8, 4) == 70 and len(r["logits"]) == 70
    assert sd.pbo_cscv(_noise(0))["n_splits"] == 70
    assert sweep_pbo.N_BLOCKS == 8
    assert sd.pbo_cscv(_noise(0), 6)["n_splits"] == 20      # мутация «6 блоков» отличима


def test_pbo_mutation_blocks_in_sweep_pbo_caught(monkeypatch):
    ex = {f"k{i}": dict(daty=list(range(T_DAYS)), excess_d=[r[i] for r in _noise(1)]) for i in range(N_COLS)}
    assert sweep_pbo.pbo(ex)["n_splits"] == 70
    monkeypatch.setattr(sweep_pbo, "N_BLOCKS", 6)
    assert sweep_pbo.pbo(ex, 6)["n_splits"] != 70


def test_pbo_rows_are_days_columns_are_trials():
    m = _noise(2)
    assert len(m) == T_DAYS and len(m[0]) == N_COLS
    assert not math.isnan(sd.pbo_cscv(m, 8)["pbo"])
    ex = {f"k{i}": dict(daty=list(range(T_DAYS)), excess_d=[r[i] for r in m]) for i in range(N_COLS)}
    out = sweep_pbo.pbo(ex)
    assert out["T"] == T_DAYS and out["n_prob"] == N_COLS


def test_pbo_degenerate_nan():
    assert math.isnan(sd.pbo_cscv([[0.1]] * 400, 8)["pbo"])           # один столбец
    assert math.isnan(sd.pbo_cscv(_noise(0)[:5], 8)["pbo"])           # дней меньше блоков


# --- T в PSR/DSR: число дневных наблюдений ---------------------------------------------------------------

def _ref_psr(sr, star, T, sk, ku):
    den = math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr * sr)
    return 0.5 * math.erfc(-((sr - star) * math.sqrt(T - 1) / den) / math.sqrt(2))


TRIALS = [0.02, 0.05, 0.08, 0.11, 0.03, 0.07]


def test_dsr_uses_T_as_given_daily_count():
    d = svs.dsr_n(0.12, TRIALS, 1100, 504, 0.1, 4.0)
    assert d["dsr"] == pytest.approx(_ref_psr(0.12, d["sr_star"], 504, 0.1, 4.0), rel=1e-9)


def test_dsr_mutation_T_equals_trades_caught(monkeypatch):
    good = svs.dsr_n(0.12, TRIALS, 1100, 504, 0.1, 4.0)["dsr"]
    real = sd.psr
    monkeypatch.setattr(sd, "psr", lambda sr, st, T, s=0.0, k=3.0: real(sr, st, 60, s, k))     # T подменён числом сделок
    bad = svs.dsr_n(0.12, TRIALS, 1100, 504, 0.1, 4.0)["dsr"]
    assert abs(good - bad) > 0.05


def test_psr_depends_on_T_with_more_days_more_confident():
    assert sd.psr(0.1, 0.0, 504) > sd.psr(0.1, 0.0, 60)


def test_metriki_e_psr_T_is_len_of_daily_series():
    from fastbt import metriki_e as me
    x = np.random.default_rng(5).normal(0.001, 0.01, 504)
    sr = me.sr_dnevnoy(x)
    exp = _ref_psr(sr, 0.0, 504, me.asimmetriya(x), me.ekscess(x) + 3.0)
    assert me.psr(x) == pytest.approx(exp, rel=1e-6)
    assert me.psr(x[:60]) != pytest.approx(me.psr(x), abs=1e-3)       # T = число наблюдений, а не константа
