"""V7-DX (G40, G38, G37): PIT-регрессия ошибки XGBoost, известные ответы на синтетике, мутации «обучение на будущих датах», «знак снижения», «профиль с днём i», «множитель по всем парам»."""
import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1m_wf as WF, v7_dx_core as K, v7_dx_dup as DUP, v7_dx_enc as ENC, v7_dx_sig as SIG


def synth(n=1500, related=False, regime=False, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-12-30", periods=n)
    z = rng.normal(size=(n, 4))
    tp = np.append(idx.to_numpy("datetime64[ns]")[1:], idx[-1].to_datetime64() + np.timedelta64(1, "D"))
    y = rng.normal(scale=0.5, size=n)
    if related:
        y = y + 0.8 * z[:, 0]
    if regime:                                     # в конце второго года у z появляется метка режима, который прошлое не знает
        z[:, 1] = (np.arange(n) >= 700).astype(float) + 0.05 * rng.normal(size=n)
        y = y + 2.0 * (np.arange(n) >= 700)
    return y, z, tp, idx


def run_pit(y, z, tp, idx):
    pos = W6.refit_positions(idx, WF.FIRST_REFIT)
    df = DUP.pit_ridge(y, z, tp, idx, pos, first_eval=str(idx[300].date()))
    ty = pd.DatetimeIndex(tp).year.to_numpy()
    return df, ty


def mean_r2(df, ty):
    r = DUP.r2_by_year(df, ty)
    return float(np.mean(list(r.values())))


def test_known_answer_z_unrelated_gives_r2_near_zero():
    df, ty = run_pit(*synth())
    assert abs(mean_r2(df, ty)) < 0.03


def test_known_answer_z_related_gives_positive_r2():
    df, ty = run_pit(*synth(related=True))
    assert mean_r2(df, ty) > 0.3


def test_mutation_training_on_future_dates_is_caught(monkeypatch):
    """Метка режима у z + сдвиг среднего y: PIT не знает будущего режима; обучение на всех парах (мутант) знает — R² заметно выше."""
    y, z, tp, idx = synth(regime=True)
    honest = mean_r2(*run_pit(y, z, tp, idx))
    monkeypatch.setattr(DUP, "closed", lambda tp_, d0: np.arange(len(tp_)))
    mutant = mean_r2(*run_pit(y, z, tp, idx))
    assert mutant > honest + 0.05


def test_pit_trains_only_on_closed_pairs():
    """Прогноз на дату d не меняется при замене y пар с t⁺ > d (прошлое закрыто)."""
    y, z, tp, idx = synth(related=True)
    pos = W6.refit_positions(idx, WF.FIRST_REFIT)
    a = DUP.pit_ridge(y, z, tp, idx, pos, first_eval=str(idx[400].date()))
    y2 = y.copy()
    y2[800:] += 50.0
    b = DUP.pit_ridge(y2, z, tp, idx, pos, first_eval=str(idx[400].date()))
    early = a["i"] < 780
    assert np.array_equal(a["pred"][early].to_numpy(), b["pred"][b["i"] < 780].to_numpy())


def test_partial_corr_known_answer_and_sign():
    rng = np.random.default_rng(1)
    C = rng.normal(size=(4000, 4))
    v = 2.0 * C[:, 0] + 0.1 * rng.normal(size=4000)
    pc = DUP.partial_corrs(v, C)
    assert pc[0] > 0.99 and all(abs(p) < 0.05 for p in pc[1:4]) and pc[4] > 0.99
    assert DUP.partial_corrs(-v, C)[0] < -0.99                                 # знак не теряется (перепутанный знак ловится)


def test_ridge_cv_known_answers():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1000, 4))
    assert abs(SIG.ridge_cv(X, rng.normal(size=1000), 10.0)) < 0.03
    assert SIG.ridge_cv(X, X[:, 0] + 0.5 * rng.normal(size=1000), 10.0) > 0.6


def test_q_red_sign_and_zero():
    rv, H = np.array([1.0, 2.0, 0.5, 1.5]), np.array([1.5, 1.0, 1.0, 1.0])
    assert K.q_red(rv, H, H)[2] == 0.0
    assert K.q_red(rv, H, rv)[2] > 99.9                                        # идеальный прогноз: QLIKE = 0, снижение 100 %
    assert K.q_red(rv, H, H * 3.0)[2] < 0                                      # хуже H — минус


def test_naive_profile_uses_only_prior_days_and_mean_only_bars_up_to_j():
    rng = np.random.default_rng(3)
    r2 = np.abs(rng.normal(size=(30, 53))) + 0.01
    ok = np.ones((30, 53), bool)
    ok[:, -1] = False
    y = np.log(np.roll(r2, -1, axis=1) + 1e-3)
    a = ENC.naive_forecasts(r2, y, ok, 1e-3, 27)
    r2b, yb = r2.copy(), y.copy()
    r2b[29, :], yb[29, :] = 9.0, 9.0                                           # изменён последний день
    b = ENC.naive_forecasts(r2b, yb, ok, 1e-3, 27)
    assert np.array_equal(a["profile"][:2], b["profile"][:2])                  # профиль дней 27, 28 не видит день 29
    r2c = r2.copy()
    r2c[28, 30:] = 9.0                                                         # бары после j = 29 меняются
    c = ENC.naive_forecasts(r2c, y, ok, 1e-3, 27)
    assert np.array_equal(a["mean"][1, :30], c["mean"][1, :30]) and np.array_equal(a["ewma"][1, :30], c["ewma"][1, :30])
    assert not np.array_equal(a["mean"][1, 30:], c["mean"][1, 30:])


def test_stability_aligns_permutation_and_sign():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(500, 4))
    perm = z[:, [2, 0, 3, 1]] * np.array([1, -1, 1, -1])
    s = ENC.stability([z, perm])
    assert s["aligned_mean"] > 0.999 and s["unaligned_mean"] < 0.3
    assert ENC.stability([z, rng.normal(size=(500, 4))])["aligned_mean"] < 0.2


def test_const_control_is_pit_and_mutation_caught(monkeypatch):
    y, z, tp, idx = synth()
    H = np.ones(len(y))
    full = pd.DataFrame({"H": H, "rv_next": np.exp(0.3 * (np.arange(len(y)) >= 700) + 0.05 * y), "t_plus": tp}, index=idx)
    ok = SIG.const_control(full)
    monkeypatch.setattr(DUP, "closed", lambda tp_, d0: np.arange(len(tp_)))
    mut = SIG.const_control(full)
    pairs = [(ok[k][1], mut[k][1]) for k in ok if np.isfinite(ok[k][1]) and np.isfinite(mut[k][1])]
    assert pairs and max(m - h for h, m in pairs) > 1.0                       # множитель по всем парам видит сдвиг уровня заранее
