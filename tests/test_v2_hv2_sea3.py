"""Ж-214, лист 3, V8: профиль сезонности (медиана долей слота, дни < d), RV на 20-минутной сетке, RV без первого бара, модельные строки Z1/Z2/Z4/Z5 и версия цели Z6.
Известные ответы на синтетике, главный тест обрезки построителя (мутант «профиль с днём d» ловится) и модели. Без данных и сети."""
import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC, hv2_sea_in as SI, hv2_seas as Z, hv2_seastage as SS, hv2_world as W
from tests.test_v2_hv2_dec import synth


def slots(n=120, seed=2, shape=None):
    rng = np.random.default_rng(seed)
    shape = np.linspace(3, 1, 53) if shape is None else shape
    sh = shape / shape.sum() * np.exp(rng.normal(0, 0.05, (n, 53)))
    return sh * np.exp(rng.normal(0, 0.5, n))[:, None]


def test_median_profile_known_answer_and_normalised():
    shape = np.linspace(3, 1, 53)
    M2 = slots(shape=shape) * 0 + shape[None, :] * np.exp(np.random.default_rng(1).normal(0, 0.5, 120))[:, None]
    s = SI.median_profile(M2)
    assert np.isnan(s[:59]).all() and np.allclose(s[60:], (shape / shape.mean())[None, :], rtol=1e-12)
    assert np.allclose(np.nanmean(s[60:], axis=1), 1.0)


def test_profile_excludes_day_d_and_lag0_mutant_differs():
    M2 = slots()
    d = 90
    M3 = M2.copy()
    M3[d] = M2[d][::-1] * 7.0                                    # день d с другой формой
    assert (SI.median_profile(M2)[d] == SI.median_profile(M3)[d]).all()
    assert not np.allclose(SI.median_profile(M2, lag=0)[d], SI.median_profile(M3, lag=0)[d])   # МУТАНТ: день d в профиле


def test_rolling_profile_uses_only_last_250_days():
    shape_a, shape_b = np.linspace(3, 1, 53), np.linspace(1, 3, 53)
    M2 = np.vstack([slots(200, 1, shape_a), slots(300, 2, shape_b)])
    s = SI.median_profile(M2, win=250)
    assert np.allclose(s[499], shape_b / shape_b.mean(), rtol=0.05) and not np.allclose(SI.median_profile(M2)[499], shape_b / shape_b.mean(), rtol=0.05)
    blk = M2[249:499]
    share = blk / blk.sum(axis=1, keepdims=True)
    med = np.median(share, axis=0)
    assert np.allclose(s[499], med / med.mean(), rtol=1e-12)


def test_rv20_known_answer_and_missing_pair_dropped():
    R = np.full((2, 53), np.nan)
    R[0, :6] = 1.0                                               # сетка 1: (0,1),(2,3),(4,5) -> 3·4 = 12; сетка 2: (1,2),(3,4) -> 2·4 = 8 (слот 6 нет); среднее 10
    R[1, :5] = [1.0, 1.0, np.nan, 1.0, 1.0]                      # сетка 1: (0,1) -> 4, (2,3) и (4,5) пропуск; сетка 2: (1,2) пропуск, (3,4) -> 4; среднее 4
    out = SI.rv20(R)
    assert out[0] == 10.0 and out[1] == 4.0
    assert np.isnan(SI.rv20(np.full((1, 53), np.nan))[0])
    R3 = np.full((1, 53), np.nan)
    R3[0, :2] = 1.0                                              # в сетке 2 нет ни одной пары -> день не определён (fallback в модели)
    assert np.isnan(SI.rv20(R3)[0])


def test_trim_first_bar_known_answer():
    M2 = np.full((2, 53), np.nan)
    M2[0, :6] = [9, 1, 1, 1, 1, 1]
    M2[1, 1:4] = 2.0                                             # слота 0 нет — RV не меняется
    z = SI.build_inputs(M2, np.sqrt(np.nan_to_num(M2)), pd.RangeIndex(2))
    assert z["z5"].iloc[0] == 5.0 and z["z5"].iloc[1] == 6.0


def test_builder_main_truncation_and_profile_day_d_mutant(monkeypatch):
    M2 = slots(200, 3)
    R = np.sqrt(M2) * np.random.default_rng(4).choice([-1, 1], M2.shape)
    idx = pd.bdate_range("2018-01-01", periods=200)
    assert SS.trunc_builder(M2, R, idx, 120) == 0.0
    orig = SI.median_profile
    monkeypatch.setattr(SI, "median_profile", lambda M, win=None, lag=1: orig(M, win, -5))    # МУТАНТ: в профиле дня d есть дни d+1…d+5
    assert SS.trunc_builder(M2, R, idx, 120) > 0


def test_rv_sa_ratio_near_one_for_stationary_profile():
    M2 = slots(150, 5)
    z = SI.build_inputs(M2, np.sqrt(M2), pd.RangeIndex(150))
    ratio = (z["z1"] / np.nansum(M2, axis=1)).dropna()
    assert len(ratio) == 90 and ratio.between(0.9, 1.2).all()


def test_model_rows_z_equal_a0_when_input_is_rv_and_truncation():
    w, m10 = synth(n=500, tail=300)
    wz = SI.with_inputs(w, pd.DataFrame({"z1": m10.rv, "z4": m10.rv * np.exp(np.random.default_rng(9).normal(0, 0.1, len(m10)))}))
    s2, nfb = SI.run_input(wz, "z1")
    a0 = DC.a0_forecast(w)
    assert np.allclose(s2, a0, rtol=1e-10) and nfb == 0                        # вход = RV: строка совпадает с A0
    s4, _ = SI.run_input(wz, "z4")
    assert np.isfinite(s4).all() and (s4 > 0).all() and not np.allclose(s4, a0, rtol=1e-6)
    i0 = int(wz.el[150])
    assert DC.trunc_diff(lambda x: SI.sigma(x, "z4"), wz, i0) == 0.0


def test_z_nan_input_falls_back_to_a0_and_counts():
    w, m10 = synth(n=500, tail=300)
    z = m10.rv.copy()
    z.iloc[int(w.el[100])] = np.nan
    wz = SI.with_inputs(w, pd.DataFrame({"z1": z}))
    a0 = DC.a0_forecast(w)
    s2, nfb = SI.run_input(wz, "z1", a0)
    assert nfb >= 1 and s2[100] == a0[100]


def test_z6_target_version_is_scaled_to_original_rv():
    w, m10 = synth(n=700, tail=420)
    wz = SI.with_inputs(w, pd.DataFrame({"z1": 2.0 * m10.rv}))    # RV_sa вдвое больше RV: шкала c_d ≈ 0,5 возвращает прогноз на исходную RV
    s2, sa, nfb = SI.run_target(wz, "z1")
    a0 = DC.a0_forecast(w)
    late = np.arange(len(s2)) > 280
    ratio = (s2 / a0)[late]
    assert (sa > 0).all() and ratio.min() > 0.8 and ratio.max() < 1.25 and (sa / a0)[late].mean() > 1.5
