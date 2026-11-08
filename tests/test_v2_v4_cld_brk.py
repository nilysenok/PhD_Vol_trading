"""V4-CLD (G32): признак «время после излома» — CUSUM: известный ответ на синтетике со сдвигом уровня, PIT (обрезка), мутации «данные d+1» и «окно стандартизации с днём d»."""
import numpy as np
import pandas as pd

from models_v2.s11 import v4_cld_brk as BK, v4_cld_run as R, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth

T0 = 400


def step_series(shift=10.0, n=520, seed=3):
    days = pd.bdate_range("2017-01-02", periods=n)
    x = np.random.default_rng(seed).normal(0, 1, n)
    x[T0:] += shift
    return pd.Series(x, index=days)


def test_known_answer_step_fires_on_shift_day_and_resets():
    x = step_series()
    brk, fired = BK.cusum(x)
    idx = np.flatnonzero(fired.to_numpy())
    assert idx[0] == T0                                        # сдвиг +10 с. о.: z ≈ 10, S⁺ = 9,5 > 5 — в день сдвига
    assert not fired.iloc[:T0].any()                           # до сдвига ложных срабатываний нет (seed фиксирован)
    assert brk.iloc[T0] == 0.0
    assert brk.iloc[T0 - 1] == np.log1p(T0 - 1)                # до первого срабатывания — дни с начала ряда


def test_two_sided_detects_downward_shift():
    _, fired = BK.cusum(-step_series())
    assert fired.to_numpy()[T0:].any() and not fired.to_numpy()[:T0].any()


def test_hand_computed_cusum_state_after_reset():
    """Сдвиг 3,2 с. о.: за день S⁺ растёт на ≈ 2,7; 2,7 < 5 < 5,4 — срабатывание на второй день (T0+1), затем сброс и снова два дня (T0+3); brk после срабатывания = ln(1 + число дней)."""
    x = step_series(shift=0.0)
    x.iloc[T0:] = 3.2                                         # без шума после сдвига: z детерминирован прошлым окном
    brk, fired = BK.cusum(x)
    mu, sd = BK.past_stats(x)
    sp = sm = 0.0
    want = []
    for i in range(T0, T0 + 8):                               # ручная рекурсия (независимо от реализации)
        z = (x.iloc[i] - mu[i]) / sd[i]
        sp, sm = max(0.0, sp + z - 0.5), max(0.0, sm - z - 0.5)
        if sp > 5.0:
            want.append(i)
            sp = sm = 0.0
    idx = [int(i) for i in np.flatnonzero(fired.to_numpy())]
    assert idx[:len(want)] == want and want[0] == T0 + 1 and want[1] == T0 + 3
    assert brk.iloc[T0 + 1] == 0.0 and brk.iloc[T0 + 2] == np.log1p(1)


def test_cut_future_does_not_change_feature():
    x = step_series()
    full, _ = BK.cusum(x)
    cut, _ = BK.cusum(x.iloc[:T0 + 5])
    assert np.array_equal(full.iloc[:T0 + 5].to_numpy(), cut.to_numpy())


def test_mutation_signal_with_data_d_plus_1_is_caught():
    """Срабатывание по данным d+1 (lead = 1): на обрезке до дня сдвига значение признака дня T0−1 отличается."""
    x = step_series()
    full, _ = BK.cusum(x, lead=1)
    cut, _ = BK.cusum(x.iloc[:T0], lead=1)
    assert not np.array_equal(full.iloc[:T0].to_numpy(), cut.to_numpy())
    f0, _ = BK.cusum(x)
    c0, _ = BK.cusum(x.iloc[:T0])
    assert np.array_equal(f0.iloc[:T0].to_numpy(), c0.to_numpy())


def test_standardization_window_excludes_current_day():
    x = step_series()
    mu0, sd0 = BK.past_stats(x)
    y = x.copy()
    y.iloc[T0] += 100.0
    mu1, sd1 = BK.past_stats(y)
    assert mu0[T0] == mu1[T0] and sd0[T0] == sd1[T0] and mu0[T0 + 1] != mu1[T0 + 1]


def test_real_path_prepare_brk_is_pit_and_lead_mutation_caught():
    """Через `prepare` (боевой путь признаков CLD): обрезка рядов после дня d не меняет brk дней ≤ d; мутация lead_brk = 1 меняет."""
    P, inp = synth(480)
    Zl = ZF.build_panel_z(P, inp)
    day = P["date"].sort_values().iloc[T0]
    step = Zl["date"] >= day
    Zl.loc[step, "med_lrv"] = Zl.loc[step, "med_lrv"] + 8.0
    cut = Zl[Zl["date"] <= day]
    def brk(Z, lead):
        Pp, X, names, *_ = R.prepare(P[P["date"] <= day] if len(Z) == len(cut) else P, Z, lead)
        return pd.Series(X[:, names.index("brk")], index=Pp["date"].to_numpy()).groupby(level=0).first()
    a, b = brk(Zl, 0), brk(cut, 0)
    assert np.array_equal(a[a.index <= day].to_numpy(), b.to_numpy())
    a1, b1 = brk(Zl, 1), brk(cut, 1)
    assert not np.array_equal(a1[a1.index <= day].to_numpy(), b1.to_numpy())


def test_build_table_columns_and_median_from_cross_column():
    P, inp = synth(300)
    Zl = ZF.build_panel_z(P, inp)
    t = BK.build_table(Zl)
    assert list(t.columns) == ["date", "med_lrv", "brk", "fired"] and t["date"].is_monotonic_increasing and (t["brk"] >= 0).all()
