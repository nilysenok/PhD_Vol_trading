"""S-C-IMPL-1 (тестировщик), синтетика: память компонент X_d / X_w / X_m, доли ŝ, Ĥ_X, сдвинутое разложение S-CKL, NaN до покрытия (Д4), разрыв 2022.
Мутации — подменой функций через monkeypatch (№ 43): метка t⁺ во входе, Ĥ по паре t + 1 > d, окно через разрыв, доли по всей выборке, S-CKL вперёд."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import cm_feat as CF
from s18_cm_guard import guard_real_files, synth_components  # noqa: F401  (autouse)

T, DAY = 420, 330
c, p, u, rv = synth_components(T, seed=0)
SEG = np.r_[np.zeros(250, int), np.ones(T - 250, int)]
J = np.abs(np.random.default_rng(1).normal(size=T)) * 1e-5


def mem(c_, p_, u_, seg=SEG, rows=(DAY,), fn=None):
    return (fn or CF.memory)(c_, p_, u_, seg, J, rows=list(rows)).iloc[DAY]


def corrupt(a, start):
    b = a.copy()
    b[start:] = b[start:] * 3.7 + 1e-3
    return b


def pit_ok(fn=None, start=DAY + 1):
    """Порча строк ≥ start (по умолчанию d + 1) не меняет ни одну колонку строки d (побайтно)."""
    a = mem(c, p, u, fn=fn)
    b = mem(corrupt(c, start), corrupt(p, start), corrupt(u, start), fn=fn)
    return a.equals(b) and np.isfinite(a.to_numpy(float)).all()


def test_memory_is_pit_including_har_forecast_on_pair_plus_one():
    assert pit_ok() and pit_ok(start=DAY + 5)
    assert not pit_ok(start=DAY)                                          # день d входит законно: тест различающий


def test_mutation_label_of_day_d_plus_1_in_input_caught():
    orig = CF.memory
    shifted = lambda c_, p_, u_, seg, j, rows=None: orig(np.roll(c_, -1), np.roll(p_, -1), np.roll(u_, -1), seg, j, rows)   # X_d := X_{d+1}
    assert not pit_ok(fn=shifted)


def test_mutation_har_forecast_with_pair_t_plus_1_after_d_caught(monkeypatch):
    real = CF.HC.harj_at
    monkeypatch.setattr(CF.HC, "harj_at", lambda panel, i: real(panel, i + 1))     # пара t + 1 = d + 1 в обучении
    assert not pit_ok()


def test_mutation_shares_over_whole_sample_caught():
    orig = CF.memory

    def whole(c_, p_, u_, seg, j, rows=None):
        m = orig(c_, p_, u_, seg, j, rows).copy()
        tot = np.nansum(c_) + np.nansum(p_) + np.nansum(u_)
        m["sc"], m["sp"], m["su"] = np.nansum(c_) / tot, np.nansum(p_) / tot, np.nansum(u_) / tot
        return m
    assert not pit_ok(fn=whole)


def test_shares_sum_to_one_in_01_and_windows_are_5_and_22_rows_of_segment():
    m = CF.memory(c, p, u, SEG, J, rows=[DAY]).iloc[200:]
    assert m[["sc", "sp", "su"]].isna().all(axis=1).sum() > 0                  # начало сегмента — NaN, не 0 (окно неполно)
    m = m.dropna(subset=["sc", "sp", "su"])
    assert (m[["sc", "sp", "su"]].sum(axis=1) - 1).abs().max() <= 1e-12 and ((m[["sc", "sp", "su"]] >= 0) & (m[["sc", "sp", "su"]] <= 1)).all().all()
    assert abs(m["c_w"].loc[DAY] - c[DAY - 4:DAY + 1].mean()) < 1e-18 and abs(m["c_m"].loc[DAY] - c[DAY - 21:DAY + 1].mean()) < 1e-18
    assert abs(m["sc"].loc[DAY] - m["c_m"].loc[DAY] / (m["c_m"] + m["p_m"] + m["u_m"]).loc[DAY]) < 1e-15


def test_window_does_not_cross_the_2022_break_and_mutation_caught(monkeypatch):
    first = 252                                                           # 2-я строка сегмента 1 (индекс 250 — первая): окно 22 строк ещё неполно
    a = CF.memory(c, p, u, SEG, J, rows=[first]).iloc[first]
    b = CF.memory(corrupt(c, 0)[:250].tolist() + c[250:].tolist(), p, u, SEG, J, rows=[first]).iloc[first]
    assert a.equals(b) or (a.isna() == b.isna()).all() and a.dropna().equals(b.dropna())          # порча сегмента 0 не меняет строки сегмента 1
    one = np.zeros(T, int)                                                # мутант: единый сегмент — окно склеено
    c2 = np.r_[corrupt(c[:250], 0), c[250:]]
    x = CF.memory(c, p, u, one, J, rows=[first]).iloc[first]
    y = CF.memory(c2, p, u, one, J, rows=[first]).iloc[first]
    assert not x.dropna().equals(y.dropna())


def tab(first=40, n=300):
    cc, pp, uu, r = synth_components(n, seed=2)
    return pd.DataFrame(dict(n_cov=np.r_[np.zeros(first), np.ones(n - first)], rv_common=cc, rv_specific=pp, rv_uncovered=uu, rv=r))


def test_components_nan_before_first_covered_session_not_zero_and_mutation_caught():
    k = CF.components(tab())
    assert np.isnan(k["c"][:40]).all() and np.isnan(k["p"][:40]).all() and np.isnan(k["u"][:40]).all()
    assert np.isfinite(k["c"][40:]).all() and (k["c"][40:] > 0).all() and (k["p"][40:] > 0).all() and (k["u"][40:] >= 0).all()
    assert np.isfinite(k["rv"]).all() and abs(k["c"][40:] + k["p"][40:] + k["u"][40:] - k["rv"][40:]).max() / k["rv"].max() <= 1e-13
    zero_filled = np.nan_to_num(k["c"])                                   # мутант Д4: ноль вместо NaN
    assert not np.isnan(zero_filled[:40]).all()


def test_shift_decomp_known_answer_sum_nonneg_and_only_past():
    lag = 30
    ct, pt, ut = CF.shift_decomp(c, p, u, rv, lag=lag)
    assert np.isnan(ct[:lag]).all() and np.isnan(pt[:lag]).all()
    assert np.nanmax(np.abs(ct + pt + ut - rv)) <= 1e-15 * rv.max() * 10 and (ct[lag:] >= 0).all() and (pt[lag:] >= 0).all()   # сумма и RV не меняются
    q = c[:-lag] / (c[:-lag] + p[:-lag])
    assert np.abs(ct[lag:] - (rv[lag:] - u[lag:]) * q).max() <= 1e-18 and (ut[lag:] == u[lag:]).all()
    k = 200
    c2 = c.copy()
    c2[k] *= 5
    ct2, _, _ = CF.shift_decomp(c2, p, u, rv, lag=lag)
    assert np.array_equal(ct2[:k + lag], ct[:k + lag], equal_nan=True) and ct2[k + lag] != ct[k + lag]   # метка c_k влияет с s = k + lag, не раньше


def test_shift_zero_gives_original_decomposition():
    """Контроль (б) со сдвигом 0 = исходное разложение (аналог к2 при lag = 0, № 9): c̃ = (rv − u)·c/(c + p) = c, p̃ = p, ũ = u (до округления: rv − u = c + p)."""
    ct, pt, ut = CF.shift_decomp(c, p, u, rv, lag=0)
    assert np.abs(ct - c).max() / c.max() <= 1e-12 and np.abs(pt - p).max() / p.max() <= 1e-12 and (ut == u).all()


def test_mutation_shift_decomp_forward_caught(monkeypatch):
    def forward(c_, p_, u_, rv_, lag=30):
        q = np.full(len(c_), np.nan)
        q[:-lag] = c_[lag:] / (c_[lag:] + p_[lag:])                       # s + lag: будущее
        return (rv_ - u_) * q, (rv_ - u_) * (1 - q), u_
    c2 = c.copy()
    c2[200] *= 5
    ct, _, _ = forward(c, p, u, rv)
    ct2, _, _ = forward(c2, p, u, rv)
    assert not np.array_equal(ct2[:230], ct[:230], equal_nan=True)        # тот же тест «только прошлое» падает на мутанте
    assert CF.SHIFT == 252


def test_harj_hat_equals_hp_ctrl_harj_at_on_component_panel_bitwise():
    """F.G2: Ĥ_X = hp_ctrl.harj_at над make_panel ряда уровней X; при подстановке X := RV получается тот же код, что у K1/K7."""
    x = rv
    h = CF.harj_hat(x, SEG, J, rows=[DAY, DAY - 7])
    panel = CF.H.make_panel(pd.DataFrame({"rv": x, "rv_w": CF.DC.wmean(x, SEG, 5), "rv_m": CF.DC.wmean(x, SEG, 22), "j": J, "seg": SEG}))
    assert h[DAY] == CF.HC.harj_at(panel, DAY) and h[DAY - 7] == CF.HC.harj_at(panel, DAY - 7)
    assert np.isnan(h[0]) and np.isfinite(h[DAY]) and h[DAY] > 0


def test_har_forecast_needs_252_pairs():
    h = CF.harj_hat(rv, SEG, J, rows=[100, 253, DAY])                     # сегмент 0 — 250 строк: пар < 252
    assert np.isnan(h[100]) and np.isfinite(h[DAY])
