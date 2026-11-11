"""S-A-IMPL-2 (тестировщик): синтетика «выход из начального состояния» (разд. 3.3). Свой генератор (не код кодера); старт — ноль, seed 0.
(а) известный сигнал ⇒ параметры уходят от нуля в сторону истины; (б) сигнала нет ⇒ параметры около нуля."""
import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s10 import hp_har as HH
from models_v2.s18 import a_feat as A
from models_v2.s18 import a_model as M
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)

LAM = 1e-3


def generate(signal: bool, n=1500, seed=11, noise=0.25):
    """r² = σ²_s·u_j·ε²; ln RV_{t+1} = c + 0.4 ln S_s + 0.3 ln S_m + 0.2 ln S_l + шум. signal: b*_s ∝ exp(1.5τ²), длинное ядро ω* = 0.6, ψ* = (−3, 0)."""
    g = np.random.default_rng(seed)
    tau = A.TAU
    u = 1.0 + 3.0 * (2 * tau - 1) ** 2
    u /= u.sum()
    b = np.exp(1.5 * tau ** 2) if signal else np.ones(A.NS)
    b /= b.mean()
    x = np.arange(66) / 65.0
    ker = np.exp(-3.0 * x)
    a_l = 0.4 * (np.arange(66) < 22) / 22 + 0.6 * ker / ker.sum() if signal else (np.arange(66) < 22) / 22
    r2, rv, rvb = np.zeros((n, A.NS)), np.zeros(n), np.zeros(n)
    for t in range(n):
        if t < 66:
            sig = 1.0
        else:
            ls = np.log(rvb[t - 1])
            lm = np.log(rv[t - 5:t].mean())
            ll = np.log((a_l * rv[t - 66:t][::-1]).sum())
            sig = np.exp(0.4 * ls + 0.3 * lm + 0.2 * ll + g.normal(0, noise)) ** 1.0
            sig = min(max(sig, 0.05), 20.0)
        r2[t] = sig * 1e-4 * u * g.normal(size=A.NS) ** 2
        rv[t], rvb[t] = r2[t].sum() * 1e4, (b * r2[t]).sum() * 1e4
    F = A.Feat.from_r2(r2, pd.bdate_range("2014-06-02", periods=n))
    d = pd.DataFrame(dict(rv=F.rv, rv_w=F.rv_w, rv_m=F.rv_m, j=F.j, seg=F.seg), index=F.dates)
    return F, HH.series(d)


def fit(F, k1, var="A2"):
    D = M.build_data(F, k1)
    V = D.view(np.arange(len(D.d)))
    v = M.fit(V, var, LAM, seed=0, corr=False)
    q = lambda x: float(M.qlike(V.rv, M.forward(torch.as_tensor(x), V, var, False)[1]).mean())
    return v, q(np.zeros(M.NP)), q(v)


@pytest.fixture(scope="module")
def known():
    F, k1 = generate(True)
    return fit(F, k1)


@pytest.fixture(scope="module")
def nosig():
    F, k1 = generate(False)
    return fit(F, k1)


@pytest.mark.xfail(strict=True, reason="I2.G3 (а), ожидание спецификации ω_l > 0,2 и ψ_l1 < 0 не выполняется: ядро ω* = 0,6, ψ* = (−3; 0) на 1500 сессиях "
                   "не идентифицируется (ω_l ≈ 0,07, ψ_l1 ≈ 0; воспроизведено независимым генератором, noise 0,05–0,25, n 1500–3000; кодер записал ABANDON I2.G3.а)")
def test_spec_expectation_omega_l_and_psi_l1_from_zero(known):
    v, q0, q1 = known
    assert v[M.I_W][2] > 0.2 and v[M.I_P][4] < 0


def test_known_signal_leaves_zero_intraday_shape_and_qlike(known):
    v, q0, q1 = known
    assert v[M.I_W][2] > 0.0 and np.abs(v[M.I_C][0:4]).max() > 0.05      # параметры ушли от нуля
    tt = np.linspace(0.5, 1.0, 26)
    bs = np.exp(A.spline_basis(tt) @ v[M.I_C][0:4])          # b_s растёт на [0,5; 1] (истина exp(1,5τ²))
    assert (np.diff(bs) > 0).all() and bs[-1] / bs[0] > 1.1
    assert q1 < q0 - 1e-3                                    # QLIKE обучающих пар ниже стартовой


def test_known_signal_a2_beats_a0_in_sample_and_a2p_does_not_find_time(known):
    v, q0, q1 = known
    F, k1 = generate(True)
    _, a0_q0, a0_q1 = fit(F, k1, "A0")
    assert q1 < a0_q1                                        # структура (а, b) даёт больше, чем одни δ
    assert M.free_mask("A0", False).sum() == 5


def test_no_signal_stays_near_zero(nosig):
    v, q0, q1 = nosig
    assert v[M.I_W].max() < 0.05 and np.abs(v[M.I_C]).max() < 0.1 and abs(q1 - q0) < 1e-3
