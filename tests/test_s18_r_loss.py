"""S-R-IMPL-1: потеря R, PIT-квантиль, поправка уровня — известные ответы и боевые проверки (L.G1–L.G4). Автор тестов — автор кода (по поручению)."""
import math

import numpy as np
import pandas as pd
import pytest
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_loss_chk as K
from models_v2.s18 import r_loss_syn as SY


def test_level_arrays_known_answer():
    assert K.known_level(RL.level_arrays) == []


def test_level_arrays_a_pairs_without_x1():
    d = pd.date_range("2020-01-01", periods=3)
    x = pd.DataFrame(dict(d=d, pi=[0.3, 0.4, 0.5], M=[2.0, 3.0, 4.0]))
    w5 = pd.DataFrame(dict(d=d, gap=[1, 1, 1], beta=[0.1, 0.2, 0.3], refit_date=d))

    class D:
        pass
    D.d = np.array(["2019-12-30", "2020-01-02", "2020-01-05"], "datetime64[ns]")
    lv = RL.level_arrays("A", D, x, w5)
    assert list(lv.has) == [0, 1, 0] and list(lv.pi) == [1.0, 0.4, 1.0] and list(lv.bg) == [0.0, 0.2, 0.0] and list(lv.lnb) == [0.0, math.log(3.0), 0.0]


def test_clip_stats_known_answer_and_only_T():
    x0 = np.arange(1.0, 101.0)
    c, m = RL.clip_stats(x0, np.arange(50))
    assert c == pytest.approx(49.51) and m == pytest.approx(np.minimum(np.arange(1.0, 51.0), 49.51).mean())
    x1 = x0.copy()
    x1[60:] = 1e6
    assert RL.clip_stats(x1, np.arange(50)) == (c, m)


def test_rho_known_values_and_convexity():
    kap = 3.0
    lk = math.log(kap)
    r = RL.rho(torch.tensor([0.0, lk, lk + 1.0], dtype=torch.float64), kap).numpy()
    assert r[0] == pytest.approx(0.0, abs=1e-15) and r[1] == pytest.approx(kap - lk - 1.0) and r[2] == pytest.approx(kap - lk - 1.0 + (kap - 1.0))
    u = np.linspace(-3, 6, 400)
    v = RL.rho(torch.as_tensor(u), kap).numpy()
    assert (np.diff(v, 2) >= -1e-12).all()


def test_zero_signal_gives_zero_theta():
    """Нулевой сигнал без шума: RV = F(0), x0 ≡ 1, m = 1 ⇒ θ̂ = 0; выброс (x = 40) θ̂ R почти не двигает."""
    D, lv = SY.synth_data(500, 3, np.zeros(6), sig=0.0, jump=0.0)
    for mode in ("R", "plain"):
        assert np.abs(SY.fit_pair(D, lv, 1e-2, mode)).max() < 1e-8


def test_outlier_pair_moves_r_less_than_a_fifth_of_plain():
    assert SY.synth() == []


def test_loss_at_zero_equals_mean_rho_of_known_level():
    """Потеря при θ = 0 без штрафа = mean ρ_κ(ln(x0/m)) — известный ответ из x0 (не из тех же функций обучения)."""
    D, lv = SY.synth_data(800, 5, np.zeros(6))
    V = RL.view(D, lv, np.arange(800))
    x0 = RL.x0_of(V, "CK")
    c, m = RL.clip_stats(x0, np.arange(800))
    exp = np.where(x0 <= c, x0 / m - np.log(x0 / m) - 1.0, (c / m - 1 - math.log(c / m)) + (c / m - 1) * (np.log(x0 / m) - math.log(c / m))).mean()
    got = float(RL.loss_r(torch.zeros(6, dtype=torch.float64), V, "CK", 0.0, c, m, "R"))
    assert got == pytest.approx(exp, rel=1e-12)


def test_real_nest_grad_pit_pass():
    assert K.check_nest() == [] and K.check_grad() == [] and K.check_pit() == []
