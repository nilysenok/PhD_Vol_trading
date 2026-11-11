"""S-R-IMPL-1: обязательные мутации потери R (мутации.md № 144 и спецификация разд. 6) — каждая обязана ловиться боевой проверкой `r_loss_chk` или известным ответом."""
import math

import numpy as np
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_loss_chk as K
from models_v2.s18 import r_loss_syn as SY
known_level = K.known_level
ORIG_CLIP, ORIG_RHO, ORIG_LV, ORIG_LOSS = RL.clip_stats, RL.rho, RL.level_arrays, RL.loss_r


def test_honest_code_passes_everything_cheap():
    assert K.check_rho() == [] and known_level(RL.level_arrays) == []


def test_mut_quantile_over_whole_sample_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "clip_stats", lambda x0, T: ORIG_CLIP(x0, np.arange(len(x0))))
    assert K.check_pit() != []


def test_mut_quantile_with_current_pair_is_caught(monkeypatch):
    def mut(x0, T):
        out = np.setdiff1d(np.arange(len(x0)), T)
        return ORIG_CLIP(x0, np.append(T, out[:1])) if len(out) else ORIG_CLIP(x0, T)
    monkeypatch.setattr(RL, "clip_stats", mut)
    assert K.check_pit() != []


def test_mut_no_level_correction_m_one_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "clip_stats", lambda x0, T: (ORIG_CLIP(x0, T)[0], 1.0))
    assert any("балл уровня" in b for b in K.check_nest())


def test_mut_slope_kappa_instead_of_kappa_minus_one_is_caught(monkeypatch):
    def mut(u, kappa):
        lk = math.log(kappa)
        return torch.where(u <= lk, torch.exp(torch.clamp(u, max=lk)) - u - 1.0, (kappa - 1.0 - lk) + kappa * (u - lk))
    monkeypatch.setattr(RL, "rho", mut)
    assert K.check_rho() != []


def test_mut_clip_from_below_is_caught(monkeypatch):
    monkeypatch.setattr(RL, "rho", lambda u, kappa: ORIG_RHO(torch.clamp(u, min=-1.0), kappa))
    assert K.check_rho() != []


def test_mut_detach_in_tail_branch_is_caught_by_gradient_check(monkeypatch):
    def mut(u, kappa):
        lk = math.log(kappa)
        return torch.where(u <= lk, torch.exp(torch.clamp(u, max=lk)) - u - 1.0, (kappa - 1.0 - lk) + (kappa - 1.0) * (u.detach() - lk))
    monkeypatch.setattr(RL, "rho", mut)
    assert K.check_grad() != []


def _patch_lv(monkeypatch, fn):
    monkeypatch.setattr(K, "_CACHE", {})
    monkeypatch.setattr(RL, "level_arrays", fn)


def test_mut_pi_of_day_t_plus_is_caught(monkeypatch):
    def mut(arch, D=None, x=None, w5=None):
        lv = ORIG_LV(arch, D, x, w5)
        lv.pi = np.roll(lv.pi, -1)
        return lv
    _patch_lv(monkeypatch, mut)
    assert known_level(mut) != [] and K.check_nest() != []


def test_mut_pi_one_is_caught(monkeypatch):
    def mut(arch, D=None, x=None, w5=None):
        lv = ORIG_LV(arch, D, x, w5)
        lv.pi = np.ones_like(lv.pi)
        return lv
    _patch_lv(monkeypatch, mut)
    assert known_level(mut) != [] and K.check_nest() != []


def test_mut_beta_on_all_days_is_caught(monkeypatch):
    def mut(arch, D=None, x=None, w5=None):
        import pandas as pd
        w5 = pd.read_parquet(RL.W5G) if w5 is None else w5
        w = w5.copy()
        w["gap"], w["refit_date"] = 1, w["d"]
        return ORIG_LV(arch, D, x, w)
    _patch_lv(monkeypatch, mut)
    assert known_level(mut) != [] and K.check_nest() != []


def test_mut_plain_loss_instead_of_r_is_caught_by_influence_check(monkeypatch):
    monkeypatch.setattr(RL, "loss_r", lambda v, V, arch, lam, c, m, mode="R", corr=True, shift=0.0: ORIG_LOSS(v, V, arch, lam, c, m, "plain", corr, shift))
    assert any(b.startswith("(в)") for b in SY.synth())
