"""S-R-IMPL-2: контекст прохода R — данные архитектуры (пары, уровень F, x0 = RV/F(0), ln F(0)). Вынесено из `r_walk` (≤ 200 строк); API: `Ctx(arch, D, lv, x, gap, dp)`, `make_ctx(arch)` (боевые данные A2 или C-K)."""
from __future__ import annotations

import numpy as np
import torch

from models_v2.s18 import a_layers as AL
from models_v2.s18 import a_model as AM
from models_v2.s18 import a_walk as AW
from models_v2.s18 import r_loss as RL


class Ctx:
    def __init__(self, arch: str, D, lv, x, gap, dp=None, lv_train=None):
        self.arch, self.D, self.lv, self.x, self.gap, self.lv_train = arch, D, lv, x, gap, lv_train
        self.dp = np.arange(len(x)) if dp is None else dp
        V = RL.view(D, lv, np.arange(len(lv.pi)))
        self.x0, self.rv = RL.x0_of(V, arch), V.rv.numpy()
        # S-D (Ж-256): потеря обучения на другом уровне (lv_train); None ⇒ тот же уровень F, x0_train is x0
        self.x0_train = self.x0 if lv_train is None else RL.x0_of(RL.view(D, lv_train, np.arange(len(lv_train.pi))), arch)
        self.lnf0 = RL.ln_f(RL.lnc_of(torch.zeros(RL.n_par(arch), dtype=torch.float64), V, arch), V).numpy()     # тем же путём, что прогноз: при θ̂ = 0 точно G = 0

    def view(self, sel):
        return RL.view(self.D, self.lv, sel)

    def view_train(self, sel):
        return self.view(sel) if self.lv_train is None else RL.view(self.D, self.lv_train, sel)

    def corr(self, sel) -> bool:
        return self.arch == "CK" or int(self.D.has8[np.asarray(sel)].sum()) >= AW.MIN_CFG


def make_ctx(arch: str, train_level: str = "F") -> Ctx:
    if train_level not in ("F", "C") or (train_level == "C" and arch != "CK"):
        raise ValueError("train_level 'C' — только для CK; допустимы 'F', 'C': %r, %r" % (arch, train_level))
    if arch == "A":
        D, (x, gap) = AM.load_data(), AL.inputs()
        return Ctx(arch, D, RL.level_arrays(arch, D, x), x, gap, AW._day_pos(D, x))
    from models_v2.s18 import cm_walk as CW
    x, gap, D = CW.inputs("CK")
    n, lv_train = len(x), None
    if train_level == "C":    # π := 1, ln B := 0, β·g := 0 — только в потере обучения
        lv_train = RL.Lv(np.ones(n), np.zeros(n), np.zeros(n), np.ones(n))
    return Ctx(arch, D, RL.level_arrays(arch, D, x), x, gap, lv_train=lv_train)
