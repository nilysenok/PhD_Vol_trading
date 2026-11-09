"""X1-пилот, лист P0, Ж-242 п. 3: ДИНАМИКА обучения MoE V7 cfg6 по шагам оптимизатора и симметрия экспертов. Код V7 не меняется; прогнозы не пишутся; результат — строки `P0_DYN.csv` и итоги по парам экспертов.
По шагам (Adam, lr и потеря V7, полный батч обучающих закрытых пар, один seed и одна точка решётки): нормы градиента (данные / штрафы / полный) по маршрутизатору и по каждому эксперту, нормы параметров, среднее и σ π_k по строкам.
Расходятся ли эксперты: корреляция и среднее |g_1 − g_2| на обучающих строках, расстояние параметров экспертов, доля смен argmax π по порядку дат."""
from __future__ import annotations

import numpy as np
import torch

from models_v2.s14 import v7_1m_fit as FT, v7_1m_loss as LS
from models_v2.s16 import x1_p0_init as I

STEP_EVERY, N_STEPS = 10, 300


def expert_norms(m) -> dict:
    return {"router": float(torch.sqrt(sum((p ** 2).sum() for k, p in m.named_parameters() if k in I.ROUTER))),
            "expert1": float(torch.sqrt(m.a[0] ** 2 + (m.b[0] ** 2).sum())), "expert2": float(torch.sqrt(m.a[1] ** 2 + (m.b[1] ** 2).sum()))}


def expert_grad_norms(m, b: dict, lg: float, lb: float) -> dict:
    """Нормы градиента данных / штрафов / полного по группам (маршрутизатор, эксперт 1, эксперт 2)."""
    out = {}
    for i, part in enumerate(("data", "pen", "total")):
        g = I.grads(m, lambda i=i: I.split_loss(m, b, lg, lb)[i])
        gn = {"router": I.group_norm(g, I.ROUTER), "expert1": float(torch.sqrt(g["a"][0] ** 2 + (g["b"][0] ** 2).sum())), "expert2": float(torch.sqrt(g["a"][1] ** 2 + (g["b"][1] ** 2).sum()))}
        out.update({(k, part): v for k, v in gn.items()})
    return out


def pair_stats(m, b: dict) -> dict:
    """Расхождение экспертов на строках b: корреляция и среднее |g1−g2|, расстояние параметров, доля смен argmax π по порядку строк."""
    with torch.no_grad():
        g = m.experts(b["z"]).numpy()
        pi = m.pi(b["Zg"], b["z"]).numpy()
    am = pi.argmax(1)
    return dict(corr_g=float(np.corrcoef(g[:, 0], g[:, 1])[0, 1]) if g.std(0).min() > 0 else float("nan"), mean_abs_diff=float(np.abs(g[:, 0] - g[:, 1]).mean()),
                dist_params=float(torch.sqrt((m.a[0] - m.a[1]) ** 2 + ((m.b[0] - m.b[1]) ** 2).sum())), argmax_switch=float((am[1:] != am[:-1]).mean()), pi_mean=pi.mean(0).tolist(), pi_sd=pi.std(0).tolist())


def trajectory(date: str, b: dict, seed: int = 0, lam=(0.1, 0.1), steps: int = N_STEPS, every: int = STEP_EVERY) -> list:
    m = FT.make(I.SPEC6, seed)
    opt = torch.optim.Adam(m.parameters(), lr=FT.LR)
    rows = []
    for st in range(steps + 1):
        if st % every == 0:
            gn, nn_, ps = expert_grad_norms(m, b, *lam), expert_norms(m), pair_stats(m, b)
            for grp in ("router", "expert1", "expert2"):
                rows.append(dict(date=date, seed=seed, lam_g=lam[0], lam_lb=lam[1], step=st, group=grp, grad_data=gn[(grp, "data")], grad_pen=gn[(grp, "pen")], grad_total=gn[(grp, "total")], norm_param=nn_[grp],
                                 pi0_mean=ps["pi_mean"][0], pi1_mean=ps["pi_mean"][1], pi2_mean=ps["pi_mean"][2], pi_sd=float(np.mean(ps["pi_sd"])), corr_g=ps["corr_g"], mean_abs_g12=ps["mean_abs_diff"],
                                 dist_experts=ps["dist_params"], argmax_switch=ps["argmax_switch"]))
        if st == steps:
            break
        opt.zero_grad()
        LS.loss(m, b, *lam).backward()
        opt.step()
    return rows


def synthetic_batch(n: int = 400, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    H = torch.exp(0.3 * rn(n))
    z = rn(n, 4)
    rv = H * torch.exp(0.4 * torch.tanh(z[:, 0]) + 0.2 * rn(n))      # известная связь: RV зависит от z₀
    return FT.batch(rn(n, 9), z, 0.3 * rn(n), H, rv)


def symmetry_dist(b: dict, router_zero: bool, seed: int = 0, steps: int = 100, lam=(0.1, 0.1)) -> float:
    """Расстояние параметров двух экспертов V7 после `steps` шагов Adam. Эксперты стартуют с ОДИНАКОВЫХ нулей; router_zero = выходной слой маршрутизатора обнулён (π ровно равномерные) —
    тогда градиенты экспертов идентичны и они остаются копиями (расстояние 0); с обычным случайным маршрутизатором V7 симметрия ломается через π_k ≠ π_j."""
    m = FT.make(I.SPEC6, seed)
    if router_zero:
        with torch.no_grad():
            m.out.weight.zero_()
            m.out.bias.zero_()
    opt = torch.optim.Adam(m.parameters(), lr=FT.LR)
    for _ in range(steps):
        opt.zero_grad()
        LS.loss(m, b, *lam).backward()
        opt.step()
    return pair_stats(m, b)["dist_params"]


def dyn_table(months=("2018-01", "2018-07", "2020-01", "2022-01"), seed: int = 0) -> list:
    import pandas as pd
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1m_data as DATA, v7_1m_feat as FE, v7_1m_wf as WF
    tab, kind = DATA.load_tab(), C3.kind3(CR.read_cp())
    Zg, pos, out = FE.build_Z(tab).to_numpy(float), W6.refit_positions(tab.index, WF.FIRST_REFIT), []
    for mo in months:
        a = next(i for i in pos if tab.index[i] >= pd.Timestamp(mo + "-01"))
        d0 = tab.index[a]
        tr, va = I.inputs(tab, Zg, CFG1.z_of_year(kind, seed, tab.index, d0.year), d0)
        out += trajectory(str(d0.date()), tr, seed)
    return out
