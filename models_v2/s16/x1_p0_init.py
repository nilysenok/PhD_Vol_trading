"""X1-пилот, лист P0, G3: выход из нулевой инициализации у MoE V7 (конфигурация 6: Residual MoE K = 2, маршрутизатор B). Код V7 не меняется: берутся `v7_1m_forms/fit/loss/wf` и `v7_2_cfg`.
1) В точке инициализации (эксперты a, b = 0 ⇒ G ≡ 0) градиент потери по параметрам маршрутизатора и экспертов: автоградиент torch и центральная разность. Потеря раскладывается:
   data = QLIKE-член + λ_g·mean(G²); pen = L2 + λ_LB-штраф + λ_s-сглаживание. Для маршрутизатора ∂data/∂логит_j = ∂L/∂G · π_j·(g_j − Σπ g) = 0 при g ≡ 0 — проверяется точным нулём.
2) Повтор зарегистрированной конфигурации 6 (`fit_config`: 5 seeds × решётка 4 точки, затем переобучение) на реальных переоценках: нормы параметров маршрутизатора и экспертов после обучения, разброс π, доля градиента данных на маршрутизаторе.
Прогнозы НЕ пишутся, `walk_forward` и файлы V7 не вызываются на запись; результат — список строк таблицы."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from models_v2.s14 import v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_loss as LS, v7_1m_wf as WF

SPEC6 = dict(kind="moe", router="B", K=2)
ROUTER, EXPERT = ("proj.weight", "out.weight", "out.bias"), ("a", "b")
EPS = 1e-6


def split_loss(model, b: dict, lg: float, lb: float) -> tuple:
    """-> (data, pen, total) как тензоры: data = QLIKE-член + λ_g·mean(G²), pen = total − data."""
    G = model(b["Zg"], b["z"], b["ell"])[0]
    data = LS.data_term(b["H"], b["rv"], G) + lg * torch.mean(G ** 2)
    total = LS.loss(model, b, lg, lb)
    return data, total - data, total


def grads(model, fn) -> dict:
    ps = dict(model.named_parameters())
    g = torch.autograd.grad(fn(), list(ps.values()), allow_unused=True)
    return {k: (torch.zeros_like(p) if v is None else v) for (k, p), v in zip(ps.items(), g)}


def numeric(model, fn) -> dict:
    """Центральная разность по КАЖДОМУ элементу каждого параметра (шаг 1e-6, float64)."""
    out = {}
    for k, p in model.named_parameters():
        g = torch.zeros_like(p)
        flat, gf = p.data.view(-1), g.view(-1)
        for i in range(flat.numel()):
            old = float(flat[i])
            flat[i] = old + EPS
            fp = float(fn())
            flat[i] = old - EPS
            fm = float(fn())
            flat[i] = old
            gf[i] = (fp - fm) / (2 * EPS)
        out[k] = g
    return out


def group_norm(g: dict, names) -> float:
    return float(torch.sqrt(sum((g[k] ** 2).sum() for k in names)))


def init_rows(date, tr: dict, seeds=FT.SEEDS, grid=FT.GRID) -> list:
    out = []
    for s in seeds:
        for lg, lb in grid:
            m = FT.make(SPEC6, s)
            d, p, t = (grads(m, lambda i=i: split_loss(m, tr, lg, lb)[i]) for i in range(3))
            num = numeric(m, lambda: split_loss(m, tr, lg, lb)[2])
            diff = max(float((t[k] - num[k]).abs().max()) for k in t)
            with torch.no_grad():
                pi = m(tr["Zg"], tr["z"], tr["ell"])[1]
            for name, grp in (("router", ROUTER), ("experts", EXPERT)):
                n0 = float(torch.sqrt(sum((dict(m.named_parameters())[k] ** 2).sum() for k in grp)))
                out.append(dict(stage="init", model="V7 cfg6 %s" % date, group=name, seed=s, grad_data=group_norm(d, grp), grad_pen=group_norm(p, grp), grad_total=group_norm(t, grp),
                                num_vs_auto=diff, extra=float((pi - 1.0 / 3).abs().mean()), norm_param=n0, norm_init=n0, sd_pi=float(pi.std(0).mean()), note="λ_g=%g λ_LB=%g; extra = среднее |π−1/3| в точке инициализации" % (lg, lb)))
    return out


def inputs(tab, Zg, z, d0):
    """Батчи обучения / валидации / всех закрытых пар на переоценке d0 — те же шаги, что внутри `WF._refit` (закрытые пары t⁺ ≤ d0, валидация — последние 126)."""
    H, rv, tp = tab["H"].to_numpy(float), tab["rv_next"].to_numpy(float), tab["t_plus"].to_numpy("datetime64[ns]")
    ell = np.log(np.sqrt(tab["K2"].to_numpy(float) * tab["K3"].to_numpy(float)) / H)
    us = WF._usable(Zg, z)
    cl = WF.closed_rows(tp, d0)
    cl = cl[us[cl]]
    va = WF.val_rows(tp, d0, us)
    tr = np.setdiff1d(cl, va)
    prep = WF._prep_fn(Zg, z, ell, H, rv, np.union1d(tr, va), False)
    return prep(tr), prep(va)


def trained_rows(date, tr: dict, va: dict, s: int) -> list:
    """Один seed = одна ветка боевого пути (`v7_2_cfg.refit_block`: `_refit(..., (s,), GRID)` со своим z энкодера того же seed)."""
    r = FT.fit_config(SPEC6, tr, va, (s,))
    full, out = FT.cat(tr, va), []
    for m in r["models"]:
        m0 = FT.make(SPEC6, s)
        d, p, t = (grads(m, lambda i=i: split_loss(m, full, *r["lam"])[i]) for i in range(3))
        with torch.no_grad():
            pi = m(full["Zg"], full["z"], full["ell"])[1]
        sd0 = {k: float(v.norm()) for k, v in m0.named_parameters()}
        from models_v2.s16.x1_p0_dyn import pair_stats
        ps = pair_stats(m, full)
        for name, grp in (("router", ROUTER), ("experts", EXPERT)):
            n1, n0 = float(torch.sqrt(sum((dict(m.named_parameters())[k] ** 2).sum() for k in grp))), float(np.sqrt(sum(sd0[k] ** 2 for k in grp)))
            out.append(dict(stage="trained", model="V7 cfg6 %s" % date, group=name, seed=s, grad_data=group_norm(d, grp), grad_pen=group_norm(p, grp), grad_total=group_norm(t, grp), num_vs_auto=np.nan,
                            extra=float(pi.mean(0).sub(1.0 / 3).abs().max()), norm_param=n1, norm_init=n0, sd_pi=float(pi.std(0).mean()),
                            corr_g=ps["corr_g"], mean_abs_g12=ps["mean_abs_diff"], dist_experts=ps["dist_params"], argmax_switch=ps["argmax_switch"],
                            note="после обучения; λ=(%g;%g); эпохи %d; extra = max_k |средний π_k − 1/3|; sd_pi = σ_π по строкам обучения" % (*r["lam"], r["epochs"][0])))
    return out


FIT_MONTHS = ("2018-01", "2018-07", "2020-01", "2022-01")        # первая переоценка (точка инициализации + обучение), затем три даты по заданию


def init_table(months=FIT_MONTHS, seeds=FT.SEEDS, train_months=FIT_MONTHS[1:]) -> list:
    """Боевые входы cfg6 (Z, z энкодера TCN-w5 года и seed переоценки); init-строки — на первой переоценке, trained-строки — на `train_months`. Ничего не пишет."""
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1m_data as DATA, v7_1m_feat as FE
    tab, kind = DATA.load_tab(), C3.kind3(CR.read_cp())
    Zg, pos, out = FE.build_Z(tab).to_numpy(float), W6.refit_positions(tab.index, WF.FIRST_REFIT), []
    for mo in months:
        a = next(i for i in pos if tab.index[i] >= pd.Timestamp(mo + "-01"))
        d0 = tab.index[a]
        for s in seeds:
            tr, va = inputs(tab, Zg, CFG1.z_of_year(kind, s, tab.index, d0.year), d0)
            if mo == months[0]:
                out += init_rows(str(d0.date()), tr, (s,))
            if mo in train_months:
                out += trained_rows(str(d0.date()), tr, va, s)
    return out


def pi_cfg6() -> dict:
    """Сохранённые π конфигурации 6 (ТОЛЬКО чтение parquet V7): среднее |π−1/3| и σ_π по датам с обученной головой, число дней."""
    from models_v2.s14 import v7_0_io as IO
    t = pd.read_parquet(IO.OUT / "2" / "cfg6.parquet")
    t = t[t["fitted"] > 0]
    p = t[["pi0", "pi1", "pi2"]].to_numpy(float)
    return dict(n=len(t), mean_abs_dev=float(np.abs(p - 1 / 3).mean()), max_abs_dev=float(np.abs(p - 1 / 3).max()), sd_pi=float(p.std(0).mean()), mean_pi=p.mean(0).tolist())
