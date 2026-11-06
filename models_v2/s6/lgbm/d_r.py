"""Волна D, поток LGBM: стадии r1 (режимные признаки внутри модели) и r4 (переключение двух конфигураций по причинному режиму). Правила — RD3, RD4."""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s4 import rows
from models_v2.s6.lgbm import d_comp, d_reg, d_run, d_world, grid
from models_v2.s6.lgbm.d_win import Tm

YEARS = range(2017, 2027)
R4A = ("D_w0", "D_w1008")    # (спокойно, стресс): расширяющееся / rolling 1008


def r1_items(W: dict) -> list:
    spec, hp = d_run.base_cfg(W)
    return [d_world.DItem(n, d_reg.with_regime(spec, W["F"].dates, which, n), hp, Tm(), "regime",
                          "R1 режимные признаки в модель: %s (только данные <= t); база C0_inh_F4, расширяющееся окно, ежедневно" % "+".join(which))
            for n, which in d_reg.R1.items()]


def stage_r1(W: dict) -> None:
    its = r1_items(W)
    d_world.run_batch(W, its)
    print("trials добавлено:", d_world.register([i.name for i in its]), d_world.checkpoint()["used_stream_d"], flush=True)


def _derived(name: str, fc, crit: dict, picks: dict, desc: str, sw: dict) -> None:
    st0 = d_world.get("D_w0")[1]
    tid = grid.trial_id("derived", {"rule": name}, [], "", {"cands": list(d_run.CAND), "picks": {str(k): v for k, v in picks.items()}})
    d_world.save(name, fc, {k: v for k, v in st0.items() if k not in ("key", "crit")},
                 dict(key="derived", trial_id=tid, crit=crit, hp_by_year={str(Y): v for Y, v in picks.items()}, tm={"switch": name}, direction="regime",
                      secs_total=0.0, desc=desc, switch=sw))


def switch_stats(fcs: dict, act: np.ndarray, calm: dict, stress: dict, st_grid, years) -> dict:
    """Число смен режима и активной конфигурации; доля дней, когда выбранная конфигурация хуже второй (ex post, описательно)."""
    q ={n: np.asarray(metrics.qlike(f["rv"], f["sigma2"]), float) for n, f in fcs.items()}
    worse, differ = 0, 0
    for i, (n, Y) in enumerate(zip(act, years)):
        a, b = calm[int(Y)], stress[int(Y)]
        if a != b:
            differ += 1
            worse += int(q[n][i] > q[b if n == a else a][i])
    s = np.nan_to_num(st_grid, nan=0.0)
    return {"n_regime_flips": int((s[1:] != s[:-1]).sum()), "n_config_changes": int((act[1:] != act[:-1]).sum()), "days_configs_differ": differ,
            "share_chosen_worse_ex_post": worse / differ if differ else float("nan")}


def stage_r4(W: dict) -> None:
    F, g = W["F"], W["grid"]
    st_rows = d_reg.stress_rows(F.dates)
    st_grid, years = st_rows[g], np.array([rows.year_of(F, int(d)) for d in g])
    fcs = {n: d_world.get(n)[0] for n in d_run.CAND}
    dfs = {n: d_world.folds_of(n) for n in d_run.CAND}
    pa = ({Y: R4A[0] for Y in YEARS}, {Y: R4A[1] for Y in YEARS})
    pb = d_comp.pick_by_regime(d_comp.regime_scores(dfs, st_rows), list(d_run.CAND), YEARS)
    for name, (calm, stress), desc in (("D_R4a_switch", pa, "R4a: стресс Г4 -> rolling 1008, спокойно -> расширяющееся (правило блока, фиксировано до счёта)"),
                                       ("D_R4b_switch_pick", pb, "R4b: окно по режиму выбирается по году Y из 8 по средним фолдовым QLIKE режима (цели < 01.01.Y); потолок")):
        fc, act = d_comp.switch_series(fcs, calm, stress, st_grid, years)
        crit = d_comp.crit_of(d_comp.switch_folds(dfs, calm, stress, st_rows), name)
        picks = {Y: "%s|%s" % (calm[Y], stress[Y]) for Y in YEARS}
        _derived(name, fc, crit, picks, desc, switch_stats(fcs, act, calm, stress, st_grid, years))
        print(name, "crit_pen %.6f" % crit["crit_pen"], d_world.get(name)[1]["switch"], flush=True)
    print("trials добавлено:", d_world.register(["D_R4a_switch", "D_R4b_switch_pick"]), d_world.checkpoint()["used_stream_d"], flush=True)
