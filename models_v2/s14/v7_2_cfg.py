"""V7-2a: конфигурации 4–7 (план п. 3–5). Энкодер — TCN-w5 (`1b.choice3`), при переоценке в году Y z всех обучающих пар и прогнозов блока — от энкодера года Y, seed s ∈ 0–4 (`v7_1b_cfg.z_by_year`, версия 2, лист FX); Z — 9 признаков режима (`v7_1m_feat.build_Z`), стандартизация по закрытым парам внутри `walk_forward`.
 4 — плоский корректор g(z, Z) = c·tanh(MLP 13→5→1), 76 параметров; решётка — λ_g ∈ {0,1; 1} (λ_LB не применяется: маршрутизатора нет);
 5 / 6 / 7 — Residual MoE K = 2, маршрутизатор A (Z) / B (Z, z) / C (z); эксперты g_k(z) = c·tanh(a_k + b_kᵀz) и решётка λ_g × λ_LB (4 точки) ОДИНАКОВЫ — различается только `router`.
Walk-forward — `v7_1m_wf.walk_forward` без изменений (помесячно, закрытые пары t⁺ ≤ d, валидация 126). Прогноз = среднее ln F по 5 seeds; π и g_k — среднее по seeds. Веса и эксперты снимаются обёрткой над `WF._refit` (G не меняется).
Данные читаются только при вызове функций."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import torch

from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_sta as STA, v7_1m_feat as FE, v7_1m_fit as FT, v7_1m_forms as FM, v7_1m_wf as WF

IDS = ("4", "5", "6", "7")
GRID_FLAT = tuple((lg, 0.0) for lg in (0.1, 1.0))                  # λ_LB не применяется
GRID_MOE = FT.GRID
SPECS = {"4": dict(kind="flat"), **{c: dict(kind="moe", router=r, K=2) for c, r in zip("567", "ABC")}}
GRIDS = {"4": GRID_FLAT, "5": GRID_MOE, "6": GRID_MOE, "7": GRID_MOE}
PI_COLS, GK_COLS = ("pi0", "pi1", "pi2"), ("g1", "g2")


def declared_problems() -> list:
    """Нарушения объявленного: у 5–7 одинаковые эксперты (K, форма) и одинаковая решётка λ_g × λ_LB; у 4 — только λ_g. МУТАНТ «разная решётка у C» подменяет GRIDS['7']."""
    out = []
    ex = {c: {k: v for k, v in SPECS[c].items() if k != "router"} for c in "567"}
    if not (ex["5"] == ex["6"] == ex["7"] and [SPECS[c]["router"] for c in "567"] == ["A", "B", "C"]):
        out.append("эксперты 5–7 не одинаковы или маршрутизаторы не A/B/C")
    if not (GRIDS["5"] == GRIDS["6"] == GRIDS["7"] == FT.GRID):
        out.append("решётка 5–7 не одна и не равна λ_g × λ_LB плана")
    if not (SPECS["4"] == dict(kind="flat") and sorted(g[0] for g in GRIDS["4"]) == [0.1, 1.0] and all(g[1] == 0.0 for g in GRIDS["4"])):
        out.append("конфигурация 4: не плоский корректор или решётка не λ_g ∈ {0,1; 1} без λ_LB")
    return out


def _aux(models, bt: dict):
    """π (n, 3) и g_k (n, 2) блока, среднее по моделям; плоский корректор — None."""
    with torch.no_grad(), FM.deterministic():
        outs = [m(bt["Zg"], bt["z"], bt["ell"]) for m in models]
        if outs[0][1] is None:
            return None
        pi = torch.stack([o[1] for o in outs]).mean(dim=0).numpy()
        gk = torch.stack([m.experts(bt["z"]) for m in models]).mean(dim=0).numpy()
    return dict(pi=pi, gk=gk)


def refit_with_aux(refit, sink: dict):
    """Обёртка над `WF._refit`: результат (res, nrow) тот же; веса и эксперты блока — в sink[lo]; с sink['keep'] — ещё и (модели, батч) для тестов."""
    def wrapped(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid):
        last, orig = [], FT.ensemble_lnF

        def ens(models, bt):
            last[:] = [(models, bt)]                                   # последний вызов в _refit — прогноз блока (прежние — валидация)
            return orig(models, bt)
        FT.ensemble_lnF = ens
        try:
            res, nrow = refit(tab_cols, Zg, z, spec, d0, lo, b, seeds, grid)
        finally:
            FT.ensemble_lnF = orig
        if res is not None:
            a = _aux(*last[0])
            if a is not None:
                sink[lo] = dict(a, raw=last[0] if sink.get("keep") else None)
        return res, nrow
    return wrapped


@contextlib.contextmanager
def aux_patch(sink: dict):
    orig = WF._refit
    WF._refit = refit_with_aux(orig, sink)
    try:
        yield
    finally:
        WF._refit = orig


def walk_seed(cfg: str, tab, Zg, z, seed: int, keep: bool = False) -> dict:
    """Один seed: walk_forward (G, fitted, fits) и π, g_k по строкам (NaN вне переоценённых блоков)."""
    sink = dict(keep=keep)
    with aux_patch(sink):
        r = CFG1.walk(tab, Zg, z, SPECS[cfg], seeds=(seed,), grid=GRIDS[cfg])          # z — {год: z энкодера года} (боевой путь) или массив (синтетика)
    n = len(tab)
    pi, gk = np.full((n, 3), np.nan), np.full((n, 2), np.nan)
    for lo, a in ((k, v) for k, v in sink.items() if k != "keep"):
        pi[lo:lo + len(a["pi"])], gk[lo:lo + len(a["gk"])] = a["pi"], a["gk"]
    return dict(G=r["G"].to_numpy(float), fitted=r["fitted"].to_numpy(float), fits=r["fits"], pi=pi, gk=gk, raw=[v["raw"] for k, v in sink.items() if k != "keep"])


def assemble(cfg: str, tab: pd.DataFrame, per_seed: list) -> pd.DataFrame:
    """Среднее по seeds: G (→ F = H·exp(G)), fitted, π, g_k. Строки до первой оцениваемой даты (F = NaN) отбрасываются."""
    G = np.mean(np.stack([p["G"] for p in per_seed]), axis=0)
    H = tab["H"].to_numpy(float)
    out = pd.DataFrame({"d": tab.index, "t_plus": tab["t_plus"].to_numpy(), "rv": tab["rv_next"].to_numpy(float), "H": H, "G": G, "F": H * np.exp(G),
                        "fitted": np.mean(np.stack([p["fitted"] for p in per_seed]), axis=0)})
    if SPECS[cfg]["kind"] == "moe":
        for j, c in enumerate(PI_COLS):
            out[c] = np.mean(np.stack([p["pi"][:, j] for p in per_seed]), axis=0)
        for j, c in enumerate(GK_COLS):
            out[c] = np.mean(np.stack([p["gk"][:, j] for p in per_seed]), axis=0)
    return out[out["F"].notna()].reset_index(drop=True)


def run_cfg(cfg: str, tab: pd.DataFrame, Zg: np.ndarray, z_of, seeds=FT.SEEDS, keep: bool = False) -> tuple:
    """z_of(seed) -> (n, 4) сырой z. -> (таблица, список результатов по seeds)."""
    bad = declared_problems()
    if bad or cfg not in IDS:
        raise ValueError("конфигурация %r: %s" % (cfg, bad or "не из 4–7"))
    per = [walk_seed(cfg, tab, Zg, z_of(s), s, keep) for s in seeds]
    return assemble(cfg, tab, per), per


def run_prod(cfg: str, tab: pd.DataFrame, kind: str, load=STA.load, seeds=FT.SEEDS, years=STA.YEARS) -> tuple:
    """Боевой путь: Z = build_Z(tab), z года d энкодера `kind` (TCN-w5) того же seed."""
    return run_cfg(cfg, tab, FE.build_Z(tab).to_numpy(float), lambda s: CFG1.z_by_year(kind, s, tab.index, load, years), seeds)


def refit_block(cfg: str, tab: pd.DataFrame, kind: str, a: int, load=STA.load, seeds=FT.SEEDS, years=STA.YEARS) -> dict:
    """Одна переоценка (блок месяца с позиции a) — тот же `WF._refit`, что внутри walk_forward. -> {G, pi, gk} (среднее по seeds)."""
    from models_v2.s13 import v6_1_wf as W6
    Zg, pos = FE.build_Z(tab).to_numpy(float), W6.refit_positions(tab.index, WF.FIRST_REFIT)
    b = (pos + [len(tab)])[pos.index(a) + 1]
    Gs, sinks = [], []
    for s in seeds:
        sink = {}
        with aux_patch(sink):
            res, _ = WF._refit(CFG1.wf_inputs(tab), Zg, CFG1.z_of_year(kind, s, tab.index, tab.index[a].year, load), SPECS[cfg], tab.index[a], a, b, (s,), GRIDS[cfg])
        Gs.append(res["G"])
        sinks.append(sink.get(a))
    out = dict(G=np.mean(np.stack(Gs), axis=0))
    if sinks[0] is not None:
        out.update(pi=np.mean(np.stack([k["pi"] for k in sinks]), axis=0), gk=np.mean(np.stack([k["gk"] for k in sinks]), axis=0))
    return out
