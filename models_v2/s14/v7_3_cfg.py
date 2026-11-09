"""V7-3: конфигурация 8 — Residual MoE с маршрутизатором B и якорем на бустинг (план, раздел 5, строка 8).
F = H·exp(w·ℓ + Σ_{k=1,2} π_k(Z, z)·g_k(z)), g₀ ≡ 0; ℓ = ln(M/H), M = √(K2·K3) (сохранённые прогнозы бустингов дня d, PIT); w — обучаемый общий скаляр (без ограничения знака, L2 как у прочих параметров), 78 параметров.
Штраф λ_g действует на G = ln(F/H) целиком, то есть включает w·ℓ (`v7_1m_loss.loss`: mean(G²) от выхода модели) — стягивание к HAR-J. Эксперты и решётка λ_g × λ_LB — как у конфигурации 6 (`v7_2_cfg`).
5 seeds, помесячно по закрытым парам t⁺ ≤ d (`v7_1m_wf.walk_forward`), z — от энкодера года переоценки (`v7_1b_cfg.z_by_year`, лист FX), первая оцениваемая дата 2018-01-03.
Прогноз = среднее ln F по seeds; w, π, g_k — среднее по seeds. Данные читаются только при вызове функций."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_cfg as CFG1, v7_1b_sta as STA, v7_1m_feat as FE, v7_1m_fit as FT, v7_1m_forms as FM, v7_2_cfg as C2

SPEC8 = dict(kind="anchor", router="B", K=2)
GRID8 = FT.GRID                                                                   # λ_g ∈ {0,1; 1} × λ_LB ∈ {0; 0,1} — как у 5–7
PARAMS8 = 67 + 10 + 1                                                             # маршрутизатор B 67 + эксперты 10 + w
COLS = ["d", "t_plus", "rv", "H", "ell", "w", "G", "F", "fitted", "pi0", "pi1", "pi2", "g1", "g2"]


def declared_problems() -> list:
    """Нарушения объявленного: вид «якорь», маршрутизатор B, K = 2, решётка 6, 78 параметров. МУТАНТЫ (K = 3, без якоря, другая решётка) подменяют SPEC8 / GRID8."""
    out = []
    if SPEC8 != dict(kind="anchor", router="B", K=2):
        out.append("конфигурация 8: не якорь на M с маршрутизатором B и K = 2")
    if GRID8 != C2.GRID_MOE:
        out.append("решётка 8 не равна решётке 5–7 (λ_g × λ_LB)")
    if FM.n_params(FM.ResMoE("B", 2, anchor=True)) != PARAMS8:
        out.append("число параметров 8 ≠ %d" % PARAMS8)
    return out


def ell_of(tab: pd.DataFrame) -> np.ndarray:
    """ℓ = ln(M/H) даты d: сохранённые прогнозы K2, K3 ДАТЫ d (как в `walk_forward`)."""
    return np.log(np.sqrt(tab["K2"].to_numpy(float) * tab["K3"].to_numpy(float)) / tab["H"].to_numpy(float))


def walk_seed(tab, Zg, z, seed: int) -> dict:
    """Один seed: walk_forward (G, fitted, fits) и w, π, g_k по строкам (NaN вне переоценённых блоков)."""
    sink = dict(keep=True)
    with C2.aux_patch(sink):
        r = CFG1.walk(tab, Zg, z, SPEC8, seeds=(seed,), grid=GRID8)
    n = len(tab)
    pi, gk, w = np.full((n, 3), np.nan), np.full((n, 2), np.nan), np.full(n, np.nan)
    for lo, a in ((k, v) for k, v in sink.items() if k != "keep"):
        m = len(a["pi"])
        pi[lo:lo + m], gk[lo:lo + m], w[lo:lo + m] = a["pi"], a["gk"], float(np.mean([float(x.w) for x in a["raw"][0]]))
    return dict(G=r["G"].to_numpy(float), fitted=r["fitted"].to_numpy(float), fits=r["fits"], pi=pi, gk=gk, w=w)


def assemble(tab: pd.DataFrame, per_seed: list) -> pd.DataFrame:
    """Среднее по seeds: G (→ F = H·exp(G)), w, π, g_k. Строки до первой оцениваемой даты (F = NaN) отбрасываются."""
    mean = lambda key, j=None: np.mean(np.stack([p[key] if j is None else p[key][:, j] for p in per_seed]), axis=0)
    H, G = tab["H"].to_numpy(float), mean("G")
    out = pd.DataFrame({"d": tab.index, "t_plus": tab["t_plus"].to_numpy(), "rv": tab["rv_next"].to_numpy(float), "H": H, "ell": ell_of(tab), "w": mean("w"), "G": G, "F": H * np.exp(G), "fitted": mean("fitted")})
    for j, c in enumerate(C2.PI_COLS):
        out[c] = mean("pi", j)
    for j, c in enumerate(C2.GK_COLS):
        out[c] = mean("gk", j)
    return out[out["F"].notna()].reset_index(drop=True)[COLS]


def run_cfg(tab: pd.DataFrame, Zg: np.ndarray, z_of, seeds=FT.SEEDS) -> tuple:
    """z_of(seed) -> {год: (n, 4)} или (n, 4). -> (таблица, список результатов по seeds)."""
    bad = declared_problems()
    if bad:
        raise ValueError("конфигурация 8: %s" % bad)
    per = [walk_seed(tab, Zg, z_of(s), s) for s in seeds]
    return assemble(tab, per), per


def run_prod(tab: pd.DataFrame, kind: str, load=STA.load, seeds=FT.SEEDS, years=STA.YEARS) -> tuple:
    """Боевой путь: Z = build_Z(tab), z года переоценки — энкодера `kind` (TCN-w5) того же seed."""
    return run_cfg(tab, FE.build_Z(tab).to_numpy(float), lambda s: CFG1.z_by_year(kind, s, tab.index, load, years), seeds)


def refit_block(tab: pd.DataFrame, kind: str, a: int, load=STA.load, seeds=FT.SEEDS) -> dict:
    """Одна переоценка (блок месяца с позиции a) — тот же `WF._refit`, что внутри walk_forward. -> {G, w, pi, gk} (среднее по seeds)."""
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1m_wf as WF
    Zg, pos = FE.build_Z(tab).to_numpy(float), W6.refit_positions(tab.index, WF.FIRST_REFIT)
    b = (pos + [len(tab)])[pos.index(a) + 1]
    res = []
    for s in seeds:
        sink = dict(keep=True)
        with C2.aux_patch(sink):
            g, _ = WF._refit(CFG1.wf_inputs(tab), Zg, CFG1.z_of_year(kind, s, tab.index, tab.index[a].year, load), SPEC8, tab.index[a], a, b, (s,), GRID8)
        res.append((g["G"], sink[a]))
    st = lambda f: np.mean(np.stack([f(g, k) for g, k in res]), axis=0)
    return dict(G=st(lambda g, k: g), w=st(lambda g, k: np.full(len(g), float(np.mean([float(x.w) for x in k["raw"][0]])))), pi=st(lambda g, k: k["pi"]), gk=st(lambda g, k: k["gk"]))
