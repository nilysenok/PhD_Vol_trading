"""S-D-EVAL (`S_D_SPEC.md`, разд. 2.4, 5): механизм абляций — сдвиг уровня δ = ln(C/cfg8), месяцы θ = 0 и λ*, разложение «замороженный π + реакция слоя», разложение при фиксированных θ («только уровень / память / затухание»), предсказания P-m, P-f, P-f2.
Только расчёт по сохранённым рядам (`R/rck`, `D/rckm`, `D/rckf`): слои X1 и W5 над C при фиксированных θ — прежним кодом `cm_layers.layers`; обучения нет. Пороги предсказаний — константы `d_eval_tab` (из спецификации).
Публичное API: `mech(frames, x, gap, w5f) -> DataFrame(section, series, item, value)`; `predictions(mech) -> DataFrame` — P-m, P-f, P-f2 выведены из строк таблицы `mech`; `TH` — колонки θ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_layers as CL
from models_v2.s18 import cm_model as CM
from models_v2.s18 import d_eval_tab as T

TH = ("gam_c", "gam_p", "th_c0", "th_p0", "th_c1", "th_p1")
PARTS = {"full": (0, 1, 2, 3, 4, 5), "level_only": (2, 3), "memory_only": (0, 1), "decay_only": (4, 5)}
VARS = ("S-RCK", "S-RCKm", "S-RCKf")


def _arrays(x) -> dict:
    V = CM.build_data(x, CF.load_feat()).view(np.arange(len(x)))
    return {k: getattr(V, k).numpy() for k in ("c8", "xc", "xp", "zc", "zp", "sc", "sp")}


def mult(a: dict, th) -> np.ndarray:
    ec, ep = th[:, 0] * a["xc"] + th[:, 2] + th[:, 4] * a["zc"], th[:, 1] * a["xp"] + th[:, 3] + th[:, 5] * a["zp"]
    return 1.0 + a["sc"] * np.expm1(ec) + a["sp"] * np.expm1(ep)


def _dpct(la, lb) -> float:
    return 100.0 * float(la.mean() - lb.mean()) / float(lb.mean())


def fixed_theta(x, gap, frames: dict, lw) -> list:
    a, rows, rv = _arrays(x), [], x["rv"].to_numpy(float)
    for k in VARS:
        th = frames[k][list(TH)].to_numpy(float)
        for part, idx in PARTS.items():
            t = np.zeros_like(th)
            t[:, list(idx)] = th[:, list(idx)]
            Cp = a["c8"] * mult(a, t)
            lay = CL.layers(x, Cp, gap)
            rows.append((k, part + "|F_vs_W5G_pct", _dpct(T.ql(rv, lay["F"].to_numpy(float)), lw)))
            if part == "full":
                rows.append((k, "full|max_rel_F_vs_saved", float(np.max(np.abs(lay["F"].to_numpy(float) / frames[k]["F"].to_numpy(float) - 1.0)))))
    return rows


def level_stats(frames: dict, c8) -> list:
    rows = []
    for k in VARS:
        dl = np.log(frames[k]["C"].to_numpy(float) / c8)
        act = np.abs(dl) > 1e-12
        rows += [(k, "n_days_C_ne_cfg8", int(act.sum())), (k, "share_days_C_ne_cfg8", float(act.mean())),
                 (k, "mean_ln_C_cfg8", float(dl[act].mean()) if act.any() else float("nan")), (k, "sd_ln_C_cfg8", float(dl[act].std(ddof=1)) if act.sum() > 1 else float("nan"))]
    return rows


def month_stats(frames: dict) -> list:
    rows = []
    for k in VARS:
        f = frames[k]
        m = f[f["active"]].drop_duplicates("refit_R")
        rows += [(k, "n_active_months", int(len(m))), (k, "n_months_theta0", int((m["lam"] == 0.0).sum()))] + [(k, "n_months_lam_%g" % lam, int((m["lam"] == lam).sum())) for lam in (0.1, 0.01, 0.001)]
    return rows


def frozen(frames: dict, lw, rv) -> list:
    rows = []
    for k in VARS:
        f, ff = _dpct(T.ql(rv, frames[k]["F"].to_numpy(float)), lw), _dpct(T.ql(rv, frames[k]["F_fr"].to_numpy(float)), lw)
        rows += [(k, "F_vs_W5G_pct", f), (k, "Ffr_vs_W5G_pct", ff), (k, "reaction_pct", f - ff)]
    return rows


def mech(frames: dict, x, gap, w5f) -> pd.DataFrame:
    rv = x["rv"].to_numpy(float)
    lw = T.ql(rv, w5f)
    rows = level_stats(frames, x["C"].to_numpy(float)) + month_stats(frames) + frozen(frames, lw, rv) + fixed_theta(x, gap, frames, lw)
    return pd.DataFrame(rows, columns=["series", "item", "value"]).assign(section=lambda d: d["item"].map(lambda i: "fixed_theta" if "|" in i else "level_months_frozen"))[["section", "series", "item", "value"]]


def _v(m: pd.DataFrame, series: str, item: str) -> float:
    return float(m[(m["series"] == series) & (m["item"] == item)]["value"].iloc[0])


def share_level(m: pd.DataFrame, k: str) -> float:
    full, lev = _v(m, k, "full|F_vs_W5G_pct"), _v(m, k, "level_only|F_vs_W5G_pct")
    return lev / full if full < 0 else float("nan")


def predictions(m: pd.DataFrame) -> pd.DataFrame:
    pm, pf = _v(m, "S-RCKm", "mean_ln_C_cfg8"), _v(m, "S-RCKf", "mean_ln_C_cfg8")
    s_f, s_rck = share_level(m, "S-RCKf"), share_level(m, "S-RCK")
    rows = [("P-m", "среднее δ у S-RCKm < %.4f (у S-RCK по спецификации)" % T.THR_PM, pm, T.THR_PM, bool(pm < T.THR_PM)),
            ("P-f", "модуль среднего δ у S-RCKf < %.4f (половина S-RCK)" % T.THR_PF, abs(pf), T.THR_PF, bool(abs(pf) < T.THR_PF)),
            ("P-f2", "доля «только уровень» в Δ(F к W5-G) при фиксированных θ у S-RCKf < доли S-RCK (спецификация: %.2f)" % T.SHARE_SPEC, s_f, s_rck, bool(s_f < s_rck))]
    return pd.DataFrame([dict(id=i, statement=s, value=v, threshold=t, fulfilled="выполнено" if ok else "нет") for i, s, v, t, ok in rows])
