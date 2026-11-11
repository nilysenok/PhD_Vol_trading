"""S-R-EVAL (Ж-254; `S_R_SPEC.md`, разд. 4.6, 7): параметры, доля месяцев θ = 0, выбор по месяцам, разложение «заморожен π / реакция слоя», общее активное окно, диагностика в выборке. Только чтение рядов `R/*.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Публичное API:
- `KEY` — вариант («S-RA2»…) → ключ r_walk (`ra2`…); `SETS` — наборов подгонок на активный месяц (R, Rb: 3 λ × 3 блока + 1 = 10; Rc: 3 + 1 = 4); `N_SEEDS` = 5 (A2 — seeds, CK — перезапуски).
- `params(frames, cost)` → long: variant, kind, item, key, value: месяцы (активных, θ = 0, доля θ = 0, по λ*), число подгонок, CPU-секунды (из `R/cost.json`), c, m, κ по месяцам, квантили θ (месяцы с λ* > 0), квантили ln(C_R/cfg8) на активных датах.
- `sel(frames)` → по месяцам: variant, refit_R, n_train, c, m, kappa, lam, w_*, G_* (блоки b1–b3).
- `decomp(ls, d)` → разложение Δ(F_R против W5-G) на «замороженный π» (F_fr) и «реакцию слоя» (F_R − F_fr); Δ на общем активном окне: d ≥ 2020-08-03 (обе архитектуры, R против прежнего и против W5-G), d ≥ 2019-08-01 (A).
- `insample(frames, ctxs)` → в выборке по месяцам с θ ≠ 0 (S-C-WHY разд. 5.4): доля выигрыша обычной QLIKE от 1 % лучших пар пула и знак выигрыша на остальных 99 %."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W

KEY = {"S-RA2": "ra2", "S-RCK": "rck", "S-RA2b": "ra2b", "S-RCKb": "rckb", "S-RA2c": "ra2c", "S-RCKc": "rckc"}
SETS, N_SEEDS, LN_EPS = {"blocks": 10, "single": 4}, 5, 1e-3
QS = (0.01, 0.05, 0.5, 0.95, 0.99)
COST = W.R_DIR / "cost.json"


def _months(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["active"]].groupby("refit_R", sort=True).first().reset_index()


def _row(v, kind, item, key, value) -> dict:
    return dict(variant=v, kind=kind, item=item, key=key, value=value)


def params(frames: dict, cost: dict) -> pd.DataFrame:
    out = []
    for v, df in frames.items():
        arch, mode, rule = W.VARS[KEY[v]]
        m, act = _months(df), df[df["active"]]
        z = int((m["lam"] == 0.0).sum())
        out += [_row(v, "months", "active", "", len(m)), _row(v, "months", "theta_zero", "", z), _row(v, "months", "share_theta_zero", "", z / len(m))]
        out += [_row(v, "lam", "%g" % lam, "", int((m["lam"] == lam).sum())) for lam in W.LAMS]
        out += [_row(v, "fits", "n_fits", "", len(m) * SETS[rule] * N_SEEDS), _row(v, "cost", "cpu_seconds", "", float(cost.get(KEY[v], float("nan"))))]
        if mode == "R":
            kap = m["c"] / m["m"]
            out += [_row(v, "clip", n, "", float(f(x))) for n, x, f in (("c_median", m["c"], np.median), ("c_min", m["c"], np.min), ("c_max", m["c"], np.max), ("m_median", m["m"], np.median), ("kappa_median", kap, np.median))]
        nz = m[m["lam"] > 0]
        for p in W.names(arch):
            if len(nz):
                out += [_row(v, "theta_q", p, "q%02d" % round(100 * q), float(np.quantile(nz[p], q))) for q in QS]
        ln = np.log(act["C"].to_numpy() / act["cfg8"].to_numpy())
        out += [_row(v, "lnratio", "q", "q%02d" % round(100 * q), float(np.quantile(ln, q))) for q in QS] + [_row(v, "lnratio", "share_abs_gt_1e-3", "", float((np.abs(ln) > LN_EPS).mean()))]
    return pd.DataFrame(out)


def sel(frames: dict) -> pd.DataFrame:
    cols = ["refit_R", "n_train", "c", "m"] + ["w_" + k for k in W.LK] + ["G_%s_b%d" % (k, b) for k in W.LK for b in (1, 2, 3)]
    out = []
    for v, df in frames.items():
        m = _months(df)
        t = m[["lam"] + cols].copy()
        t.insert(0, "variant", v)
        t["kappa"] = t["c"] / t["m"]
        out.append(t[["variant", "refit_R", "n_train", "c", "m", "kappa", "lam"] + cols[4:]])
    return pd.concat(out, ignore_index=True)


def _dl(ls, h, b, m) -> float:
    return 100.0 * (float(ls[h][m].mean()) - float(ls[b][m].mean())) / float(ls[b][m].mean())


def decomp(ls: dict, d) -> pd.DataFrame:
    d, rows, allm = pd.DatetimeIndex(d), [], np.ones(len(d), bool)
    for c, o in (("S-RA2", "S-A2"), ("S-RCK", "S-CK")):
        tot, fr = _dl(ls, c, "W5-G", allm), _dl(ls, "Ffr(%s)" % c, "W5-G", allm)
        rows += [dict(candidate=c, kind="decomp", item=i, window="D_M", n=len(d), value=v) for i, v in (("total_F_vs_W5G", tot), ("frozen_pi", fr), ("layer_reaction", tot - fr))]
        for w, first in (("d>=2020-08-03", "2020-08-03"), ("d>=2019-08-01", "2019-08-01")):
            if w == "d>=2019-08-01" and c != "S-RA2":
                continue
            m = np.asarray(d >= pd.Timestamp(first))
            rows += [dict(candidate=c, kind="common_window", item=i, window=w, n=int(m.sum()), value=_dl(ls, h, b, m)) for i, h, b in (("R_vs_W5G", c, "W5-G"), ("old_vs_W5G", o, "W5-G"), ("R_vs_old", c, o))]
    return pd.DataFrame(rows)


def insample(frames: dict, ctxs: dict) -> pd.DataFrame:
    rows = []
    for v in ("S-RA2", "S-RCK"):
        arch = W.VARS[KEY[v]][0]
        ctx, df = ctxs[arch], frames[v]
        xd = pd.DatetimeIndex(ctx.x["d"])
        shares, rest_ok, n0 = [], [], 0
        for _, r in _months(df).iterrows():
            if r["lam"] == 0.0:
                n0 += 1
                continue
            a = int(np.flatnonzero(xd == pd.Timestamp(r["refit_R"]))[0])
            P = RL.pool_of(arch, ctx.D, ctx.x, a)
            V = ctx.view(P)
            lnf = RL.ln_f(RL.lnc_of(torch.as_tensor(r[W.names(arch)].to_numpy(float)), V, arch), V).numpy()
            gain = RL.ell(ctx.rv[P], ctx.lnf0[P]) - RL.ell(ctx.rv[P], lnf)
            k = int(math.ceil(0.01 * len(P)))
            top = float(np.sort(gain)[-k:].sum())
            shares.append(top / float(gain.sum()) if gain.sum() > 0 else float("nan"))
            rest_ok.append(float(gain.sum()) - top > 0)
        rows += [dict(variant=v, kind="insample", item=i, key="", value=x) for i, x in (("months_theta_nonzero", len(shares)), ("months_theta_zero", n0), ("share_top1pct_median", float(np.nanmedian(shares)) if shares else float("nan")),
                                                                                       ("frac_rest99_better", float(np.mean(rest_ok)) if rest_ok else float("nan")))]
    return pd.DataFrame(rows)


def read_cost() -> dict:
    return json.loads(Path(COST).read_text(encoding="utf-8")) if Path(COST).exists() else {}
