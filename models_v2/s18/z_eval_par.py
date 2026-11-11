"""S-Z-EVAL (Ж-258; `S_Z_SPEC.md`, разд. 5.2 (З-5), 7.2, 7.6): параметры, выбор по месяцам, механизм (доля контроля, разложение «замороженный π / реакция слоя», ln(C_Z/C_S-RCK), сравнение с нулевым распределением), диагностика z̃.
Читает СОХРАНЁННЫЕ `Z/*.parquet`, `R/rck.parquet` и `zmaps.npz`; ничего не обучает и не пишет. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Исследовательский результат на раскрытой истории, не OOS.
Публичное API:
- `params(frames) -> long(variant, kind, item, key, value)`: месяцы (активных, θ = 0, доля θ ≠ 0, по λ*), c, m, κ, θ_Z по месяцам (среднее и разброс по головам; θ разных лет несравнимы), `n_clip`, число дат с `z_ok` = 0 и `n_excl` > 0.
- `ztilde(frames, ctx, zm) -> long`: доля обрезанных z̃ и std z̃ на днях прогноза по годам (З-5); μ, σ — по закрытому пулу месяца, как при подгонке.
- `sel(frames)` — по месяцам: refit, n_train, c, m, κ, λ*, w_λ, G_λ,b (блоки b1–b3).
- `share(d_z, d_zs) -> float`: Δ(S-RCKZs, S-RCK)/Δ(S-RCKZ, S-RCK); NaN, если выигрыша S-RCKZ нет (Δ ≥ 0).
- `mech(inp, ls, null) -> DataFrame(section, item, key, value)`: H-Z2 и доля контроля (правило: доля ≥ 0,5 ⇒ «выигрыш в основном не связан с датой z»), разложение Δ на F^fr и реакцию слоя, ln(C_Z/C_S-RCK) и корреляция с ln(RV/C_S-RCK) (ex post), сравнение реальных Δ с нулевым распределением."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import r_eval_par as RP
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18 import z_model as ZM

QS = (0.01, 0.05, 0.5, 0.95, 0.99)
SHARE_HALF = 0.5
LN_EPS = 1e-3
sel, _row, _months = RP.sel, RP._row, RP._months


def params(frames: dict) -> pd.DataFrame:
    out = []
    for v in ("S-RCK",) + C.REAL_Z:
        df = frames[v]
        m = _months(df)
        z, nz = int((m["lam"] == 0.0).sum()), m[m["lam"] > 0]
        out += [_row(v, "months", "active", "", len(m)), _row(v, "months", "theta_zero", "", z), _row(v, "months", "share_theta_nonzero", "", 1.0 - z / len(m))]
        out += [_row(v, "lam", "%g" % lam, "", int((m["lam"] == lam).sum())) for lam in W.LAMS]
        kap = m["c"] / m["m"]
        out += [_row(v, "clip", n, "", float(f(x))) for n, x, f in (("c_median", m["c"], np.median), ("c_min", m["c"], np.min), ("c_max", m["c"], np.max), ("m_median", m["m"], np.median), ("kappa_median", kap, np.median))]
        if v != "S-RCK" and len(nz):
            for j in range(4):
                out += [_row(v, "thz", "mean_over_months", "z%d" % j, float(nz["thz_%d" % j].mean())), _row(v, "thz", "std_over_months", "z%d" % j, float(nz["thz_%d" % j].std(ddof=0))),
                        _row(v, "thz", "mean_thsd_heads", "z%d" % j, float(nz["thsd_z%d" % j].mean()))]
            out += [_row(v, "n_clip", "median", "", float(nz["n_clip"].median())), _row(v, "n_clip", "max", "", float(nz["n_clip"].max()))]
        if v != "S-RCK":
            out += [_row(v, "z", "dates_z_ok_zero", "", int((df["z_ok"] == 0).sum())), _row(v, "z", "dates_n_excl_positive", "", int((df["n_excl"] > 0).sum()))]
    return pd.DataFrame(out)


def ztilde(frames: dict, ctx, zm: dict) -> pd.DataFrame:
    df, xd, rows = frames["S-RCKZ"], pd.DatetimeIndex(ctx.x["d"]), {}
    for r, g in df[df["active"]].groupby("refit_R", sort=True):
        a, idx = int(np.flatnonzero(xd == pd.Timestamp(r))[0]), np.flatnonzero((df["refit_R"] == r).to_numpy())
        P, days, Y = RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[idx], pd.Timestamp(r).year
        for s in range(5):
            z = zm[Y][s]
            raw = ZM.ztil_raw(z[days], *ZM.std_stats(z, P))
            rows.setdefault(Y, []).append(raw)
    out = []
    for Y, lst in rows.items():
        raw = np.vstack(lst)
        out += [_row("S-RCKZ", "ztilde_forecast", "share_clipped", str(Y), float((np.abs(raw) > ZM.CLIP).mean()))] + [_row("S-RCKZ", "ztilde_forecast", "std_z%d" % j, str(Y), float(np.clip(raw[:, j], -ZM.CLIP, ZM.CLIP).std())) for j in range(4)]
    return pd.DataFrame(out)


def share(d_z: float, d_zs: float) -> float:
    return float(d_zs / d_z) if d_z < 0 else float("nan")


def _dp(lh, lb, m=None) -> float:
    m = np.ones(len(lh), bool) if m is None else m
    return 100.0 * float(lh[m].sum() - lb[m].sum()) / float(lb[m].sum())


def _null_rows(ls: dict, act: np.ndarray, null: tuple) -> list:
    s, m = null
    q = m.set_index(["item", "stat"])["value"]
    lo, hi = float(q[("qlike_rel_pct", "q05")]), float(q[("qlike_rel_pct", "q95")])
    out = [("noise", "q05", "", lo), ("noise", "q95", "", hi), ("noise", "mean", "", float(q[("qlike_rel_pct", "mean")])), ("noise", "std", "", float(q[("qlike_rel_pct", "std")])), ("noise", "share_nonzero_mean", "", float(q[("share_theta_nonzero", "mean")]))]
    for tag, h, b in (("Z_vs_RCK", "S-RCKZ", "S-RCK"), ("Z_vs_Zs", "S-RCKZ", "S-RCKZs")):
        for w, mk in (("D_M", None), ("active", act)):
            d = _dp(ls[h], ls[b], mk)
            out += [("noise_real", "%s|%s" % (tag, w), "", d), ("noise_rule", "%s|%s" % (tag, w), ZN.rule(d, lo, hi), d)]    # секции не «null»: pandas читает слово как NaN
    return out


def mech(inp: dict, ls: dict, null: tuple) -> pd.DataFrame:
    fr, F, rv = inp["frames"], inp["F"], inp["rv"]
    z, rk, zs = fr["S-RCKZ"], fr["S-RCK"], fr["S-RCKZs"]
    act = z["active"].to_numpy(bool)
    d_z, d_zs, d_zzs = (_dp(ls["S-RCKZ"], ls["S-RCK"]), _dp(ls["S-RCKZs"], ls["S-RCK"]), _dp(ls["S-RCKZ"], ls["S-RCKZs"]))
    sh = share(d_z, d_zs)
    txt = "доли нет: выигрыша S-RCKZ над S-RCK нет" if np.isnan(sh) else "выигрыш в основном не связан с датой z" if sh >= SHARE_HALF else "доля контроля < 0,5"
    rows = [("hz2", "delta_Z_vs_RCK_pct", "", d_z), ("hz2", "delta_Zs_vs_RCK_pct", "", d_zs), ("hz2", "delta_Z_vs_Zs_pct", "", d_zzs), ("hz2", "share_control", txt, sh)]
    fr_z, fr_k = z["F_fr"].to_numpy(float), rk["F_fr"].to_numpy(float)
    frozen = 100.0 * float(S.Q.ql(rv, fr_z).sum() - S.Q.ql(rv, fr_k).sum()) / float(ls["S-RCK"].sum())
    rows += [("decomp", "total_vs_RCK", "", d_z), ("decomp", "frozen_pi", "", frozen), ("decomp", "layer_reaction", "", d_z - frozen)]
    lnr = np.log(z["C"].to_numpy(float) / rk["C"].to_numpy(float))[act]
    tgt = np.log(rv / rk["C"].to_numpy(float))[act]
    nzm = np.abs(lnr) > LN_EPS
    rows += [("lnratio", "q", "q%02d" % round(100 * q), float(np.quantile(lnr, q))) for q in QS] + [("lnratio", "share_abs_gt_1e-3", "", float(nzm.mean())), ("lnratio", "n_active_dates", "", int(act.sum())),
             ("corr_ex_post", "corr_all_active", "ex post", float(np.corrcoef(lnr, tgt)[0, 1]) if lnr.std() > 0 else float("nan")),
             ("corr_ex_post", "corr_changed_dates", "ex post", float(np.corrcoef(lnr[nzm], tgt[nzm])[0, 1]) if nzm.sum() > 2 else float("nan"))]
    return pd.DataFrame(rows + _null_rows(ls, act, null), columns=["section", "item", "key", "value"])
