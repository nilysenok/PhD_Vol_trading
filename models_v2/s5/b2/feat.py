"""B2: признаки F1–F6 (GATES_B2 §1). Строка d — только бары <= 18:40 дня d и раньше; нормировка как в Б-2 (минус ln S, S = RV_m)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s2 import features as ft
from models_v2.s4 import mkt
from models_v2.s5.b2 import cusum, feat_day
from models_v2.s5.b2.util import lag, ln_floor, roll, roll_corr

GROUPS = {
    "F1": ("f1_lag1", "f1_lag2", "f1_m10", "f1_m63", "f1_d_over_w", "f1_chg", "f1_acc", "f1_std22", "f1_dev63"),
    "F2": ("f2_jshare", "f2_jshare_m", "f2_jfreq22", "f2_jfreq5", "f2_jint_chg", "f2_since", "f2_last_size", "f2_last_share"),
    "F3": ("f3_rs_ratio", "f3_rs_ratio5", "f3_ret1", "f3_neg1", "f3_cumneg5", "f3_cumneg22", "f3_cum5", "f3_lev63"),
    "F4": ("f4_rq", "f4_rq_rel", "f4_medrv", "f4_park", "f4_range", "f4_vov", "f4_hhi", "f4_hhi_chg"),
    "F5": ("f5_corr_d", "f5_corr22", "f5_beta22", "f5_mret", "f5_mneg5", "f5_dvol", "f5_rel22", "f5_vol_dev"),
    "F6": ("f6_dev252", "f6_pct", "f6_acc", "f6_m_over_long", "f6_mkt_chg", "f6_har_bias22", "f6_har_abs63", "f6_cusum", "f6_since",
           "f6_dcorr"),
}
ALL = tuple(c for g in GROUPS.values() for c in g)
H_MAIN = 5.0


def _ln_mean(x, seg, w, mp=None):
    return np.log(roll(x, seg, w, "mean", mp))


def f1(F, a) -> dict:
    s, lr, lS = F.seg, F.lnrv, F.X[:, 2]
    chg = lr - lag(lr, s)
    return {"f1_lag1": lag(lr, s) - lS, "f1_lag2": lag(lr, s, 2) - lS, "f1_m10": _ln_mean(F.rv, s, 10) - lS,
            "f1_m63": _ln_mean(F.rv, s, 63) - lS, "f1_d_over_w": F.X[:, 0] - F.X[:, 1], "f1_chg": chg,
            "f1_acc": chg - lag(chg, s), "f1_std22": roll(lr, s, 22, "std"),
            "f1_dev63": lr - lag(_ln_mean(F.rv, s, 63), s)}


def f2(F, a) -> dict:
    s, lS = F.seg, F.X[:, 2]
    J = np.expm1(F.X[:, 3])
    share = J / F.rv
    flag = mkt.paper(a["bars"])["day"]["jump"].reindex(pd.DatetimeIndex(F.dates)).to_numpy(float)
    f22, f5 = roll(flag, s, 22, "mean", 11), roll(flag, s, 5, "mean", 3)
    since, size, lsh = np.full(len(J), 63.0), np.full(len(J), np.nan), np.full(len(J), np.nan)
    last = None
    for i in range(len(J)):
        if i > 0 and s[i] != s[i - 1]:
            last = None
        if flag[i] == 1.0:
            last = i
        if last is not None:
            since[i], size[i], lsh[i] = min(63, i - last), np.log1p(J[last] / np.exp(lS[i])), share[last]
    return {"f2_jshare": share, "f2_jshare_m": roll(share, s, 22, "mean"), "f2_jfreq22": f22, "f2_jfreq5": f5,
            "f2_jint_chg": f5 - f22, "f2_since": since, "f2_last_size": size, "f2_last_share": lsh}


def f3(F, a) -> dict:
    s, X = F.seg, F.X
    ret = 100.0 * np.log(a["day"]["close"].to_numpy() / lag(a["day"]["close"].to_numpy(), s))
    neg = np.minimum(ret, 0.0)
    lev = roll_corr(ret, F.lnrv - lag(F.lnrv, s), s, 63, 40)
    return {"f3_rs_ratio": ln_floor(X[:, 8]) - ln_floor(X[:, 7]),
            "f3_rs_ratio5": ln_floor(roll(X[:, 8], s, 5, "sum")) - ln_floor(roll(X[:, 7], s, 5, "sum")), "f3_ret1": ret,
            "f3_neg1": neg, "f3_cumneg5": roll(neg, s, 5, "sum"), "f3_cumneg22": roll(neg, s, 22, "sum"),
            "f3_cum5": roll(ret, s, 5, "sum"), "f3_lev63": lev}


def f4(F, a) -> dict:
    s, lS, D = F.seg, F.X[:, 2], a["day"]
    lrq = np.log(D["rq"].to_numpy())
    hhi = D["hhi"].to_numpy()
    chg = F.lnrv - lag(F.lnrv, s)
    return {"f4_rq": lrq - 2 * lS, "f4_rq_rel": lrq - 2 * F.lnrv, "f4_medrv": np.log(D["medrv"].to_numpy()) - lS,
            "f4_park": np.log(D["park"].to_numpy()) - lS, "f4_range": np.log(D["range2"].to_numpy()) - lS,
            "f4_vov": roll(chg, s, 22, "std"), "f4_hhi": hhi, "f4_hhi_chg": hhi - lag(roll(hhi, s, 22, "mean"), s)}


def f5(F, a) -> dict:
    s, C, D = F.seg, a["cross"], a["day"]
    corr = (C["sxy"] / np.sqrt(C["sss"] * C["smm"])).to_numpy(float)
    ret = 100.0 * np.log(D["close"].to_numpy() / lag(D["close"].to_numpy(), s))
    mret = C["mret"].to_numpy(float)
    lv = np.log(D["volume"].to_numpy())
    return {"f5_corr_d": corr, "f5_corr22": roll(corr, s, 22, "mean"),
            "f5_beta22": roll(C["sxy"].to_numpy(float), s, 22, "sum") / roll(C["smm"].to_numpy(float), s, 22, "sum"),
            "f5_mret": mret, "f5_mneg5": roll(np.minimum(mret, 0.0), s, 5, "sum"),
            "f5_dvol": np.log(C["i_rv"].to_numpy(float)) - lag(np.log(C["i_rv"].to_numpy(float)), s),
            "f5_rel22": roll(ret - mret, s, 22, "sum"), "f5_vol_dev": lv - lag(roll(lv, s, 22, "mean"), s)}


def _pct(x, seg, w=252, mp=126) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for g in np.unique(seg):
        m = np.flatnonzero(seg == g)
        out[m] = pd.Series(x[m]).rolling(w, min_periods=mp).apply(lambda v: float((v <= v[-1]).mean()), raw=True).to_numpy()
    return out


def f6(F, a) -> dict:
    s, C = F.seg, a["cross"]
    lw, lm = F.X[:, 1], F.X[:, 2]
    l252 = _ln_mean(F.rv, s, 252, 126)
    e = cusum.har_arrivals(F)
    sp, sm, al = cusum.run(e, H_MAIN)
    corr22 = f5(F, a)["f5_corr22"]
    return {"f6_dev252": lw - l252, "f6_pct": _pct(np.exp(lw), s), "f6_acc": (lw - lm) - lag(lw - lm, s, 5), "f6_m_over_long": lm - l252,
            "f6_mkt_chg": np.log(C["i_rv_w"].to_numpy(float)) - np.log(C["i_rv_m"].to_numpy(float)),
            "f6_har_bias22": roll(e, s, 22, "mean"), "f6_har_abs63": roll(np.abs(e), s, 63, "mean"),
            "f6_cusum": np.maximum(sp, sm), "f6_since": cusum.since(al, 252), "f6_dcorr": corr22 - lag(corr22, s, 5)}


FUN = {"F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6}


def build(F: ft.Fp, bars: pd.DataFrame, imoex: pd.DataFrame) -> pd.DataFrame:
    """Таблица всех 51 столбцов F1–F6 по строкам F.dates (индекс — дата)."""
    a = {"bars": bars, "day": feat_day.day_stats(bars, F.dates), "cross": feat_day.cross_stats(bars, imoex, F.dates)}
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for g, fn in FUN.items():
            r = fn(F, a)
            assert tuple(r) == GROUPS[g], g
            out.update(r)
    df = pd.DataFrame(out, index=pd.DatetimeIndex(F.dates, name="d"))
    return df.replace([np.inf, -np.inf], np.nan)
