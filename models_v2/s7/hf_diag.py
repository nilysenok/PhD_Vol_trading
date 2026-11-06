"""Этап I (п. 4 блока): диагностика ошибки HAR-J без обучения. Ошибка e = ln RV_{d+1} − ln σ̂²(d); потери QLIKE; признаки на d (PIT).
Выходы: HAR_RESIDUAL_DIAGNOSTICS.csv (по дням), HAR_ERROR_STRUCTURE.csv (8 пунктов), YEARLY_STRUCTURE.csv (почему 2017–18), HAR_RIDGE_BY_YEAR.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s6.e import comp, ser, st_ex1
from models_v2.s7 import hf_data as D

TOP = 5


def daily(w: dict) -> pd.DataFrame:
    df, F, g, fg = w["df"], w["F"], w["g"], w["fg"]
    o = df[["d", "target_date", "rv", "HAR", "XJ", "LF", "st_g4", "st_rv"]].copy()
    for m in ("HAR", "XJ", "LF"):
        o["e_" + m] = np.log(o["rv"] / o[m])
        o["q_" + m] = D.qs(df, o[m])
    o["dq_HAR_XJ"], o["dq_HAR_LF"] = o["q_HAR"] - o["q_XJ"], o["q_HAR"] - o["q_LF"]
    X = F.X[g]
    o["rv_d"], o["rv_w"], o["rv_m"] = F.rv[g], np.exp(X[:, 1]), np.exp(X[:, 2])
    o["ln_rvw_over_rvm"] = X[:, 1] - X[:, 2]
    o["jump_share_d"] = fg["f2_jshare"].to_numpy()
    o["jump_freq22"], o["jump_freq5"], o["since_jump"] = fg["f2_jfreq22"].to_numpy(), fg["f2_jfreq5"].to_numpy(), fg["f2_since"].to_numpy()
    o["rs_plus"], o["rs_minus"] = X[:, 7], X[:, 8]
    o["ret1_pct"], o["lev63"], o["rq_rel"], o["vov"] = (fg[c].to_numpy() for c in ("f3_ret1", "f3_lev63", "f4_rq_rel", "f4_vov"))
    for c in ("f5_corr22", "f5_mret", "f5_dvol", "f6_mkt_chg", "f6_cusum", "f6_since", "f6_har_bias22", "f6_pct"):
        o[c] = fg[c].to_numpy()
    o["ter_rv_w"] = D.ter_w(F, g)
    return o


def _row(o, name: str, m: np.ndarray, grp: str) -> dict:
    n = int(m.sum())
    if n < 20:
        return {"point": grp, "condition": name, "n": n}
    e = o["e_HAR"].to_numpy()[m]
    t = e.mean() / np.sqrt(dmod.nw_variance(e)) if e.std() > 0 else np.nan
    return {"point": grp, "condition": name, "n": n, "share": n / len(o), "mean_e_HAR": e.mean(), "t_nw_e_HAR": t, "mean_e_XJ": o["e_XJ"][m].mean(),
            "mean_e_LF": o["e_LF"][m].mean(), "qlike_HAR": o["q_HAR"][m].mean(), "qlike_XJ": o["q_XJ"][m].mean(), "qlike_LF": o["q_LF"][m].mean(),
            "mean_dq_HAR_minus_XJ": o["dq_HAR_XJ"][m].mean()}


def structure(o: pd.DataFrame) -> pd.DataFrame:
    ex = np.log(o["rv"] / o["rv_w"]).to_numpy()          # реализованный скачок цели против RV_w (ex post, диагностика)
    ter, ret, yr = o["ter_rv_w"].to_numpy(), o["ret1_pct"].to_numpy(), D.year(o)
    prev_hi = pd.Series(ter).rolling(5).max().shift(1).to_numpy() == 2
    falling = (o["rv_w"] < o["rv_w"].shift(5) * 0.8).to_numpy()
    conds = [("1 резкий рост RV: ln(RV_{d+1}/RV_w) > 0,7 (ex post)", ex > 0.7, "1"), ("1 умеренный/без роста (|.|<0,3)", np.abs(ex) < 0.3, "1"),
             ("1 резкое падение: < −0,7 (ex post)", ex < -0.7, "1"),
             ("2 после стресса: терциль 2 за последние 5 дн. и RV_w упала >20 % за 5 дн.", prev_hi & falling, "2"),
             ("3 структурное: CUSUM-сигнал давно (f6_since<=22)", o["f6_since"].to_numpy() <= 22, "3"), ("3 без недавнего сигнала", o["f6_since"].to_numpy() > 22, "3"),
             ("4 после отрицательной доходности d (<0)", ret < 0, "4"), ("4 после доходности <= −2 %", ret <= -2, "4"), ("4 после неотрицательной", ret >= 0, "4"),
             ("5 скачок в последние 5 дн. (jfreq5>0)", o["jump_freq5"].to_numpy() > 0, "5"), ("5 без скачков за 5 дн.", o["jump_freq5"].to_numpy() == 0, "5"),
             ("5 доля скачка в RV дня d > 0,3", o["jump_share_d"].to_numpy() > 0.3, "5")]
    for k, nm in enumerate(("6 ln(RV_w/RV_m) низ", "6 середина", "6 верх")):
        q = o["ln_rvw_over_rvm"].quantile([1 / 3, 2 / 3]).to_numpy()   # пороги по всей выборке — диагностика, не прогноз
        conds.append((nm + " (терцили по выборке)", [o["ln_rvw_over_rvm"] <= q[0], (o["ln_rvw_over_rvm"] > q[0]) & (o["ln_rvw_over_rvm"] <= q[1]), o["ln_rvw_over_rvm"] > q[1]][k].to_numpy(), "6"))
    for k, nm in enumerate(("7 RV_w SBER низ (PIT-терциль)", "7 середина", "7 верх")):
        conds.append((nm, ter == k, "7"))
    mk = o["f6_mkt_chg"]
    qm = mk.quantile([1 / 3, 2 / 3]).to_numpy()
    conds += [("7 рынок IMOEX ln(RV_w/RV_m) верх (терциль выборки)", (mk > qm[1]).to_numpy(), "7"), ("7 рынок: низ", (mk <= qm[0]).to_numpy(), "7")]
    rows = [_row(o, n, np.asarray(m, bool), g) for n, m, g in conds]
    for y in sorted(set(yr)):
        rows.append(_row(o, "8 год %d" % y, yr == y, "8"))
    e = o["e_HAR"].to_numpy()
    for lag in (1, 5, 22):
        rows.append({"point": "8", "condition": "автокорреляция e_HAR лаг %d" % lag, "n": len(e) - lag, "mean_e_HAR": float(np.corrcoef(e[lag:], e[:-lag])[0, 1])})
    return pd.DataFrame(rows)


def _opt_w(h, x, rv) -> float:
    ws = np.linspace(0, 1, 21)
    from models_v2 import metrics
    return float(ws[int(np.argmin([metrics.qlike(rv, w * h + (1 - w) * x).mean() for w in ws]))])


def yearly(o: pd.DataFrame, ser_df: pd.DataFrame) -> pd.DataFrame:
    from models_v2 import metrics
    yr, rows = D.year(o), []
    for y in sorted(set(yr)):
        m = yr == y
        rv, h, x, l = (o[c].to_numpy()[m] for c in ("rv", "HAR", "XJ", "LF"))
        dq = o["dq_HAR_XJ"].to_numpy()[m]
        top = np.sort(np.abs(dq))[::-1][:TOP]
        eh, ex_ = o["e_HAR"].to_numpy()[m], o["e_XJ"].to_numpy()[m]
        rows.append({"year": y, "n": int(m.sum()), "qlike_HAR": o["q_HAR"][m].mean(), "qlike_XJ": o["q_XJ"][m].mean(), "qlike_LF": o["q_LF"][m].mean(),
                     "ratio_HAR_over_XJ": o["q_HAR"][m].mean() / o["q_XJ"][m].mean(), "corr_e_HAR_XJ": np.corrcoef(eh, ex_)[0, 1], "corr_e_XJ_LF": np.corrcoef(ex_, o["e_LF"][m])[0, 1],
                     "sd_e_HAR": eh.std(), "sd_e_XJ": ex_.std(), "mean_e_HAR": eh.mean(), "mean_e_XJ": ex_.mean(), "mean_dq_HAR_minus_XJ": dq.mean(),
                     "share_top%d_abs_dq" % TOP: top.sum() / np.abs(dq).sum(), "qlike_mean_HAR_XJ": metrics.qlike(rv, 0.5 * h + 0.5 * x).mean(),
                     "gain_mean_vs_best_comp": min(o["q_HAR"][m].mean(), o["q_XJ"][m].mean()) - metrics.qlike(rv, 0.5 * h + 0.5 * x).mean(),
                     "w_HAR_expost_diag": _opt_w(h, x, rv), "qlike_A1_roll_126": float(ser_df["A1_roll_126"].pipe(lambda s: D.qs(o, s))[m].mean()) if "A1_roll_126" in ser_df else np.nan,
                     "qlike_A1_minus_XJ": float((D.qs(o, ser_df["A1_roll_126"]) - o["q_XJ"].to_numpy())[m].mean()) if "A1_roll_126" in ser_df else np.nan})
    return pd.DataFrame(rows)


def stage(_a=None) -> int:
    w = D.world()
    o = daily(w)
    D.write(o, "HAR_RESIDUAL_DIAGNOSTICS.csv", "I: ежедневная таблица; e=ln RV_{d+1}−ln σ̂²(d); признаки на d; компоненты замороженные Ж-206/210")
    D.write(structure(o), "HAR_ERROR_STRUCTURE.csv", "I: восемь пунктов 'особого внимания' (п. 4 блока); пункты 1,6(выборочные терцили),7(рынок) — ex post/диагностика, не правило прогноза")
    sdf = ser.load(("a",))
    y = yearly(o, sdf)
    D.write(y, "YEARLY_STRUCTURE.csv", "I: структура ошибок по годам; w_HAR_expost_diag — оптимальный вес по году ПО ФАКТУ (только диагностика)")
    G = {"const": None, **{k: D.group_matrix(w["fg"], v) for k, v in D.feature_groups().items()}}
    G["F1-F6"] = np.hstack([v for v in G.values() if v is not None])
    r = st_ex1.ridge_rows(w["df"], G, None)
    D.write(r, "HAR_RIDGE_BY_YEAR.csv", "I: ridge годовой LOYO (alpha=n_train) остатка HAR-J на группы F1–F6; dq_exp_g_minus_har<0 — поправка улучшает QLIKE")
    a = r[r.year == "all"].set_index("model")["dq_exp_g_minus_har"].round(5).to_dict()
    print("G3 diag: n=%d; ridge ΔQLIKE (все годы): %s" % (len(o), a), flush=True)
    print(y[["year", "qlike_HAR", "qlike_XJ", "ratio_HAR_over_XJ", "corr_e_HAR_XJ", "w_HAR_expost_diag", "qlike_A1_minus_XJ"]].round(4).to_string(index=False), flush=True)
    return 0
