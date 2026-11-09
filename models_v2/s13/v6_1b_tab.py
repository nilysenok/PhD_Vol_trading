"""V6-1b: таблицы листа 1b — правило «новая информация» (A_INFO, план п. 2.2), сравнение с V5 SP1–SP4 (A_VS_SP, G20), коэффициенты LIN и важность GBM, чувствительность, уровни 1–5 (A_LEVELS, G22).
Все функции чистые (от прогнозов и контролей на общих датах); ничего не пишут."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_load as LD, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV, v6_1_wf as WF

SP = ["SP1", "SP2", "SP3", "SP4"]
INFO_WIN = ("2017_2021", "2022_2025", "2026", "all")


def sp_series(index) -> pd.DataFrame:
    """Сохранённые прогнозы V5 SP1–SP4 по SBER на общих датах (не переобучаются; чтение как в листе 0)."""
    out = {}
    for c in SP:
        f = pd.read_parquet(LD.V5 / (c + ".parquet"), columns=["ticker", "date", "t_plus", "F"])
        f = f[f["ticker"] == "SBER"]
        out[c] = pd.Series(f["F"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(f["date"]))).reindex(index).to_numpy()
    return pd.DataFrame(out, index=index)


def info_table(fs: dict, pairs: list, w: pd.DataFrame, cp0: dict) -> pd.DataFrame:
    """Строка на (пара X против Y, окно): Δ QLIKE, Δ в % уровня K1 с 90 % интервалом, DM p (односторонний «X лучше»), лет с Δ < 0; для 2017–2021 — условия (i)–(iii) п. 2.2 и вывод; p Холма (N, N_wide) внутри окна по парам."""
    rv, tp, rows = w["rv"].to_numpy(), w["t_plus"].to_numpy(), []
    for x, y in pairs:
        for win in INFO_WIN:
            m = Q.wmask(tp, win)
            qx, qy = Q.ql(rv[m], fs[x][m]), Q.ql(rv[m], fs[y][m])
            ax, ay = Q.annual(qx, tp[m]), Q.annual(qy, tp[m])
            r = dict(pair="%s vs %s" % (x, y), x=x, y=y, window=win, n=int(m.sum()), delta_qlike=float(qx.mean() - qy.mean()), years_neg=int(sum(ax[k] - ay[k] < 0 for k in ax if k in ay)), years_n=len(ax))
            r.update(EV.delta_ci(qx, qy, float(Q.ql(rv[m], w["K1"].to_numpy()[m]).mean())))
            r.update({"dm_" + k: v for k, v in EV.dm_one(rv[m], fs[y][m], fs[x][m]).items() if k in ("stat", "p_one")})
            ni = EV.new_information(rv, fs[x], fs[y], tp) if win == "2017_2021" else {}
            r.update(cond_i=ni.get("cond_i"), cond_ii=ni.get("cond_ii"), cond_iii=ni.get("cond_iii"), new_information=ni.get("new_information"))
            rows.append(r)
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for win in INFO_WIN:
        s = (t["window"] == win) & t["dm_p_one"].notna()
        t.loc[s, "p_holm_N"], t.loc[s, "p_holm_Nwide"] = EV.holm_both(t.loc[s, "dm_p_one"].to_numpy(), cp0)
    return t


def vs_sp_table(fs: dict, cids: list, sp: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """A_k против SP1–SP4 SBER: Δ в % уровня K1 с интервалом, DM (p «A лучше»), охват: вес w в SP^(1−w)·A^w с 90 % интервалом. Окна: вся история, 2017–2021, 2022–2025."""
    rv, tp, rows = w["rv"].to_numpy(), w["t_plus"].to_numpy(), []
    for c in cids:
        for s in SP:
            for win in ("all", "2017_2021", "2022_2025"):
                m = Q.wmask(tp, win)
                qa, qs = Q.ql(rv[m], fs[c][m]), Q.ql(rv[m], sp[s].to_numpy()[m])
                dm = EV.dm_one(rv[m], sp[s].to_numpy()[m], fs[c][m])
                r = dict(config=c, sp=s, window=win, n=int(m.sum()), qlike_config=float(qa.mean()), qlike_sp=float(qs.mean()), **EV.delta_ci(qa, qs, float(Q.ql(rv[m], w["K1"].to_numpy()[m]).mean())), dm_stat=dm["stat"], dm_p_one=dm["p_one"])
                r.update({"cover_" + k: v for k, v in CV.coverage(rv[m], sp[s].to_numpy()[m], fs[c][m]).items()})
                rows.append(r)
    return pd.DataFrame(rows)


def coef_table(run: dict, names: list) -> pd.DataFrame:
    """θ LIN по датам переоценки (в единицах z: нормировка по закрытым парам переоценки): d0, feature, theta, lam, n_train."""
    rows = [dict(d0=str(f["d0"].date()), feature=n, theta=float(t), lam=float(f["model"].lam), n_train=int(f["n_train"]))
            for f in run["fits"] for n, t in zip(["const"] + list(names), f["model"].theta)]
    return pd.DataFrame(rows)


def coef_stability(coef: pd.DataFrame, feats: list) -> dict:
    """По признакам: среднее θ, СКО, доля переоценок со знаком среднего. «Устойчивый вес» — доля ≥ 0,9 и |среднее| > СКО."""
    out = {}
    for f in feats:
        t = coef.loc[coef["feature"] == f, "theta"].to_numpy(float)
        sh = float((np.sign(t) == np.sign(t.mean())).mean())
        out[f] = dict(mean=float(t.mean()), sd=float(t.std()), share_same_sign=sh, stable=bool(sh >= 0.9 and abs(t.mean()) > t.std()))
    return out


def imp_table(run: dict, names: list) -> pd.DataFrame:
    """Важность признаков GBM: доля прироста (gain) и число разбиений — среднее по переоценкам и на последней."""
    g = np.array([f["model"].booster.feature_importance("gain") for f in run["fits"]], float)
    sp = np.array([f["model"].booster.feature_importance("split") for f in run["fits"]], float)
    sh = g / np.where(g.sum(1, keepdims=True) > 0, g.sum(1, keepdims=True), 1.0)
    return pd.DataFrame(dict(feature=list(names), gain_share_mean=sh.mean(0), gain_share_last=sh[-1], splits_mean=sp.mean(0), splits_last=sp[-1], n_refits=len(g)))


def sens_rows(tab, X, w, variants: dict, f_main, f_a0) -> pd.DataFrame:
    """Соседние точки (не конфигурации): variants — имя -> (run_fn(tab, X) -> dict с g, fits). Δ к K1 и к A0 в % уровня K1."""
    rv, tp, rows = w["rv"].to_numpy(), w["t_plus"].to_numpy(), []
    for name, fn in variants.items():
        r = fn(tab, X)
        f = WF.forecasts(tab, r["g"]).reindex(w.index).to_numpy()
        row = dict(variant=name, params=json.dumps(_params(r), ensure_ascii=False))
        for win in ("all", "2017_2021"):
            m = Q.wmask(tp, win)
            q, q1, q0 = Q.ql(rv[m], f[m]), Q.ql(rv[m], w["K1"].to_numpy()[m]), Q.ql(rv[m], f_a0[m])
            row.update({"qlike_" + win: float(q.mean()), "d_pct_K1_" + win: 100 * float((q - q1).mean()) / float(q1.mean()), "d_pct_A0_" + win: 100 * float((q - q0).mean()) / float(q1.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def _params(r: dict) -> dict:
    mod = r["fits"][0]["model"]
    key, vals = ("lam", [x["model"].lam for x in r["fits"]]) if hasattr(mod, "lam") else ("n_trees", [x["model"].n_trees for x in r["fits"]])
    return {key + "_counts": {str(k): int(v) for k, v in pd.Series(vals).value_counts().sort_index().items()}}
