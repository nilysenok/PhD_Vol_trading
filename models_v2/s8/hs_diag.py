"""Ж-212, этап A, вопросы 1–6: когда HAR лучше бустинга (и наоборот), срезы по режимам, смещение, корреляция ошибок, признаки преимущества.
dq = QLIKE(HAR) − QLIKE(XJ): плюс — бустинг лучше. Без обучения и без trials. Уже посчитанное Ж-211 (hybrid_final_v1) не пересчитывается."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod

MODELS = ("HAR", "XJ", "LF")
PERIODS = (("P1_2017_2021", 2017, 2021), ("P1b_2019_2021", 2019, 2021), ("P2_2022", 2022, 2022), ("P3_2023_2025", 2023, 2025), ("P4_2026", 2026, 2026))


def tnw(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / np.sqrt(dmod.nw_variance(x)))


def year(o) -> np.ndarray:
    return pd.Series(o["target_date"]).dt.year.to_numpy()


def period_masks(o) -> dict:
    y = year(o)
    m = {"all": np.ones(len(o), bool), "no_240222": (o["target_date"] != "2022-02-24").to_numpy()}
    m.update({k: (y >= a) & (y <= b) for k, a, b in PERIODS})
    return m


def slice_row(o, mk, group: str, name: str) -> dict:
    s = o[mk]
    row = {"group": group, "slice": name, "n": int(mk.sum())}
    for m in MODELS:
        row.update({"qlike_" + m: s["q_" + m].mean(), "mean_e_" + m: s["e_" + m].mean(), "t_e_" + m: tnw(s["e_" + m]), "mean_r_" + m: (s["rv"] / s[m]).mean()})
    row.update({"mean_dq_HAR_XJ": s["dq_HAR_XJ"].mean(), "t_dq_HAR_XJ": tnw(s["dq_HAR_XJ"]), "share_HAR_better": float((s["dq_HAR_XJ"] < 0).mean()),
                "ratio_q_HAR_over_XJ": s["q_HAR"].mean() / s["q_XJ"].mean(), "corr_e_HAR_XJ": s["e_HAR"].corr(s["e_XJ"]), "corr_e_XJ_LF": s["e_XJ"].corr(s["e_LF"]),
                "corr_q_HAR_XJ": s["q_HAR"].corr(s["q_XJ"]), "corr_q_XJ_LF": s["q_XJ"].corr(s["q_LF"])})
    return row


def slice_table(o: pd.DataFrame) -> pd.DataFrame:
    """Вопросы 4–6: срезы — периоды, годы, PIT-терцили RV_w SBER и рынка, режим RVI, стресс Ж-210 (по прошлому), знак доходности d."""
    rows = [slice_row(o, mk, "period", k) for k, mk in period_masks(o).items()]
    y = year(o)
    rows += [slice_row(o, y == k, "year", str(k)) for k in sorted(set(y))]
    for col, grp, vals in (("ter_rv_w", "ter_rv_w_SBER", (0, 1, 2)), ("mkt_ter_w", "ter_rv_w_IMOEX", (0, 1, 2)), ("rvi_regime", "rvi_regime", ("low", "high"))):
        rows += [slice_row(o, (o[col] == v).to_numpy(), grp, str(v)) for v in vals]
    rows += [slice_row(o, o["st_g4"].to_numpy(bool), "stress_g4", "stress"), slice_row(o, ~o["st_g4"].to_numpy(bool), "stress_g4", "calm")]
    rows += [slice_row(o, (o["F_f3_ret1"] < 0).to_numpy(), "ret_d", "neg"), slice_row(o, (o["F_f3_ret1"] >= 0).to_numpy(), "ret_d", "nonneg")]
    r = o["e_HAR"].rolling(252).corr(o["e_XJ"]).dropna()
    rows.append({"group": "rolling252", "slice": "corr_e_HAR_XJ min/median/max", "n": len(r), "corr_e_HAR_XJ": r.min(), "corr_e_XJ_LF": r.median(), "corr_q_HAR_XJ": r.max()})
    return pd.DataFrame(rows)


def when_table(o: pd.DataFrame) -> pd.DataFrame:
    """Вопросы 1–2: когда лучше HAR, когда бустинг — по положению факта RV относительно прогнозов (ex post), по знаку расхождения и вкладу дней."""
    dq, rv = o["dq_HAR_XJ"].to_numpy(), o["rv"].to_numpy()
    hi, lo = np.maximum(o["HAR"], o["XJ"]).to_numpy(), np.minimum(o["HAR"], o["XJ"]).to_numpy()
    conds = [("factual", "RV выше обоих прогнозов", rv > hi), ("factual", "RV между прогнозами", (rv <= hi) & (rv >= lo)), ("factual", "RV ниже обоих", rv < lo),
             ("delta_sign", "HAR выше XGB (δ>0)", (o["delta_HX"] > 0).to_numpy()), ("delta_sign", "HAR ниже XGB (δ<0)", (o["delta_HX"] < 0).to_numpy()),
             ("delta_size", "|δ| верхний квартиль", (o["delta_HX"].abs() > o["delta_HX"].abs().quantile(0.75)).to_numpy()),
             ("delta_size", "|δ| нижний квартиль", (o["delta_HX"].abs() <= o["delta_HX"].abs().quantile(0.25)).to_numpy())]
    rows = []
    for g, nm, mk in conds:
        rows.append({"group": g, "condition": nm, "n": int(mk.sum()), "share": float(mk.mean()), "share_HAR_better": float((dq[mk] < 0).mean()), "mean_dq": float(dq[mk].mean()),
                     "sum_dq": float(dq[mk].sum()), "qlike_HAR": float(o["q_HAR"].to_numpy()[mk].mean()), "qlike_XJ": float(o["q_XJ"].to_numpy()[mk].mean())})
    a = np.abs(dq)
    for k in (5, 20, 1):
        idx = np.argsort(a)[::-1][:(k if k > 1 else int(0.01 * len(a)))]
        rows.append({"group": "tail_days", "condition": "топ-%s дней по |dq|" % (k if k > 1 else "1 %"), "n": len(idx), "share": float(a[idx].sum() / a.sum()),
                     "share_HAR_better": float((dq[idx] < 0).mean()), "mean_dq": float(dq[idx].mean()), "sum_dq": float(dq[idx].sum())})
    return pd.DataFrame(rows)
