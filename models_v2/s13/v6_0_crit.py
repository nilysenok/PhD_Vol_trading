"""V6-0, G7/G8: CRIT_SBER (crit_SBER, без 2017, без 2017–2018, без 1 % крайних, лет из 5, EqY; формула весов воспроизводит `crit_wf` ≤ 1e-12), шум меры (круговой блочный бутстрап: блок 20, 2000 повторов, seed 0)
и пороги «go» формы GATES_V6.md (δ = 0,0005). Мин-планка — по рядам класса «PIT-сопоставим» (контроли K1, K2, K3, M, K8, D2 и PIT-сопоставимые прежние гибриды). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s12 import v5_0_crit as CR
from models_v2.s13 import v6_0_best as BE, v6_0_q as Q

DELTA = Q.MARGIN
B, SEED, BLOCK = 2000, 0, 20
BASE5 = ["K1", "K2", "K3", "K8", "D2"]


def rows_for(w: pd.DataFrame, series: list, cls: dict) -> pd.DataFrame:
    """Меры отбора на датах окна 2017–2021 (цели). Для лет и тримминга «лучший из K2/K3» — по дате/году."""
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    rv, tp = w["rv"].to_numpy()[m], w["t_plus"].to_numpy()[m]
    qk2, qk3 = Q.ql(rv, w["K2"].to_numpy()[m]), Q.ql(rv, w["K3"].to_numpy()[m])
    out = []
    for s in series:
        r = Q.crit_row(rv, w[s].to_numpy()[m], tp, qk2, qk3)
        out.append(dict(series=s, class_=cls[s], **r))
    return pd.DataFrame(out)


def formula_check(w: pd.DataFrame, series: list) -> float:
    """max |crit_wf − аналитическая формула V5 (`pen_parts`)| по рядам: веса лет и штраф воспроизведены (≤ 1e-12)."""
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    d = 0.0
    for s in series:
        q = Q.ql(w["rv"].to_numpy()[m], w[s].to_numpy()[m])
        ann = Q.annual(q, w["t_plus"].to_numpy()[m])
        d = max(d, abs(Q.crit_sber(q, w["t_plus"].to_numpy()[m]) - CR.pen_parts([ann[y] for y in Q.YEARS])[2]))
    return d


def noise(w: pd.DataFrame, best: str) -> dict:
    """СКО бутстрапа разности QLIKE K8 − лучший из K2/K3 (ряд с меньшим crit_SBER) за цели 2017–2021; различимый эффект = 1,96·СКО в % среднего QLIKE K1 окна."""
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    rv = w["rv"].to_numpy()[m]
    d = Q.ql(rv, w["K8"].to_numpy()[m]) - Q.ql(rv, w[best].to_numpy()[m])
    bm = SD.boot_means(d, SD.block_idx(len(d), B_=B, block=BLOCK, seed=SEED))
    k1 = float(Q.ql(rv, w["K1"].to_numpy()[m]).mean())
    sd = float(bm.std(ddof=1))
    return dict(best_k23=best, n=len(d), mean_diff=float(d.mean()), sd_boot=sd, k1_qlike=k1, mde_95_pct_of_k1=100 * 1.96 * sd / k1, sd_pct_of_k1=100 * sd / k1, mean_diff_pct_of_k1=100 * float(d.mean()) / k1, B=B, block=BLOCK, seed=SEED)


def thresholds(t: pd.DataFrame, go: list) -> dict:
    """Условия 1–4 числами: min по планке − δ; условие 5 — годовые QLIKE K2 и K3 и их минимум по году. Планка — ряды `go`."""
    g = t[t["series"].isin(go)]

    def mn(col, sub=g):
        i = sub[col].idxmin()
        return {"value": float(sub.loc[i, col]), "series": sub.loc[i, "series"]}
    th = {"delta": DELTA, "go_series": list(go), "n_go_series": len(go)}
    for k, col in (("cond1_crit", "crit"), ("cond2_crit_no2017", "crit_no2017"), ("cond3_crit_no2017_18", "crit_no2017_18"), ("cond4_crit_trim1", "crit_trim1")):
        m = mn(col)
        b5 = mn(col, t[t["series"].isin(BASE5)])
        th[k] = {"threshold": m["value"] - DELTA, "min_over_go": m["value"], "attained_by": m["series"], "base5_threshold": b5["value"] - DELTA, "base5_attained_by": b5["series"]}
    qy = {y: {"K2": float(t.set_index("series").loc["K2", "q%d" % y]), "K3": float(t.set_index("series").loc["K3", "q%d" % y])} for y in Q.YEARS}
    th["cond5_years_better_min"] = 3
    th["cond5_annual_best_k23"] = {str(y): min(v.values()) for y, v in qy.items()}
    th["cond5_annual_k2_k3"] = {str(y): v for y, v in qy.items()}
    return th


def crit_table(w: pd.DataFrame, prior: pd.DataFrame, bres: dict) -> tuple:
    """-> (CRIT_SBER, резюме с порогами, шумом, проверкой формулы)."""
    pit, ceil = bres["pit"], bres["full"]          # ряды с неполным покрытием общих дат в меры отбора не входят
    cls = {s: "контроль" for s in BE.CTRL} | {s: c for s, c in zip(prior["series"], prior["class"])}
    cls.update({s: "контроль" for s in BE.CTRL})
    t = rows_for(w, pit + ceil, cls)
    t["in_go_min"] = t["series"].isin(pit)
    th = thresholds(t, pit)
    best = min(("K2", "K3"), key=lambda s: float(t.set_index("series").loc[s, "crit"]))
    nz = noise(w, best)
    fc = formula_check(w, pit[:40])
    return t, dict(thresholds=th, noise=nz, formula_max_diff=fc, table=t)
