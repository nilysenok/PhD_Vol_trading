"""V4-DX, G26, пункт 7 Ж-225 п. 7: какие признаки Z связаны с D = QLIKE(H) − QLIKE(M) (M = √(K2·K3), H = K1) и с попаданием RV в диапазон [min(H, M), max(H, M)] — ЗАДНИМ ЧИСЛОМ, описательно, в выбор не идёт.
Условные средние D по квинтилям признака; границы квинтилей — по распределению ПРОШЛОГО окна: строки M16 с датой строго до 1 января года среза (расширяющееся окно до начала года), поэтому граница не знает будущего;
для 2017 прошлого нет — квинтили не определены (строки 2017 не входят). Окна — те же строки с годовыми границами. Дни после структурных изломов 06.04.2018, 09.03.2020, 24.02.2022 (фиксированы заранее):
флаг = дата в [излом; излом + 28 календарных дней] (≈ 20 торговых дней), «квинтиль» 0/1. Признаки — `v4_z_feat.ALL` (Z_<T>.parquet, PIT, известны на конец дня d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_cache as BC, v4_dx_years as DY, v4_io as IO, v4_z_feat as ZF

BREAKS = ("2018-04-06", "2020-03-09", "2022-02-24")
POST_DAYS = 28
FEATS = ZF.ALL


def load_z(tickers) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(BC.OUT / ("Z_%s.parquet" % t)) for t in tickers], ignore_index=True)


def frame(P: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    """Строки M16 с D, индикатором «RV в диапазоне» и признаками Z (ключ ticker, date)."""
    p = P[P["ticker"].isin(IO.M16)].reset_index(drop=True)
    m = DY.MS["KG"](p)
    rv, h = p["rv"].to_numpy(float), p["h"].to_numpy(float)
    p = p.assign(d=DY.ql(rv, h) - DY.ql(rv, m), in_range=((rv >= np.minimum(h, m)) & (rv <= np.maximum(h, m))).astype(float), year=p["t_plus"].dt.year, qh=DY.ql(rv, h), qm=DY.ql(rv, m))
    p = p.merge(Z[["ticker", "date"] + FEATS], on=["ticker", "date"], how="left")
    for b in BREAKS:
        dt0 = pd.Timestamp(b)
        p["post_" + b] = ((p["date"] >= dt0) & (p["date"] <= dt0 + pd.Timedelta(days=POST_DAYS))).astype(float)
    return p


def quintile_labels(f: pd.DataFrame, feat: str, year: int) -> pd.Series:
    """Номер квинтиля 1…5 строк года `year` по границам прошлого окна (даты < 1 января года); нет прошлого — NaN."""
    past = f.loc[f["date"] < pd.Timestamp(year=year, month=1, day=1), feat].dropna()
    cur = f[f["year"] == year]
    if len(past) < 100:
        return pd.Series(np.nan, index=cur.index)
    edges = past.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    lab = np.searchsorted(edges, cur[feat].to_numpy(float), side="right") + 1.0
    return pd.Series(np.where(np.isfinite(cur[feat].to_numpy(float)), lab, np.nan), index=cur.index)


def _agg(g: pd.DataFrame) -> dict:
    return {"n": len(g), "mean_d": float(g["d"].mean()), "share_d_neg": float((g["d"] < 0).mean()), "share_in_range": float(g["in_range"].mean()), "mean_q_h": float(g["qh"].mean()), "mean_q_m": float(g["qm"].mean())}


def features_table(f: pd.DataFrame) -> pd.DataFrame:
    """DX_FEATURES: срез × признак × квинтиль (1…5; для post_* — 0/1; «all» — Спирмен признака с D и с индикатором попадания в диапазон по строкам среза с определёнными квинтилями)."""
    rows = []
    wins = {"Y%d" % y: [y] for y in range(2017, 2027)} | {"2018_2021": [2018, 2019, 2020, 2021], "2022_2025": [2022, 2023, 2024, 2025]}
    for feat in FEATS:
        lab = pd.concat([quintile_labels(f, feat, y) for y in range(2017, 2027)]).reindex(f.index)
        for name, years in wins.items():
            s = f[f["year"].isin(years)].assign(q=lab)
            s = s[s["q"].notna()]
            if len(s) < 100:
                continue
            for q, g in s.groupby("q"):
                rows.append({"slice": name, "feature": feat, "quintile": str(int(q)), **_agg(g), "spearman_d": np.nan, "spearman_in_range": np.nan})
            rows.append({"slice": name, "feature": feat, "quintile": "all", **_agg(s), "spearman_d": float(s[feat].corr(s["d"], method="spearman")), "spearman_in_range": float(s[feat].corr(s["in_range"], method="spearman"))})
    for b in BREAKS:
        col = "post_" + b
        for name, years in wins.items():
            s = f[f["year"].isin(years)]
            for flag, g in s.groupby(col):
                if len(g) >= 30 and (flag == 0 or flag == 1):
                    rows.append({"slice": name, "feature": col, "quintile": str(int(flag)), **_agg(g), "spearman_d": np.nan, "spearman_in_range": np.nan})
    return pd.DataFrame(rows)
