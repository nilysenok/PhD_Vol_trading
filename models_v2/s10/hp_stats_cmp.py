"""P3, G11: `COMPARISON.csv` и `PER_TICKER.csv` (R12): QLIKE, RMSE, MAE, R² (на RV и на ln RV) по бумаге и по группам M16 / SBER / E11 / все 28 / сектора (среднее и медиана по бумагам),
срезы: all, 2017–2021, 2022, 2023–2025, 2026, 2022–2026 (подтверждение), без 24.02.2022, без gap, без 5 дат с наибольшим QLIKE K1, по годам; доли бумаг лучше K1, K4, K5.
K2/K3 — на всех 28 бумагах (`k23.parquet`, Ж-219 п. 5, G15; прежде — только 4, У-9): срезы с префиксом `k23_` — общие даты бумаги всех 27 рядов и K2, K3 (цели с 2017, без строк через остановку 2022, где K2/K3 нет).
Все ряды — на ОБЩИХ датах бумаги. Статус: исследовательский, HISTORICAL WALK-FORWARD, потолок."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_hyb_out as O, hp_paths as P, hp_stats_sel as S

SIGS = S.model_cols()
K23 = list(P.TICKERS)       # бумаги, у которых есть K2/K3 (G15: все 28; до Ж-219 п. 5 — четыре)
K23_PREF = "k23_"           # префикс периода/окна срезов с K2, K3 (общие даты уже с ними)
D0 = pd.Timestamp("2022-02-24")
METR = ["qlike", "rmse", "mae", "r2_rv", "r2_ln"]


def own_ctrl(model: str) -> str:
    return "k4" if model.startswith("xgb_") else "k5"


def k23_frame() -> pd.DataFrame:
    """K2/K3 по бумагам, где они есть: `k23_<T>.parquet` (Ж-220 (б): собранный `k23.parquet` не требуется; если он есть — читается он). Нет ни одного — пустая таблица."""
    cols = S.KEY + ["sigma2_k2", "sigma2_k3"]
    full = P.OUT / "k23.parquet"
    if full.exists():
        return pd.read_parquet(full)[cols]
    parts = [pd.read_parquet(P.OUT / ("k23_%s.parquet" % t))[cols] for t in K23 if (P.OUT / ("k23_%s.parquet" % t)).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({c: [] for c in cols})


def k23_tickers() -> list:
    """Бумаги из K23, у которых есть K2/K3 (число — в каждой таблице сравнений с ними)."""
    return sorted(set(k23_frame()["ticker"]) & set(K23))


def with_k23(k: pd.DataFrame) -> pd.DataFrame:
    """Добавляет k2, k3 (NaN там, где K2/K3 нет: нет бумаги или замороженная процедура не прогнозирует)."""
    x = k23_frame().rename(columns={"sigma2_k2": "k2", "sigma2_k3": "k3"})
    return k.merge(x, on=S.KEY, how="left")


def qrow(c: pd.DataFrame, s: str) -> np.ndarray:
    r = c["rv_tplus"].to_numpy(float) / c[s].to_numpy(float)
    return r - np.log(r) - 1.0


def top_dates(dates, v, n: int = 5) -> np.ndarray:
    """n дат с наибольшим панельным (среднее по бумагам дня) значением v по строкам."""
    return pd.Series(np.asarray(v, float), index=np.asarray(dates)).groupby(level=0).mean().nlargest(n).index.to_numpy()


def slice_masks(c: pd.DataFrame) -> dict:
    y, t = c["t_plus"].dt.year, c["t_plus"]
    m = {"all": np.ones(len(c), bool), "p2017_2021": (y <= 2021).to_numpy(), "y2022": (y == 2022).to_numpy(), "p2023_2025": ((y >= 2023) & (y <= 2025)).to_numpy(), "y2026": (y == 2026).to_numpy(),
         "confirm_2022_2026": (y >= 2022).to_numpy(), "no_240222": ((t != D0) & (c["date"] != D0)).to_numpy(), "no_gap": ~c["gap"].to_numpy(bool)}
    m["no_top5dates_k1"] = ~c["date"].isin(top_dates(c["date"], qrow(c, "k1"))).to_numpy()
    m.update({"y%d" % v: (y == v).to_numpy() for v in sorted(y.unique())})
    return m


def per_ticker(c: pd.DataFrame, sigs: list, masks: dict) -> pd.DataFrame:
    """Метрики бумаги в каждом срезе: ticker, group, sector, period, model, n, qlike, rmse, mae, r2_rv, r2_ln (формулы `hp_hyb_out`)."""
    m = O._row_metrics(c.rename(columns={s: "sigma2_" + s for s in sigs}), sigs)
    pt = pd.concat([O._per_ticker(m[mk], sigs, name) for name, mk in masks.items() if mk.any()], ignore_index=True)
    return pt.assign(sector=pt["ticker"].map(P.SECTOR))


def groups_of(pt: pd.DataFrame) -> list:
    g = [("M16", pt["group"] == "M16"), ("SBER", pt["group"] == "SBER"), ("E11", pt["group"] == "E11"), ("all28", pt["group"].notna())]
    return g + [("sector:" + s, pt["sector"] == s) for s in sorted(pt["sector"].dropna().unique())]


def comparison(pt: pd.DataFrame, extra_ctrl: bool = False) -> pd.DataFrame:
    """Агрегаты по группам: среднее и медиана по бумагам, число бумаг, доли бумаг с QLIKE ниже K1, K4, K5 (и K2, K3 в срезе k23_common)."""
    q = pt.pivot_table(index=["ticker", "period"], columns="model", values="qlike")
    refs = ["k1", "k4", "k5"] + (["k2", "k3"] if extra_ctrl else [])
    flags = {(mn, r): ((q[mn] < q[r]).where(q[mn].notna() & q[r].notna()).astype(float) if mn != r else pd.Series(np.nan, index=q.index)) for r in refs for mn in q.columns}
    rows = []
    for name, sel in groups_of(pt):
        x = pt[sel]
        for (per, mod), g in x.groupby(["period", "model"]):
            row = dict(scope=name, period=per, model=mod, n_tickers=g["ticker"].nunique(), n=int(g["n"].sum()))
            row.update({m_ + "_mean": g[m_].mean() for m_ in METR})
            row.update({m_ + "_median": g[m_].median() for m_ in ("qlike", "rmse", "mae")})
            idx = pd.MultiIndex.from_arrays([g["ticker"], g["period"]])
            for r in refs:
                row["frac_better_" + r] = flags[(mod, r)].reindex(idx).mean() if mod != r else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def tables(k: pd.DataFrame) -> tuple:
    """-> (per-ticker всех срезов, COMPARISON, общие строки): K2/K3 — на бумагах из `k23_tickers()` (число — в `n_k23_tickers`); без пустых срезов K2/K3."""
    c = S.common(k, SIGS)
    pt = per_ticker(c, SIGS, slice_masks(c))
    cmp_ = comparison(pt)
    k2 = S.common(with_k23(k)[k["ticker"].isin(k23_tickers()).to_numpy()], SIGS + ["k2", "k3"])
    if len(k2):
        pt2 = per_ticker(k2, SIGS + ["k2", "k3"], {K23_PREF + n: m for n, m in slice_masks(k2).items()})
        cmp2 = comparison(pt2, True)
        cmp2["scope"] = cmp2["scope"].map(lambda s: "K23:" + s)
        cmp2["n_k23_tickers"] = cmp2["n_tickers"]                 # (б): бумаг с K2/K3 в области строки — в каждой таблице
        pt, cmp_ = pd.concat([pt, pt2], ignore_index=True), pd.concat([cmp_, cmp2], ignore_index=True)
    return pt, cmp_, c


def run(k: pd.DataFrame, log=print) -> pd.DataFrame:
    """-> таблица per-ticker (всех срезов); пишет COMPARISON.csv и PER_TICKER.csv."""
    pt, cmp_, c = tables(k)
    stat = "%s; QLIKE: среднее по бумагам; no_top5dates_k1 — без 5 дат с наибольшим панельным QLIKE K1; no_240222 — без пар с датой или целью 24.02.2022" % S.STAT
    nk = "K2/K3 на %d бумагах: %s" % (len(k23_tickers()), ",".join(k23_tickers()))
    P.write_csv(cmp_, "statistics/COMPARISON.csv", "%s; %s" % (stat, nk))
    P.write_csv(pt, "statistics/PER_TICKER.csv", "метрики по бумаге и срезам, все 25 рядов на общих датах бумаги; k23_* — общие даты со всеми рядами и K2, K3 (%s); %s" % (nk, S.STAT))
    log("COMPARISON %d строк, PER_TICKER %d строк; общих строк %d; бумаг %d" % (len(cmp_), len(pt), len(c), c["ticker"].nunique()))
    return pt
