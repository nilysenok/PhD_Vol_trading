"""X1-пилот, лист P0, G2: сохраняется ли масштаб RV при нормировке. Только чтение баров и дневного ряда SBER; ничего не обучается и не пишется.
Три блока: (1) построитель M10 V7 (`v7_0_seq`): сумма за день нормированного канала r2 (и r²) против RV_d; (2) признаки V6 S (`v6_1_feat`): части дня в уровнях и в долях против RV_d;
(3) предполагаемая агрегация B(w) = SCALE·Σ_k w_k·r_k² (тот же порядок суммирования, что в `daily.build_daily`): при w ≡ 1 B = RV_d побитно; при положительных w со средним 1 B/RV лежит между min w и max w.
Строка таблицы: block, year, n_days, pearson, spearman, median_ratio, extra (смысл — в колонке note)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M
from models_v2.s13 import v6_1_feat as FEAT
from models_v2.s14 import v7_0_io as IO, v7_0_seq as SQ

YEARS = tuple(range(2018, 2026))


def agg_B(r: np.ndarray, day, w: np.ndarray | None = None) -> pd.Series:
    """B(w) = SCALE·Σ_k w_k·r_k² по дням. w = None (≡ 1) — тот же вызов, что в `daily.build_daily` ((r**2).groupby(date).sum() · SCALE): побитно RV_d."""
    q = np.asarray(r, float) ** 2
    q = q if w is None else q * np.asarray(w, float)
    return pd.Series(q).groupby(np.asarray(day)).sum() * D.SCALE


def _row(block: str, year, x: np.ndarray, rv: np.ndarray, ratio: np.ndarray, note: str, extra: float = np.nan) -> dict:
    ok = np.isfinite(x) & np.isfinite(rv)
    pe = float(np.corrcoef(x[ok], rv[ok])[0, 1]) if ok.sum() > 2 else np.nan
    sp = float(spearmanr(x[ok], rv[ok])[0]) if ok.sum() > 2 else np.nan
    return dict(block=block, year=year, n_days=int(ok.sum()), pearson=pe, spearman=sp, median_ratio=float(np.nanmedian(ratio[ok])), extra=extra, note=note)


def m10_norm_sums(store: dict, rv: pd.Series) -> pd.DataFrame:
    """На каждый день d (нормировка по сессиям ≤ d, как `x_of`): Σ_слотов z(r2), Σ_слотов z(r)², Σ r2 (сырой), RV_d, 1/σ(r2), μ(r2). Дни с norm_ok = 0 отброшены."""
    ir2, ir = SQ.CH.index("r2"), SQ.CH.index("r")
    rows = []
    for i, d in enumerate(store["days"]):
        if i + 1 < SQ.MIN_SESS or d not in rv.index:
            continue
        mu, sd = SQ.norm_stats(store, i)
        blk, pres = store["raw"][i], store["present"][i]
        z2, zr = ((blk[pres, ir2] - mu[ir2]) / sd[ir2]), ((blk[pres, ir] - mu[ir]) / sd[ir])
        rows.append((d, float(np.nansum(z2)), float(np.nansum(zr ** 2)), float(np.nansum(blk[pres, ir2])), float(rv[d]), 1.0 / sd[ir2], mu[ir2]))
    return pd.DataFrame(rows, columns=["d", "sum_z_r2", "sum_z_r_sq", "sum_raw_r2", "rv", "inv_sd_r2", "mu_r2"]).set_index("d")


def m10_table(store: dict, rv: pd.Series) -> list:
    t = m10_norm_sums(store, rv)
    out = []
    for y in YEARS:
        g = t[t.index.year == y]
        x = {k: g[k].to_numpy(float) for k in g.columns}
        out.append(_row("M10_norm_r2", y, x["sum_z_r2"], x["rv"], x["sum_z_r2"] / x["rv"], "Σ_слотов нормированного канала r2 / RV_d; extra = медиана 1/σ(r2) (масштаб, который вносит нормировка)", float(np.median(x["inv_sd_r2"]))))
        out.append(_row("M10_norm_r_sq", y, x["sum_z_r_sq"], x["rv"], x["sum_z_r_sq"] / x["rv"], "Σ_слотов нормированного канала r, в квадрате / RV_d", float(np.median(x["inv_sd_r2"]))))
        out.append(_row("M10_raw_r2", y, x["sum_raw_r2"], x["rv"], x["sum_raw_r2"] / x["rv"], "Σ_слотов сырого канала r2 / RV_d (до нормировки; тождество V7-0 G3)", float(np.max(np.abs(x["sum_raw_r2"] / x["rv"] - 1)))))
    return out


def s_table(bars: pd.DataFrame, daily: pd.DataFrame) -> list:
    """Признаки V6 S: части дня (первая / последняя треть баров) в уровнях ×SCALE и в долях s8, s9; RV_d — из daily_SBER."""
    a = FEAT.day_aggregates(bars).reindex(daily.index)
    out, rv = [], daily["rv"]
    for name, col in (("S_level_first", "rv_first"), ("S_level_last", "rv_last")):
        lv = a[col] * D.SCALE
        for y in YEARS:
            m = rv.index.year == y
            out.append(_row(name, y, lv[m].to_numpy(float), rv[m].to_numpy(float), (lv / rv)[m].to_numpy(float), "доля трети дня в RV_d × RV_d (уровень, единицы %²) против RV_d; медиана = типичная доля", float(np.nan)))
    for name, col in (("S_share_s8", "rv_first"), ("S_share_s9", "rv_last")):
        sh = a[col] / a["rv"]
        for y in YEARS:
            m = rv.index.year == y
            out.append(_row(name, y, sh[m].to_numpy(float), rv[m].to_numpy(float), sh[m].to_numpy(float), "доля (безразмерная) против RV_d; масштаб RV не несёт: корреляция с уровнем — случайная связь", float(np.nan)))
    return out


def b_tables(bars: pd.DataFrame, daily: pd.DataFrame) -> list:
    """B(w≡1) против RV_d побитно (extra = доля побитно равных дней; median_ratio — медиана B/RV) и B с положительным профилем слотов со средним 1 (диагностика: профиль по всей истории, не PIT, прогнозов нет)."""
    r = M.bar_returns(bars)
    r = r[r["date"].isin(daily.index)].reset_index(drop=True)
    B1 = agg_B(r["r"].to_numpy(), r["date"].to_numpy()).reindex(daily.index)
    slot = ((r["mins"].to_numpy() - SQ.T0) // 10).astype(int)
    q = r["r"].to_numpy(float) ** 2
    prof = pd.Series(q).groupby(slot).mean()
    p = prof.reindex(slot).to_numpy()
    w = p / pd.Series(p).groupby(r["date"].to_numpy()).transform("mean").to_numpy()      # положительные, среднее по барам дня = 1
    Bw = agg_B(r["r"].to_numpy(), r["date"].to_numpy(), w).reindex(daily.index)
    rv, out = daily["rv"], []
    for y in YEARS:
        m = rv.index.year == y
        eq = B1[m].to_numpy() == rv[m].to_numpy()
        out.append(_row("B_w1", y, B1[m].to_numpy(float), rv[m].to_numpy(float), (B1 / rv)[m].to_numpy(float), "B(w≡1) против RV_d; extra = доля побитно равных дней", float(eq.mean())))
        out.append(_row("B_profile_mean1", y, Bw[m].to_numpy(float), rv[m].to_numpy(float), (Bw / rv)[m].to_numpy(float), "B с положительным профилем слотов, среднее w = 1, B/RV; extra = доля дней B > RV", float((Bw[m] > rv[m]).mean())))
    return out


def load_inputs() -> tuple:
    bars, daily = pd.read_parquet(IO.BARS), pd.read_parquet(IO.DAILY)
    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index))
    return bars, daily


def scale_table() -> pd.DataFrame:
    bars, daily = load_inputs()
    store = SQ.build_store(bars, daily.index)
    return pd.DataFrame(m10_table(store, daily["rv"]) + s_table(bars, daily) + b_tables(bars, daily))
