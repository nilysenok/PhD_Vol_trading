"""Ж-214, этап V (п. 15 блока): диагностика сезонности и качества измерения RV. Это ИЗМЕРЕНИЕ, не прогноз: на ошибки моделей не смотрит.
Единицы: r — в %, r² — в %² (как RV). Слоты — hv2_seas (0..52). Полные дни для сигнатуры: слоты 0..49 без дыр."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_sched as S
from models_v2.s9 import hv2_seas as Z


def _hhmm(slot: int) -> str:
    m = S.START_MIN + slot * S.BAR_MIN
    return "%02d:%02d" % (m // 60, m % 60)


def profile_table(M2: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Среднее r² слота и доля слота в сумме средних, по годам и за всё время."""
    rows = []
    for y, ix in [("all", np.arange(len(index)))] + [(int(k), np.where(index.year == k)[0]) for k in sorted(set(index.year))]:
        mu = np.nanmean(M2[ix], axis=0)
        for k in range(Z.NSLOT):
            rows.append({"year": y, "slot": k, "time": _hhmm(k), "mean_r2": mu[k], "share": mu[k] / np.nansum(mu)})
    return pd.DataFrame(rows)


def profile_summary(pt: pd.DataFrame) -> pd.DataFrame:
    """По годам: отношение первого / последнего слота к медиане слотов 1..49, корреляция профиля года с общим."""
    allp = pt[pt["year"] == "all"].set_index("slot")["mean_r2"]
    rows = []
    for y, g in pt[pt["year"] != "all"].groupby("year"):
        g = g.set_index("slot")["mean_r2"]
        mid = float(g.loc[1:49].median())
        last = int(g.last_valid_index())
        rows.append({"year": y, "first_over_median": float(g.loc[0] / mid), "last_slot": _hhmm(last), "last_over_median": float(g.loc[last] / mid),
                     "corr_with_all": float(np.corrcoef(g.loc[0:51].fillna(0), allp.loc[0:51].fillna(0))[0, 1])})
    return pd.DataFrame(rows)


def signature(R: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Сигнатура волатильности по полным дням (слоты 0..49): RV10/RV20, RV10/RV30 (одинаковые окна по 48 слотов, усреднение сдвигов),
    автокорреляция лага 1 доходностей. >1 в отношениях и отрицательная АК — признак шума микроструктуры."""
    full = np.isfinite(R[:, :50]).all(axis=1)
    rows = []
    for y in sorted(set(index.year)):
        r = R[full & (index.year == y)][:, :50]
        if len(r) < 20:
            continue
        rv10 = np.mean([(r[:, o:o + 48] ** 2).sum(axis=1).mean() for o in range(3)])
        rv20 = np.mean([(r[:, o:o + 48].reshape(len(r), 24, 2).sum(axis=2) ** 2).sum(axis=1).mean() for o in range(3)])
        rv30 = np.mean([(r[:, o:o + 48].reshape(len(r), 16, 3).sum(axis=2) ** 2).sum(axis=1).mean() for o in range(3)])
        ac1 = float((r[:, 1:] * r[:, :-1]).sum() / (r ** 2).sum())
        rows.append({"year": y, "days": len(r), "rv10_over_rv20": rv10 / rv20, "rv10_over_rv30": rv10 / rv30, "ac1": ac1})
    return pd.DataFrame(rows)


def quality(M2: np.ndarray, R: np.ndarray, index: pd.DatetimeIndex, raw: pd.DataFrame) -> pd.DataFrame:
    """По годам: нулевые доходности, плоские бары (high=low), нулевой объём, доля первого/последнего слота в RV, дни с одним баром > 30 %/50 % RV."""
    m = D.main_session(raw)
    y = m["begin"].dt.year
    tot = np.nansum(M2, axis=1)
    mx = np.nanmax(np.where(np.isfinite(M2), M2, -1.0), axis=1) / tot
    rows = []
    for k in sorted(set(index.year)):
        ix = index.year == k
        rr = R[ix]
        fin = np.isfinite(rr)
        mk = (y == k)
        last = np.where(np.isfinite(M2[ix][:, 52]), M2[ix][:, 52], M2[ix][:, 51])
        rows.append({"year": k, "days": int(ix.sum()), "zero_ret_share": float(((rr == 0) & fin).sum() / fin.sum()),
                     "flat_bar_share": float((m["high"][mk] == m["low"][mk]).mean()), "zero_volume_bars": int((m["volume"][mk] == 0).sum()),
                     "first_slot_share": float(np.nanmean(M2[ix][:, 0] / tot[ix])), "last_slot_share": float(np.nanmean(last / tot[ix])),
                     "days_maxbar_gt30": int((mx[ix] > 0.3).sum()), "days_maxbar_gt50": int((mx[ix] > 0.5).sum())})
    return pd.DataFrame(rows)


def _jump(raw, index, ta, ca, tb, cb, main_mean) -> pd.Series:
    """Средний квадрат скачка ln(цена_b / цена_a) между баром ta (колонка ca) и баром tb (колонка cb) того же дня, в долях среднего r² основных баров."""
    t, day = raw["begin"].dt.strftime("%H:%M"), raw["begin"].dt.normalize()
    ok = day.isin(index)
    a = raw[(t == ta) & ok].set_index(day[(t == ta) & ok])[ca]
    b = raw[(t == tb) & ok].set_index(day[(t == tb) & ok])[cb]
    j = (np.log(b / a).dropna() ** 2) * D.SCALE
    return j.groupby(j.index.year).mean() / main_mean


def edge_and_gap(raw: pd.DataFrame, index: pd.DatetimeIndex, rv: pd.Series) -> pd.DataFrame:
    """Исключённые из канона: бары аукциона 09:50 и 18:50 (r² = ln(c/o)², относительно среднего r² основных баров) и ночной гэп (в RV не входит):
    средний гэп² / RV по годам."""
    t = raw["begin"].dt.strftime("%H:%M")
    day = raw["begin"].dt.normalize()
    r2 = (np.log(raw["close"] / raw["open"]) ** 2) * D.SCALE
    ok = day.isin(index)
    main_mean = r2[(t >= "10:00") & (t <= "18:40") & ok].groupby(raw["begin"].dt.year).mean()
    out = pd.DataFrame({"main_mean_r2": main_mean})
    for lab in ("09:50", "18:50"):
        s = r2[(t == lab) & ok].groupby(raw["begin"].dt.year)
        out[lab + "_n"], out[lab + "_over_main"] = s.size(), s.mean() / main_mean
    out["open_auction_jump_over_main"] = _jump(raw, index, "09:50", "close", "10:00", "open", main_mean)
    out["close_auction_jump_over_main"] = _jump(raw, index, "18:40", "close", "19:00", "close", main_mean)
    main = D.main_session(raw).sort_values("begin")
    g = main.groupby(main["begin"].dt.normalize()).agg(o=("open", "first"), c=("close", "last"))
    g = g[g.index.isin(index)]
    gap2 = (np.log(g["o"] / g["c"].shift(1)) ** 2 * D.SCALE).where(g.index.to_series().diff().dt.days <= 5)
    ratio = (gap2 / rv.reindex(g.index)).groupby(g.index.year).mean()
    out["overnight_gap2_over_rv"] = ratio
    return out.reset_index().rename(columns={"begin": "year", "index": "year"})


def sa_compare(t: pd.DataFrame) -> pd.DataFrame:
    """RV_sa против RV по годам: среднее отношение, корреляция ln, АК лагов 1/5 ln RV и ln RV_sa (внутри дней с обоими значениями)."""
    ok = t[["rv", "rv_sa"]].dropna()
    rows = []
    for y, g in ok.groupby(ok.index.year):
        a, b = np.log(g["rv"]), np.log(g["rv_sa"])
        rows.append({"year": y, "days": len(g), "mean_ratio": float((g["rv_sa"] / g["rv"]).mean()), "corr_ln": float(a.corr(b)),
                     "ac1_rv": float(a.autocorr(1)), "ac1_rv_sa": float(b.autocorr(1)), "ac5_rv": float(a.autocorr(5)), "ac5_rv_sa": float(b.autocorr(5))})
    return pd.DataFrame(rows)
