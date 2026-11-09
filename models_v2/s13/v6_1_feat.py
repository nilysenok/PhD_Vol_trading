"""V6-1a, G11: загрузчик ТОЛЬКО SBER и обычная сводка S (план V6, п. 2.1). Читаются ровно два файла: бары M10 `data/iss_check/bars_10m_full/SBER.parquet` и дневной ряд `daily_SBER.parquet` (канонические дни, rv, bv, j, rv_w, rv_m).
Других бумаг, IMOEX и внешних рядов нет (`build_daily` с IMOEX не вызывается). Признаки дня d — только из баров сессий ≤ d (окна — назад, включая d). Определение RV — проекта: 1e4·Σr², r_1 = ln(c/o), далее close-к-close, бары 10:00–18:40.
S: s1–s3 = ln RV_d, ln RV_w, ln RV_m − ln H; s4 = RS⁻/RV; s5 = r_d/√RV_d (r_d = ln(close/open) сессии); s6 = J/RV; s7 = ln(Vol_d / среднее Vol за 20 сессий, включая d); s8, s9 = доля RV в первой и последней трети баров (3k < n, 3k ≥ 2n).
Единицы: ratio-признаки не зависят от масштаба; s5 — в десятичных долях (r и √Σr²)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s9 import hv2_m10 as M

NAMES = ["s%d" % i for i in range(1, 10)]
WIN_VOL = 20
TOL_ID = 1e-10


def allowed_files() -> list:
    return [P.bars_path("SBER"), P.OUT / "daily_SBER.parquet"]


def load_inputs() -> tuple:
    """-> (bars SBER, daily SBER). Единственное место чтения данных признаков (тест-аудит путей)."""
    bars, daily = pd.read_parquet(allowed_files()[0]), pd.read_parquet(allowed_files()[1])
    daily.index = pd.to_datetime(daily.index)
    return bars, daily


def day_aggregates(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары -> по сессиям (десятичные доли): rv = Σr², rs_neg, rv_first, rv_last, r_day, vol, n."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = M.bar_returns(bars)
    x = r["r"].to_numpy(float)
    q = x * x
    k = r.groupby("date").cumcount().to_numpy()
    n = r.groupby("date")["r"].transform("size").to_numpy()
    f = pd.DataFrame({"date": r["date"], "rv": q, "rs_neg": np.where(x < 0, q, 0.0), "rv_first": np.where(3 * k < n, q, 0.0), "rv_last": np.where(3 * k >= 2 * n, q, 0.0)})
    g = f.groupby("date").sum()
    day = main["begin"].dt.normalize()
    g["vol"] = main.groupby(day)["volume"].sum().reindex(g.index)
    g["r_day"] = np.log(main.groupby(day)["close"].last() / main.groupby(day)["open"].first()).reindex(g.index)
    g["n"] = r.groupby("date").size().reindex(g.index)
    return g


def vol_ratio(vol: pd.Series, win: int = WIN_VOL) -> pd.Series:
    """Vol_d / среднее Vol за последние `win` сессий, включая d (окно назад). МУТАНТ «с будущими днями» подменяет эту функцию в тесте."""
    return vol / vol.rolling(win, min_periods=win).mean()


def raw_features(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """index = дни панели; lnrv, lnrv_w, lnrv_m (из daily), s4…s9 (из баров ≤ d). Без H."""
    a = day_aggregates(bars).reindex(daily.index)
    out = pd.DataFrame(index=daily.index)
    out["lnrv"], out["lnrv_w"], out["lnrv_m"] = np.log(daily["rv"]), np.log(daily["rv_w"]), np.log(daily["rv_m"])
    out["s4"] = a["rs_neg"] / a["rv"]
    out["s5"] = a["r_day"] / np.sqrt(a["rv"])
    out["s6"] = daily["j"] / daily["rv"]
    out["s7"] = np.log(vol_ratio(a["vol"]))
    out["s8"], out["s9"] = a["rv_first"] / a["rv"], a["rv_last"] / a["rv"]
    return out


def assemble(raw: pd.DataFrame, h: pd.Series) -> pd.DataFrame:
    """S = s1…s9 на днях, где есть H: s1–s3 = ln(·) − ln H."""
    r = raw.reindex(h.index)
    lh = np.log(h.to_numpy(float))
    s = pd.DataFrame(index=h.index)
    s["s1"], s["s2"], s["s3"] = r["lnrv"] - lh, r["lnrv_w"] - lh, r["lnrv_m"] - lh
    for c in NAMES[3:]:
        s[c] = r[c]
    return s[NAMES]


def identity_max(bars: pd.DataFrame, daily: pd.DataFrame) -> float:
    """max относительная разность 1e4·Σr² по барам и RV канона (daily) на днях панели; допуск TOL_ID."""
    a = day_aggregates(bars).reindex(daily.index)["rv"] * D.SCALE
    return float((np.abs(a - daily["rv"]) / daily["rv"]).max())
