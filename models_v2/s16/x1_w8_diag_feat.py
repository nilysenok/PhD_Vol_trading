"""W8-A: перенос `imoex_day` и `features` из `w8/core.py` БЕЗ изменения логики (`W8_SPEC.md` разд. 1.0). Только чтение.
Публичное API:
- `imoex_day() -> DataFrame` (индекс — день): close, rv = Σ r²_бар по барам основной сессии с началом 10:00…18:30 (архив `data/iss_check/index_10m/IMOEX.parquet`, продолжение `data/fwd/mkt_m10/IMOEX_m10_fwd.parquet`;
  первый бар дня — ln(close/open), далее лог-разность закрытий), r = ln(close_d/close_{d−1}).
- `features(x) -> (dict группы W8D1_type … W8D10_cold, info)`: группы на всех днях (используются только дни gap = 1); PIT-терцили — `terc_past`; W8D10 — z = β̂/SE пула переоценки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s16.x1_w8_diag_core import CUT_BAR, ROOT, beta_se, terc_past


def imoex_day():
    a = pd.read_parquet(ROOT / "data/iss_check/index_10m/IMOEX.parquet", columns=["begin", "open", "close"])
    n = pd.read_parquet(ROOT / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet", columns=["begin", "open", "close"])
    a["begin"], n["begin"] = pd.to_datetime(a["begin"]), pd.to_datetime(n["begin"])
    b = pd.concat([a, n[n["begin"] > a["begin"].max()]], ignore_index=True).sort_values("begin")
    tod = b["begin"] - b["begin"].dt.normalize()
    b = b[(tod >= pd.Timedelta(hours=10)) & (tod <= CUT_BAR)].copy()
    b["day"] = b["begin"].dt.normalize()
    b["lr"] = np.log(b["close"] / b["close"].groupby(b["day"]).shift(1))
    first = b.groupby("day").head(1).index
    b.loc[first, "lr"] = np.log(b.loc[first, "close"] / b.loc[first, "open"])
    g = b.groupby("day")
    out = pd.DataFrame({"close": g["close"].last(), "rv": g["lr"].apply(lambda s: float(np.nansum(s ** 2)))})
    out["r"] = np.log(out["close"] / out["close"].shift(1))
    return out


def features(x):  # xtask: длинная — перенос w8/core.py без изменения логики
    """Группы W8D1…W8D10 (строки на всех днях; используются только дни gap = 1)."""
    ka = pd.read_parquet(ROOT / C.SRC["k23_a"]).rename(columns={"date": "d"})
    kn = pd.read_parquet(ROOT / C.SRC["k23_n"]).rename(columns={"date": "d"})
    for k in (ka, kn):
        k["t_plus"] = pd.to_datetime(k["t_plus"])
    k = pd.concat([ka[ka["t_plus"] <= C.ARCH_END], kn], ignore_index=True).sort_values("t_plus")
    if k["t_plus"].duplicated().any():
        raise SystemExit("k23: дубль t⁺")
    rvmap = pd.Series(k["rv"].to_numpy(), index=k["t_plus"].to_numpy())
    dd = x["d"].to_numpy()
    rv_d = rvmap.reindex(dd).to_numpy()
    tps, rvv = rvmap.index.to_numpy(), rvmap.to_numpy()
    v = np.full(len(x), np.nan)
    for i, di in enumerate(dd):
        j = np.searchsorted(tps, di)
        if j >= 20 and np.isfinite(rv_d[i]):
            v[i] = np.log(rv_d[i]) - np.log(rvv[j - 20:j].mean())
    im = imoex_day()
    ir = im["r"].reindex(pd.DatetimeIndex(dd)).to_numpy()
    irv = im["rv"].reindex(pd.DatetimeIndex(dd)).to_numpy()
    # холодный старт: z = β̂/SE пула переоценки (t⁺ ≤ D, D = refit_date)
    z = np.full(len(x), np.nan)
    u, g, tp = x["u"].to_numpy(), x["gap"].to_numpy(), x["t_plus"].to_numpy()
    for D in x["refit"].dropna().unique():
        pool = tp <= D
        b, se = beta_se(u[pool], g[pool])
        z[(x["refit"] == D).to_numpy()] = b / se
    zg = np.where(np.isnan(z), "na", np.where(z < 1, "z<1", np.where(z < 2, "1<=z<2", "z>=2"))).astype(object)
    L = x["L"].to_numpy()
    f = {
        "W8D1_type": x["type"].to_numpy(object),
        "W8D2_len": np.where(L >= 3, "L>=3", np.where(L == 2, "L=2", "L=1")).astype(object),
        "W8D3_holiday": np.where(x["type"].to_numpy() == "weekend", "weekend", "holiday").astype(object),
        "W8D4_lvl": terc_past(np.log(rv_d)),
        "W8D5_shock": terc_past(v),
        "W8D6_e": np.where(x["s"].to_numpy(int) == 1, "e1", "e0").astype(object),
        "W8D7_imoex_absr": terc_past(np.abs(ir)),
        "W8D8_imoex_rv": terc_past(np.log(np.where(irv > 0, irv, np.nan))),
        "W8D9_season": x["season"].to_numpy(object),
        "W8D10_cold": zg,
    }
    info = dict(rv_d_na=int(np.isnan(rv_d).sum()), imoex_na=int(np.isnan(ir).sum()), imoex_na_gap=int((np.isnan(ir) & (g == 1)).sum()), z=z)
    return f, info
