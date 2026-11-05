"""B2: дневные агрегаты баров (F3 ret1, F4, F5, объём). Каждый день считается только по барам этого дня (и close предыдущего)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models_v2 import daily

MEDRV_C = math.pi / (6 - 4 * math.sqrt(3) + math.pi)
LN2x4 = 4 * math.log(2)


def _main(bars: pd.DataFrame) -> tuple:
    main = daily.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = daily.intraday_returns(main)
    return main, r["r"].to_numpy(float), r["date"]


def day_stats(bars: pd.DataFrame, dates) -> pd.DataFrame:
    """Строки dates: rq, medrv, park, range2 (%²-единицы, до логарифмов), hhi, close, volume."""
    main, r, day = _main(bars)
    r100, n = 100.0 * r, pd.Series(r).groupby(day).size().astype(float)
    g = lambda v: pd.Series(v).groupby(day)  # noqa: E731
    a = pd.Series(np.abs(r))
    ga = a.groupby(day)
    prv, nxt = ga.shift(1), ga.shift(-1)
    med = pd.concat([prv, a, nxt], axis=1).median(axis=1).where(prv.notna() & nxt.notna())
    out = pd.DataFrame({"rq": n / 3 * g(r100 ** 4).sum()})
    out["medrv"] = MEDRV_C * n / (n - 2) * (med ** 2).groupby(day).sum() * 1e4
    out["park"] = g(np.log(main["high"].to_numpy() / main["low"].to_numpy()) ** 2).sum() / LN2x4 * 1e4
    hl = main.groupby(day)
    out["range2"] = (100.0 * np.log(hl["high"].max() / hl["low"].min())) ** 2
    rv_h = pd.Series(r ** 2).groupby([day, main["begin"].dt.hour.to_numpy()]).sum()
    tot = rv_h.groupby(level=0).sum()
    out["hhi"] = ((rv_h / tot.reindex(rv_h.index.get_level_values(0)).to_numpy()) ** 2).groupby(level=0).sum()
    out["close"], out["volume"] = hl["close"].last(), hl["volume"].sum().astype(float)
    return out.reindex(pd.DatetimeIndex(dates))


def cross_stats(bars: pd.DataFrame, imoex: pd.DataFrame, dates) -> pd.DataFrame:
    """SBER × IMOEX на общих begin: sxy, sss, smm по дням; плюс close и дневной ряд IMOEX (rv, rv_w, rv_m) на dates."""
    ms, rs, _ = _main(bars)
    mi, ri, _ = _main(imoex)
    s = pd.Series(rs, index=pd.DatetimeIndex(ms["begin"]))
    m = pd.Series(ri, index=pd.DatetimeIndex(mi["begin"]))
    j = pd.concat([s.rename("s"), m.rename("m")], axis=1, join="inner")
    day = j.index.normalize()
    out = pd.DataFrame({"sxy": (j.s * j.m).groupby(day).sum(), "sss": (j.s ** 2).groupby(day).sum(), "smm": (j.m ** 2).groupby(day).sum()})
    idly, _ = daily.build_daily(imoex, index_days=None)
    ic = mi.groupby(mi["begin"].dt.normalize())["close"].last()
    out["mret"] = 100.0 * np.log(ic / ic.shift(1)).reindex(out.index)
    for c in ("rv", "rv_w", "rv_m"):
        out["i_" + c] = idly[c].reindex(out.index)
    return out.reindex(pd.DatetimeIndex(dates))
