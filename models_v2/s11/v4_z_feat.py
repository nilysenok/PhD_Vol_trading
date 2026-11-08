"""V4-C, Z: PIT-признаки состояния (Ж-222 этап C, объявлено в GATES_V4.md). Все признаки строки d известны на конец дня d; прогноз — на t⁺ = d+1.
z_div: ell = ln(M/H), |ell|; z_perf: EWMA (полураспад 20 пар) QLIKE(H) − QLIKE(M) по парам с t⁺ ≤ d, ln(RV_d/H_{d−1}), ln(RV_d/M_{d−1}); z_mkt: медиана по бумагам ln RV_d, Δ ln RV бумаги, J_d/RV_d;
z_new: рыночный HAR-surprise = медиана по бумагам ln(RV_d/H_{d−1}), доля RV дня в последний час, Херфиндаль баров. Поперечные медианы — только по 17 бумагам (MAIN17), в LOTO — по 16.
Пары (d−1 → d) берутся по t⁺ == d: если предыдущей строки нет (дыра календаря), признак NaN (не заполняется). `lead` — только для теста-мутации (признак дня d+lead); боевое значение 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_io as IO

BLOCKS = {"z_div": ["ell", "abs_ell"], "z_perf": ["ewqd", "lerr_h", "lerr_m"], "z_mkt": ["med_lrv", "dlrv", "jshare"], "z_new": ["sur_mkt", "lasth", "herf"]}
CFG_BLOCKS = {"C1": ["z_div"], "C2": ["z_div", "z_perf"], "C3": ["z_div", "z_perf", "z_mkt"], "C4": ["z_div", "z_perf", "z_mkt", "z_new"]}
ALL = [f for b in BLOCKS.values() for f in b]
CROSS = ["med_lrv", "sur_mkt"]
HALFLIFE = 20


def cfg_feats(cfg: str) -> list:
    return [f for b in CFG_BLOCKS[cfg] for f in BLOCKS[b]]


def _qlike(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def _at_dates(s: pd.Series, dates) -> np.ndarray:
    return s.reindex(pd.DatetimeIndex(dates)).to_numpy(float)


def _ewqd(a: pd.DataFrame, M: np.ndarray) -> np.ndarray:
    """EWMA разности QLIKE(H) − QLIKE(M) по парам, закрытым к концу дня d (t⁺ ≤ d)."""
    qd = pd.Series(_qlike(a["rv"], a["h"]) - _qlike(a["rv"], M)).ewm(halflife=HALFLIFE).mean().to_numpy()
    tp = a["t_plus"].to_numpy("datetime64[ns]")
    pos = np.searchsorted(tp, a["date"].to_numpy("datetime64[ns]"), side="right") - 1
    return np.where(pos >= 0, qd[np.maximum(pos, 0)], np.nan)


def own(a: pd.DataFrame, daily: pd.DataFrame, herf: pd.Series, lead: int = 0) -> pd.DataFrame:
    """Собственные признаки бумаги по строкам a (date, t_plus, rv, h, k2, k3; отсортированы по дате). Колонки: ALL без поперечных + lrv (компонента медианы)."""
    M = np.sqrt(a["k2"].to_numpy(float) * a["k3"].to_numpy(float))
    ell = np.log(M / a["h"].to_numpy(float))
    ti = pd.DatetimeIndex(a["t_plus"])
    pe_h = pd.Series(np.log(a["rv"].to_numpy(float) / a["h"].to_numpy(float)), index=ti)
    pe_m = pd.Series(np.log(a["rv"].to_numpy(float) / M), index=ti)
    lrv = np.log(daily["rv"].where(daily["rv"] > 0))
    z = pd.DataFrame({"ell": ell, "abs_ell": np.abs(ell), "ewqd": _ewqd(a, M), "lerr_h": _at_dates(pe_h, a["date"]), "lerr_m": _at_dates(pe_m, a["date"]),
                      "lrv": _at_dates(lrv, a["date"]), "dlrv": _at_dates(lrv.diff(), a["date"]), "jshare": _at_dates((daily["j"] / daily["rv"]).where(daily["rv"] > 0), a["date"]),
                      "lasth": _at_dates(daily["rv_close"] / daily["rv"].where(daily["rv"] > 0), a["date"]), "herf": _at_dates(herf, a["date"])}, index=a.index)
    return z.shift(-lead) if lead else z


def assert_cross_set(tickers) -> None:
    """Поперечные признаки — только по 17 бумагам (E11 и прочие не входят, У-25)."""
    bad = sorted(set(tickers) - set(IO.T17))
    if bad:
        raise ValueError("поперечные признаки: бумаги вне 17: %s" % bad)


def cross(zl: pd.DataFrame, tickers) -> pd.DataFrame:
    """Медианы по бумагам `tickers` на каждую дату: med_lrv (медиана ln RV_d), sur_mkt (медиана ln(RV_d/H_{d−1})). zl: ticker, date, lrv, lerr_h. Индекс — дата."""
    assert_cross_set(tickers)
    s = zl[zl["ticker"].isin(list(tickers))]
    g = s.groupby("date")
    return pd.DataFrame({"med_lrv": g["lrv"].median(), "sur_mkt": g["lerr_h"].median()})


def recross(zl: pd.DataFrame, tickers) -> pd.DataFrame:
    """Длинная таблица с поперечными колонками, пересчитанными по `tickers` (LOTO: 16). Строки всех бумаг получают медиану по этому набору на свою дату."""
    c = cross(zl, tickers)
    out = zl.drop(columns=CROSS, errors="ignore")
    return out.join(c, on="date")


def build_panel_z(P: pd.DataFrame, inputs: dict, tickers=None, lead: int = 0) -> pd.DataFrame:
    """P (ticker, date, t_plus, rv, h, k2, k3) и inputs {ticker: {'daily': ..., 'herf': ...}} -> длинная таблица Z (ticker, date, t_plus, ALL, lrv). Поперечные — по tickers (по умолчанию все бумаги P)."""
    parts = []
    for t, a in P.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        a = a.reset_index(drop=True)
        z = own(a, inputs[t]["daily"], inputs[t]["herf"], lead)
        parts.append(pd.concat([a[["ticker", "date", "t_plus"]], z], axis=1))
    zl = pd.concat(parts, ignore_index=True)
    return recross(zl, sorted(P["ticker"].unique()) if tickers is None else tickers)[["ticker", "date", "t_plus"] + ALL + ["lrv"]]
