"""P0, G2: дневные M10-величины бумаги — ТЕМ ЖЕ кодом, что у SBER в Ж-214 (s9.hv2_m10.build, s9.hv2_sync.build, daily.build_daily); без правки s9/.
Единицы %² (×1e4). Все величины дня d — из баров основной сессии d (begin <= 18:40). β, идиосинкратическая часть — по синхронным барам с IMOEX."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s9 import hv2_m10 as M, hv2_sync as Y

TOL_ID = 1e-12
WINDOW_COLS = ["seg", "rv_w", "rv_m"]


def read_bars(t: str) -> pd.DataFrame:
    return pd.read_parquet(P.bars_path(t))


def core(bars: pd.DataFrame, index_days="auto") -> pd.DataFrame:
    """Канонический дневной ряд (daily.build_daily): n_bars, rv, bv, j, seg, rv_w, rv_m."""
    return D.build_daily(bars, index_days)[0]


def ticker_daily(bars: pd.DataFrame, imoex: pd.DataFrame) -> pd.DataFrame:
    """Бары бумаги и IMOEX -> таблица дня на канонических датах бумаги: M10 + синхронные с IMOEX + окна + β, идио."""
    t = M.build(bars)
    t = t.join(Y.build(bars, imoex, t.index)).join(core(bars)[WINDOW_COLS])
    sys_ = t["cov"] ** 2 / t["rv_m_sync"]
    t["sys_part"] = sys_
    t["idio"] = t["rv_s_sync"] - sys_
    return t


def index_daily(imoex: pd.DataFrame) -> pd.DataFrame:
    """IMOEX: те же величины без синхронных (собственные дни индекса)."""
    t = M.build(imoex)
    return t.join(core(imoex, None)[WINDOW_COLS])


def _rel(a: pd.Series, b: pd.Series) -> float:
    ok = b > 0
    return float((np.abs(a[ok] - b[ok]) / b[ok]).max()) if ok.any() else float("nan")


def identities(t: pd.DataFrame) -> dict:
    """Максимальные относительные отклонения тождеств; cov_split — по строкам с положительной синхронной RV (число исключённых — в n_cov_skipped)."""
    out = {k: v for k, v in M.identities(t).items() if k != "bv_plus_j_wrong"}
    if "cov" in t:
        z = t.dropna(subset=["cov"])
        den = np.sqrt(z["rv_s_sync"] * z["rv_m_sync"])
        num = (z["cov_pp"] + z["cov_nn"] + z["cov_mix"] - z["cov"]).abs()
        out["cov_split"] = float((num[den > 0] / den[den > 0]).max()) if (den > 0).any() else float("nan")
        out["n_cov_skipped"] = int(len(t) - int((den > 0).sum()))
    out["all_ok"] = bool(all(v <= TOL_ID for k, v in out.items() if k not in ("n_cov_skipped",)))
    return out


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, cols: list) -> dict:
    """Побитное сравнение: индексы равны; |a−b| max по колонкам (NaN должны совпадать по месту)."""
    same_index = bool(a.index.equals(b.index))
    res = {}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype == bool or y.dtype == bool or x.dtype.kind in "iu":
            res[c] = 0.0 if np.array_equal(x, y) else float("inf")
            continue
        x, y = x.astype(float), y.astype(float)
        res[c] = float("inf") if (np.isnan(x) != np.isnan(y)).any() else float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
    return {"same_index": same_index, "max": max(res.values()), "by_col": res}
