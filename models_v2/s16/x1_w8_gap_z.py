"""W8-B: признак z_d = ln RV^IMOEX дня d и метки терцилей для `x1_w8_gap` (`W8_SPEC.md` разд. 2). Только чтение. Те же функции доступны как атрибуты `x1_w8_gap`.
Публичное API:
- `load_bars()`: бары IMOEX (begin, open, close): архив `data/iss_check/index_10m/IMOEX.parquet` + продолжение `data/fwd/mkt_m10/IMOEX_m10_fwd.parquet` (бары позже конца архива).
- `day_rv(bars) -> Series (индекс — день)`: Σ r²_бар по барам с началом 10:00…18:30 (первый бар дня — ln(close/open), далее лог-разность закрытий). Бары вне 10:00–18:30 в z не входят.
- `z_series(d, bars=None) -> ndarray`: ln RV^IMOEX дня d на датах окна M (NaN, если дня нет или RV ≤ 0).
- `labels(z) -> ndarray` T1/T2/T3/na: `x1_w8_diag_core.terc_past` — пороги по z дней СТРОГО до d, не меньше 60 прошлых.
- `shift_z(z, s=252)` = `np.roll(z, s)`; `corrupt_bars(bars, D)`: подложенная порча (бары дней > D и все бары вне 10:00…18:30: open и close умножены на 7 у чётных и на 3 у нечётных баров, чтобы менялись лог-доходности и RV) — для проверки PIT."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16.x1_w8_diag_core import CUT_BAR, MINPAST, ROOT, terc_past

SHIFT = 252


def load_bars() -> pd.DataFrame:
    a = pd.read_parquet(ROOT / "data/iss_check/index_10m/IMOEX.parquet", columns=["begin", "open", "close"])
    n = pd.read_parquet(ROOT / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet", columns=["begin", "open", "close"])
    a["begin"], n["begin"] = pd.to_datetime(a["begin"]), pd.to_datetime(n["begin"])
    return pd.concat([a, n[n["begin"] > a["begin"].max()]], ignore_index=True).sort_values("begin")


def day_rv(bars: pd.DataFrame) -> pd.Series:
    b = bars.copy()
    tod = b["begin"] - b["begin"].dt.normalize()
    b = b[(tod >= pd.Timedelta(hours=10)) & (tod <= CUT_BAR)].copy()
    b["day"] = b["begin"].dt.normalize()
    b["lr"] = np.log(b["close"] / b["close"].groupby(b["day"]).shift(1))
    first = b.groupby("day").head(1).index
    b.loc[first, "lr"] = np.log(b.loc[first, "close"] / b.loc[first, "open"])
    return b.groupby("day")["lr"].apply(lambda s: float(np.nansum(s ** 2)))


def z_series(d, bars: pd.DataFrame | None = None) -> np.ndarray:
    rv = day_rv(load_bars() if bars is None else bars).reindex(pd.DatetimeIndex(d)).to_numpy(float)
    return np.log(np.where(rv > 0, rv, np.nan))


def labels(z) -> np.ndarray:
    return terc_past(z, MINPAST)


def shift_z(z, s: int = SHIFT) -> np.ndarray:
    return np.roll(np.asarray(z, float), s)


def corrupt_bars(bars: pd.DataFrame, D) -> pd.DataFrame:
    """Подложенная порча: все бары дней > D и все бары вне 10:00…18:30 (вечер любого дня) умножены на 7."""
    b = bars.copy().reset_index(drop=True)
    tod = b["begin"] - b["begin"].dt.normalize()
    m = ((b["begin"].dt.normalize() > pd.Timestamp(D)) | (tod > CUT_BAR)).to_numpy()
    f = np.where(np.arange(len(b)) % 2 == 0, 7.0, 3.0)      # разные множители у соседних баров: лог-доходности и RV дня меняются
    for c in ("open", "close"):
        b[c] = np.where(m, b[c].to_numpy(float) * f, b[c].to_numpy(float))
    return b
