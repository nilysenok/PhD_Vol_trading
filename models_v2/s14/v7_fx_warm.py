"""V7-FX: прогрев H для конфигураций 1–3 (им нужны только H и z; K2/K3 не нужны): пары с датой d от 2015-07-07 до дня перед первой датой таблицы SBER (2016-12-30), 377 строк `v6_1_h` (`H_table.parquet`, src = 'warm').
K2, K3 — NaN (корректор `Lin5` их не читает; признак ℓ не используется). Читает файл при вызове; ничего не пишет. 4–7 учатся только на парах с K2/K3 (с 2016-12-30)."""
from __future__ import annotations

import pandas as pd

from models_v2.s14 import v7_0_io as IO

PATH = IO.ROOT / "data/regen/harj_hybrid_v6/H_table.parquet"
FIRST, N_WARM = "2015-07-07", 377


def load_warm(path=None) -> pd.DataFrame:
    """-> таблица с индексом d и колонками H, K2, K3 (NaN), rv_next, t_plus; t⁺ возрастает; ровно 377 строк с 2015-07-07."""
    h = pd.read_parquet(path or PATH)
    h = h[h["src"] == "warm"].sort_values("d")
    out = pd.DataFrame({"H": h["h"].to_numpy(float), "K2": float("nan"), "K3": float("nan"), "rv_next": h["rv_next"].to_numpy(float), "t_plus": pd.to_datetime(h["t_plus"]).to_numpy("datetime64[ns]")},
                       index=pd.DatetimeIndex(pd.to_datetime(h["d"]), name=None))
    if not (len(out) == N_WARM and out.index[0] == pd.Timestamp(FIRST) and out["t_plus"].is_monotonic_increasing and (out[["H", "rv_next"]] > 0).all().all()):
        raise ValueError("прогрев H: не 377 строк с 2015-07-07, t⁺ не возрастает или H/RV не положительны")
    return out
