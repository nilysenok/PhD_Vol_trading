"""DE0: данные — СОХРАНЁННЫЕ прогнозы (загрузчик V7 `v7_4_evrun.load`, файлы по sha256 CHECKPOINT V7), таблица дней SBER, склад M10 V7 (`store_raw.npz`), сохранённые z энкодера TCN-w5 (`stageA`), признаки S и Σ V6.
Ничего не обучает и не пишет. Даты — общие даты V7 (2018-01-03 … 2026-02-26)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO

OUT7 = IO.ROOT / "data/regen/harj_hybrid_v7"
OUT6 = IO.ROOT / "data/regen/harj_hybrid_v6"
SEEDS = (0, 1, 2, 3, 4)
NAMES = {"K1": "K1", "K2": "K2", "K3": "K3", "A2": "V6:A2", "NTP2": "V4:NTP2", "A0g": "V6:A0g", "cfg8": "cfg8", "cfg4": "cfg4"}


def load() -> dict:
    """-> dict(idx, rv, tp, f (имя → прогноз на idx), tab (таблица дней SBER 2016-12-30…2026-02-26: H, K2, K3, rv_next, t_plus, rv_d), all_series)."""
    from models_v2.s14 import v7_1m_data as DATA, v7_4_evrun as ER      # загрузчики, не обучатели
    inp, tab = ER.load(), DATA.load_tab()
    f = {k: np.asarray(inp["series"][v], float) for k, v in NAMES.items()}
    if not np.array_equal(inp["rv"], tab.loc[inp["idx"], "rv_next"].to_numpy(float)):
        raise SystemExit("rv загрузчика ≠ таблице дней")
    return dict(idx=pd.DatetimeIndex(inp["idx"]), rv=np.asarray(inp["rv"], float), tp=pd.DatetimeIndex(inp["tp"]), f=f, tab=tab, all_series=inp["series"])


def load_store() -> dict:
    b = np.load(OUT7 / "store_raw.npz", allow_pickle=True)
    ch = [str(c) for c in b["channels"]]
    return dict(days=pd.DatetimeIndex(b["days"]), r=np.where(b["present"], b["raw"][:, :, ch.index("r")], np.nan).astype(float))


def z_file(year: int, seed: int, kind: str = "tcn"):
    """(дни энкодера, z (n, 4)) сохранённого энкодера Stage A года `year` (окно 5 дней); дни — до конца года `year` включительно."""
    a = np.load(OUT7 / "stageA" / ("%s_w5_%d_s%d.npz" % (kind, year, seed)))
    return pd.DatetimeIndex(a["days"].astype("datetime64[ns]")), np.asarray(a["z"], float)


def sigma_s_features(idx: pd.DatetimeIndex) -> dict:
    """S (s1–s9) и Σ (22 колонки) V6 на датах idx; пропуск — ошибка."""
    out = {}
    for name, key in (("S_features", "S"), ("SIGMA_features", "SIGMA")):
        t = pd.read_parquet(OUT6 / (name + ".parquet")).set_index("d").sort_index()
        x = t.reindex(idx)
        if x.isna().any().any():
            raise SystemExit("%s: пропуски на датах idx" % name)
        out[key] = x.to_numpy(float)
    return out
