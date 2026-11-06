"""Ж-211: данные программы — компоненты Ж-206/210 (замороженные), признаки F1–F6 на строках сетки, запись таблиц со статусом.
Все признаки строки d известны на закрытии d (PIT); цель — RV дня d+1 (target_date)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import pit
from models_v2.s6 import paths6
from models_v2.s6.e import comp, crit_e, resid
from models_v2.s7 import hf_ledger as L

STATUS = "HISTORICAL WALK-FORWARD 2017–02.2026 (2271 дата, просмотрено; не независимый OOS); DEVELOPMENT; Ж-211; первая цифра — потолок"
PERIODS = (("P1_2017_2021", 2017, 2021), ("P2_2022", 2022, 2022), ("P3_2023_2025", 2023, 2025), ("P4_2026", 2026, 2026),
           ("early_2017_2018", 2017, 2018), ("late_2019_2026", 2019, 2026), ("all", 2017, 2026))


def write(df: pd.DataFrame, name: str, note: str) -> None:
    L.REP.mkdir(parents=True, exist_ok=True)
    with open(L.REP / name, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")


def world() -> dict:
    """Компоненты + строки сетки F + признаки F1–F6 на d (по сетке)."""
    W = resid.build_world()
    df, g, F = comp.load(), W["grid"], W["F"]
    if not (df["d"].to_numpy("datetime64[ns]") == F.dates[g].astype("datetime64[ns]")).all():
        raise SystemExit("стоп: сетка comp != сетка мира")
    feats = pd.read_parquet(resid.B2DIR / "feats_b2.parquet")
    return {"W": W, "df": df, "g": g, "F": F, "feats": feats, "fg": feats.iloc[g].reset_index(drop=True)}


def ter_w(F, g) -> np.ndarray:
    """Терциль RV_w по 252 предыдущим (PIT); -1 — нет истории."""
    return np.nan_to_num(pit.past_bucket(np.exp(F.X[:, 1]), [1 / 3, 2 / 3]), nan=-1)[g].astype(int)


def qs(df: pd.DataFrame, s2) -> np.ndarray:
    return crit_e.qlike_series(df["rv"].to_numpy(float), np.asarray(s2, float))


def year(df: pd.DataFrame) -> np.ndarray:
    return pd.Series(df["target_date"]).dt.year.to_numpy()


def pmask(df: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    y = year(df)
    return (y >= lo) & (y <= hi)


def feature_groups() -> dict:
    import json
    return {k: list(v) for k, v in json.loads((resid.B2DIR / "accepted.json").read_text()).items()}


def group_matrix(fg: pd.DataFrame, cols: list) -> np.ndarray:
    return np.nan_to_num(fg[cols].to_numpy(float))


def paths() -> dict:
    return {"out": L.OUT, "rep": L.REP, "root": paths6.ROOT}
