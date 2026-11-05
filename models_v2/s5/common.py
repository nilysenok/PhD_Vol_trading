"""Волна A (Ж-204): пути, контроли Б-2 / HAR-J(б), срезы по периодам. Только чтение готовых прогнозов пилота."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import prov

ROOT = prov.ROOT
OUT = ROOT / "data/regen/v2_sber_h1"
TAB = ROOT / "docs/tables/v2_sber_h1"
DEEP = ROOT / "data/regen/v2_sber_h1_deep_v1"
REP = ROOT / "docs/otchety/sber_h1_deep_v1"
IMOEX = ROOT / "data/iss_check/index_10m/IMOEX.parquet"
B2, HJ = "fx_b2_g132_b", "harj_b"
N_GRID = 2271
PERIODS = ("p2017_2021", "y2022", "p2023_2025", "y2026", "all")


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_comment_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def controls() -> pd.DataFrame:
    """d, target_date, rv, s2_b2, s2_harj на общей сетке 2271 дат; несовпадение сетки или RV — ошибка."""
    c = pd.read_parquet(OUT / "fix_forecasts_c.parquet")
    p = pd.read_parquet(OUT / "forecasts_p1.parquet")
    b, h = c[c.model == B2].reset_index(drop=True), p[p.model == HJ].reset_index(drop=True)
    if len(b) != len(h) or len(b) != N_GRID:
        raise ValueError("сетка: %d и %d строк, ожидается %d" % (len(b), len(h), N_GRID))
    for col in ("d", "target_date"):
        if not (b[col].to_numpy("datetime64[ns]") == h[col].to_numpy("datetime64[ns]")).all():
            raise ValueError("сетки %s Б-2 и HAR-J(б) различаются" % col)
    if not np.array_equal(b["rv"].to_numpy(), h["rv"].to_numpy()):
        raise ValueError("RV Б-2 и HAR-J(б) различаются")
    return pd.DataFrame({"d": b["d"].astype("datetime64[ns]"), "target_date": b["target_date"].astype("datetime64[ns]"),
                         "rv": b["rv"], "s2_b2": b["sigma2"], "s2_harj": h["sigma2"]})


def period_masks(target_date: pd.Series) -> dict:
    y = pd.Series(target_date).dt.year.to_numpy()
    return {"p2017_2021": y <= 2021, "y2022": y == 2022, "p2023_2025": (y >= 2023) & (y <= 2025),
            "y2026": y == 2026, "all": np.ones(len(y), bool)}


def write_csv(df: pd.DataFrame, path, note: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# разработка; волна A Ж-204; до аудита; оценка 2017–02.2026 (2022+ — просмотр контроля гл. 3 по SBER); %s\n" % note)
        df.to_csv(f, index=False)
