"""V5, лист 1b, G17: статистика отношения Барндорфф-Нильсена — Шепарда (Хуанг–Таучен) по M10-доходностям дня.
RV = Σ r², BV = (π/2)·Σ_{i≥2} |r_i||r_{i-1}| (определение проекта, без поправки n/(n−1)), TQ = n·μ_{4/3}^{-3}·Σ_{i≥3} (|r_i||r_{i-1}||r_{i-2}|)^{4/3}, μ_{4/3} = 2^{2/3}Γ(7/6)/Γ(1/2) (те же M10-бары);
z = ((RV − BV)/RV) / sqrt((π²/4 + π − 5)/n · max(1, TQ/BV²)); значимый скачок: z > Φ⁻¹(0,999) = 3,09 (уровень 0,1 %). n — число M10-доходностей дня.
Кэш `data/regen/harj_hybrid_v5/dx1_bns_<ключ>.parquet`; ключ = хэш входов и кода этого файла (Д5); `--check` кэш не создаёт."""
from __future__ import annotations

import hashlib
from math import gamma, pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from models_v2.s10 import hp_paths as P
from models_v2.s12 import v5_0_io as IO
from models_v2.s9 import hv2_m10 as M

Z_CRIT = float(norm.ppf(0.999))
C_BNS = pi ** 2 / 4 + pi - 5
MU43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)
SCALE = 1e4


def day_stat(r) -> tuple:
    """Доходности дня -> (n, rv, bv, tq, z) в единицах r (без масштаба 1e4); n < 4 или rv ≤ 0 или bv ≤ 0 -> z = NaN."""
    r = np.asarray(r, float)
    n = len(r)
    a = np.abs(r)
    rv = float((r * r).sum())
    bv = float(pi / 2 * (a[1:] * a[:-1]).sum()) if n >= 2 else np.nan
    tq = float(n * MU43 ** -3 * ((a[2:] * a[1:-1] * a[:-2]) ** (4 / 3)).sum()) if n >= 3 else np.nan
    if n < 4 or not (rv > 0 and bv > 0 and tq >= 0):
        return n, rv, bv, tq, np.nan
    z = (rv - bv) / rv / np.sqrt(C_BNS / n * max(1.0, tq / bv ** 2))
    return n, rv, bv, tq, float(z)


def ticker_days(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары бумаги -> по дням: n, rv, bv, tq, z (rv/bv/tq в %², как в проекте)."""
    r = M.bar_returns(bars)
    rows = {d: day_stat(x.to_numpy(float)) for d, x in r.groupby("date")["r"]}
    t = pd.DataFrame.from_dict(rows, orient="index", columns=["n", "rv", "bv", "tq", "z"]).rename_axis("date")
    t[["rv", "bv"]] *= SCALE
    t["tq"] *= SCALE ** 2
    return t


def all_days() -> pd.DataFrame:
    fr = []
    for t in IO.T17:
        x = ticker_days(pd.read_parquet(P.bars_path(t)))
        x.insert(0, "ticker", t)
        fr.append(x.reset_index())
    return pd.concat(fr, ignore_index=True)


def cache_key() -> str:
    return IO.inputs_hash() + "_" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def cached_days(write: bool = False) -> pd.DataFrame:
    p = IO.OUT / ("dx1_bns_%s.parquet" % cache_key())
    if p.exists():
        return pd.read_parquet(p)
    d = all_days()
    if write:
        IO.OUT.mkdir(parents=True, exist_ok=True)
        d.to_parquet(p)
    return d
