"""σ̂ зоны Z2 «тест-1» (2022–2024) для TEST_C/D: склейка dev ≤ 2021 (прогрев) + части Z2 — одно семейство R2 `final_d0_s2emb`.

Вход теста главы 4 (26.10): ch4_v2_sborka.yaml vhody.sigma_test = R2/final/final_d0_s2emb_h{1,5}_test.parquet, столбец rv_pred.
Dev того же семейства — data/regen/ch4_dev/sigma_hyb_h{1,5}_dev.parquet (scripts/analysis/ch4_dev_sigma.py: из тех же файлов R2,
`h2_d0_ch4.sigma_ch3`: σ = √rv_pred, float32, сортировка (date, ticker); столбец sigma_pred_h{h}). Z2 — тот же файл, срезанный
по датам (put_z2, scripts/dannye/suzhenie_rezka.py): <каталог>/final_d0_s2emb_h{1,5}__t1.parquet; то же преобразование
(`_sigma_ch3` ниже), сдвигов нет: значение на дату d известно на d. Горизонта 22 в семействе нет (колонки h22 нет → NaN).
Каталоги — FASTBT_DEV_SIGMA (ch4_dev) и FASTBT_Z2_R2FINAL (Z2 R2/final). Отказ: дата ≥ 2025 (zona.HARD), дубль date×ticker,
дата Z2 раньше 2022-01-01, отрицательный rv_pred, несколько значений `design` в файле или `design` ≠ DESIGN[h] (семейство, по горизонту).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import zona

HORIZONS = (1, 5)
COLS = {h: f"sigma_pred_h{h}" for h in HORIZONS}
# Ожидаемый столбец design по горизонту. В спецификациях (ch4_v2_sborka.yaml, r2_A*_sborka.yaml) полных строк нет (только имена
# точек D22_h1/D12_h5): строки прочитаны дирижёром 29.10 из реальных частей Z2 после нарезки.
DESIGN = {1: "D22_h1 top5|V0_XGB|B-sm|L0 d0 | s² t−1, эмбарго",
          5: "D12_h5 wrap2|V0_XGB|B-sm|L0 d0 | s² t−1, эмбарго"}
ROOT = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))     # как fastbt.data.MAIN


def dev_path(h: int, katalog: Path | None = None) -> Path:
    k = katalog or os.environ.get("FASTBT_DEV_SIGMA") or ROOT / "data/regen/ch4_dev"
    return Path(k) / f"sigma_hyb_h{h}_dev.parquet"


def z2_path(h: int, katalog: Path | None = None) -> Path:
    k = katalog or os.environ.get("FASTBT_Z2_R2FINAL") or ROOT / "data/regen/test1_2022_2024/regen/R2/final"
    return Path(k) / f"final_d0_s2emb_h{h}__t1.parquet"


def digest(dev_kat: Path | None = None, z2_kat: Path | None = None) -> str:
    """Д5: sha1 всех файлов dev и Z2 (оба семейства-части входят в ключ кэша)."""
    h = hashlib.sha1()
    for p in [dev_path(x, dev_kat) for x in HORIZONS] + [z2_path(x, z2_kat) for x in HORIZONS]:
        h.update(p.name.encode() + hashlib.sha1(p.read_bytes()).digest())
    return h.hexdigest()


def _norm(d: pd.DataFrame, what: str) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    if d.empty:
        raise ValueError(f"{what}: пустой файл")
    zona.check(d["date"].max(), what)
    if d.duplicated(["date", "ticker"]).any():
        raise ValueError(f"{what}: дубли date×ticker")
    return d


def _sigma_ch3(rv_pred: pd.Series) -> np.ndarray:
    """Преобразование ch4_dev (h2_d0_ch4.sigma_ch3): √ряда, float32."""
    return np.sqrt(rv_pred.to_numpy()).astype("float32")


def chast_z2(katalog: Path | None = None) -> pd.DataFrame:
    """Часть Z2: date, ticker, sigma_pred_h1/h5 (float32). Нет файла горизонта — FileNotFoundError."""
    out = None
    for h in HORIZONS:
        p = z2_path(h, katalog)
        d = pd.read_parquet(p)
        d = _norm(d[[c for c in ("date", "ticker", "rv_pred", "design") if c in d.columns]], p.name)
        if d["date"].min() < zona.END_DEV:
            raise ValueError(f"{p.name}: даты раньше {zona.END_DEV.date()} (пересечение с dev)")
        if (d["rv_pred"] < 0).any():
            raise ValueError(f"{p.name}: отрицательный rv_pred")
        if "design" in d.columns:
            u = d["design"].dropna().unique()
            if len(u) != 1:
                raise ValueError(f"{p.name}: в файле {len(u)} значений design — не одно семейство")
            if str(u[0]) != DESIGN[h]:
                raise ValueError(f"{p.name}: design {u[0]!r} ≠ ожидаемого для h{h} {DESIGN[h]!r} — не то семейство")
        d[COLS[h]] = _sigma_ch3(d["rv_pred"])
        d = d[["date", "ticker", COLS[h]]]
        out = d if out is None else out.merge(d, on=["date", "ticker"], how="outer")
    return out


def chast_dev(katalog: Path | None = None) -> pd.DataFrame:
    """Dev ch4_dev: date, ticker, sigma_pred_h1/h5, обрезано < 2022-01-01 (формат файлов — столбец sigma_pred_h{h})."""
    out = None
    for h in HORIZONS:
        p = dev_path(h, katalog)
        d = pd.read_parquet(p, columns=["date", "ticker", COLS[h]])
        d = _norm(d, p.name)
        d = d[d["date"] < zona.END_DEV]
        out = d if out is None else out.merge(d, on=["date", "ticker"], how="outer")
    return out


def sklei(dev: pd.DataFrame, katalog: Path | None = None) -> pd.DataFrame:
    """dev (формат ch4_dev: sigma_pred_h*; обрезается < 2022-01-01) + Z2 → date, ticker, sigma_pred_h*; без дат ≥ 2025 и дублей."""
    dev = dev.assign(date=pd.to_datetime(dev["date"]))
    dev = _norm(dev[dev["date"] < zona.END_DEV], "σ̂ dev")
    dev = dev[["date", "ticker", *COLS.values()]]
    out = pd.concat([dev, chast_z2(katalog)], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return _norm(out, "σ̂ dev+Z2")
