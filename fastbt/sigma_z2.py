"""σ̂ зоны Z2 «тест-1» (2022–2024) для интерфейса R-P: склейка dev ≤ 2021 (прогрев) + части Z2 прогнозов главы 3.

Z2-файлы (схема `put_z2`, scripts/dannye/suzhenie_rezka.py): <каталог>/final_h{1,5,22}__t1.parquet, колонки date, ticker,
rv_pred, …; каталог — FASTBT_Z2_FINAL3 (data/regen/test1_2022_2024/regen/final3). Преобразование то же, что дало dev-ряды
(sigma_pred_h1_final3_full.py, final3.py фаза build): σ = √rv_pred, float32, сортировка (date, ticker) — оно и есть «выравнивание на d0»:
значение на дату d известно на d, сдвигов нет. Отказ: дата ≥ 2025 (zona.HARD), дубль date×ticker, дата Z2 раньше 2022-01-01
(пересечение с dev), отрицательный rv_pred.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import zona

HORIZONS = (1, 5, 22)
COLS = {h: f"sigma_pred_h{h}" for h in HORIZONS}


def _norm(d: pd.DataFrame, what: str) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    if d.empty:
        raise ValueError(f"{what}: пустой файл")
    zona.check(d["date"].max(), what)
    if d.duplicated(["date", "ticker"]).any():
        raise ValueError(f"{what}: дубли date×ticker")
    return d


def chast_z2(katalog: Path) -> pd.DataFrame:
    """Часть Z2: date, ticker, sigma_pred_h1/h5/h22 (float32). Нет файла горизонта — FileNotFoundError."""
    out = None
    for h in HORIZONS:
        p = Path(katalog) / f"final_h{h}__t1.parquet"
        d = _norm(pd.read_parquet(p, columns=["date", "ticker", "rv_pred"]), p.name)
        if d["date"].min() < zona.END_DEV:
            raise ValueError(f"{p.name}: даты раньше {zona.END_DEV.date()} (пересечение с dev)")
        if (d["rv_pred"] < 0).any():
            raise ValueError(f"{p.name}: отрицательный rv_pred")
        d[COLS[h]] = np.sqrt(d["rv_pred"].to_numpy()).astype("float32")
        d = d[["date", "ticker", COLS[h]]]
        out = d if out is None else out.merge(d, on=["date", "ticker"], how="outer")
    return out


def _dev_h(dev: pd.DataFrame, h: int) -> pd.Series:
    col = next((c for c in (f"sigma_h{h}", COLS[h]) if c in dev.columns), None)
    return dev[col].astype("float32") if col else pd.Series(np.float32("nan"), index=dev.index)


def sklei(dev: pd.DataFrame, katalog: Path) -> pd.DataFrame:
    """dev (обрезается < 2022-01-01; имена sigma_h*/sigma_pred_h*) + Z2 → date, ticker, sigma_pred_h*; без дат ≥ 2025 и дублей."""
    dev = dev.assign(date=pd.to_datetime(dev["date"]))
    dev = _norm(dev[dev["date"] < zona.END_DEV], "σ̂ dev")
    dev = pd.DataFrame({"date": dev["date"], "ticker": dev["ticker"], **{COLS[h]: _dev_h(dev, h) for h in HORIZONS}})
    z2 = chast_z2(katalog)
    out = pd.concat([dev, z2], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return _norm(out, "σ̂ dev+Z2")
