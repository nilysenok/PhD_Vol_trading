"""Правило качества и RV по пересобранному 10-мин ряду 2014–2015 (Ж-116 п. 1, п. 3). Загрузчики проекта не правятся.

- состав вселенной: fastbt.vselennaya_kachestvo.Kachestvo на счётчиках баров по дням: до 2016 — из
  data/regen/candles_10m_2014_2015 (пересборка), с 2016 — как в проекте (raw/candles_10m основной копии, чтение);
  таблицы пишутся рядом со старыми: docs/tables/inv/vselennaya_{po_mesyacam,vne}_peresobr.csv;
- RV 2014–2015: fastbt.sigma_real.rv_1850 на пересобранных барах против load_rv (raw) и против final_v6;
  сверка — docs/tables/inv/rv_peresobr_2014_2015_sverka.csv.
Запуск: python3 scripts/data/peresborka_10m_kachestvo_rv.py (после peresborka_10m_2014_2015.py).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import sigma_real as sr
from fastbt import vselennaya_kachestvo as vk

_spec = importlib.util.spec_from_file_location("peresb", Path(__file__).with_name("peresborka_10m_2014_2015.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)
OUT = Path("docs/tables/inv")
BOUND = pd.Timestamp("2016-01-01")


def counts_mixed(tiker: str) -> pd.Series:
    """Бары окна по дням: до 2016 — пересборка, с 2016 — raw основной копии (как в проекте)."""
    new = pd.read_parquet(pb.OUT / f"{tiker}.parquet")
    c_new = new.groupby(new["timestamp"].dt.normalize()).size()
    c_old = vk.bars_per_day(tiker, pb.MAIN)
    return pd.concat([c_new[c_new.index < BOUND], c_old[c_old.index >= BOUND]]).sort_index()


def rv_sverka() -> pd.DataFrame:
    raw = sr.load_rv(pb.TICKERS, pb.MAIN)
    rows = []
    for t in pb.TICKERS:
        new = sr.rv_1850(pd.read_parquet(pb.OUT / f"{t}.parquet"))
        r = raw[(raw["ticker"] == t) & (raw["date"] < BOUND)].set_index("date")["rv"]
        f = sr.rv_1850(pb.read_source("final_v6", t, pb.MAIN))
        lo = pd.Timestamp("2014-08-26")
        new, r, f = (x[x.index >= lo] for x in (new, r, f))
        j = pd.concat({"new": new, "raw": r, "f6": f}, axis=1)
        rows.append({"ticker": t, "days_new": int(new.size), "days_raw": int(r.size), "days_final_v6": int(f.size),
                     "max_abs_new_minus_raw": float((j["new"] - j["raw"]).abs().max()),
                     "days_rv_differs_final_v6": int(((j["new"] - j["f6"]).abs() > 1e-12).sum()),
                     "median_rv_new_over_final_v6": float(np.nanmedian(j["new"] / j["f6"]))})
    return pd.DataFrame(rows)


def main() -> None:
    k = vk.Kachestvo(pd.DataFrame({t: counts_mixed(t) for t in pb.TICKERS}).fillna(0).astype(int))
    df = k.sostav_po_mesyacam()
    df = df[(df["mesyac"] >= "2014-08") & (df["mesyac"] <= "2019-12")]
    df.to_csv(OUT / "vselennaya_po_mesyacam_peresobr.csv", index=False, float_format="%.4f")
    vne = k.intervaly_vne(df)
    vne.to_csv(OUT / "vselennaya_vne_peresobr.csv", index=False)
    old = pd.read_csv(OUT / "vselennaya_po_mesyacam.csv")
    m = old.merge(df, on=["mesyac", "tiker"], suffixes=("_old", "_new"))
    diff = m[m["vo_vselennoj_old"] != m["vo_vselennoj_new"]]
    print("месяцев-тикеров:", len(m), "состав отличается от старой таблицы:", len(diff))
    print(diff.to_string(index=False))
    s = rv_sverka()
    s.to_csv(OUT / "rv_peresobr_2014_2015_sverka.csv", index=False, float_format="%.6g")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
