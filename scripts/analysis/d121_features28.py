"""Д12.1: features_pit_v3 для 28 тикеров той же функцией сборки m0_pit_lib.assemble(fill='asof') (ветка features-v3, cf834c7).

Строки: 17 основных — ключи склейки data/prepared (как в v3), 11 новых — дни, которые даёт тот же калькулятор
intraday_features.py (так строились stock_features -> features_all -> prepared; для 17 он воспроизводит ключи prepared
побайтно) на хранимых свечах пайплайна moex_discovery/data/clean/candles_10m (свечи без баров карантина; для 17 файлы
равны dataset_final/01_stocks/candles_10m). Служебные колонки уровня даты (low_bars*, source*) — с той же даты у 17.
Собственные rv, bv, jump, rsv± — из data/iss_check/rv_1850_noquar_28.parquet (ISS, [09:50; 18:50), без карантина),
остальные собственные — тот же калькулятор на свечах (окно выбирает own_features). Внешние ряды — общие, те же лаги, as-of.
Добавлены rq (день t), rq_d (RQ(t−1)) — новые колонки, в feature_cols v3 их нет. Цели — add_targets.
Выход: data/regen/features_pit_v3_28.parquet (+ .config.json), сверка 17 с features_pit_v3 побайтно.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import m0_pit_lib as L
from m0_common import MV, calendar, load_prepared

sys.path.insert(0, str(MV))
from src.data.intraday_features import IntradayFeatureCalculator  # noqa: E402

MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
CLEAN = MAIN / "moex_discovery/data/clean/candles_10m"
EXTRA11 = ["TGKA", "MSNG", "TRMK", "UTAR", "RGSS", "MSRS", "GCHE", "SELG", "MRKC", "PIKK", "MRKP"]


def skeleton(prep, tickers):
    """Строки новых тикеров в формате prepared: date, ticker, служебные колонки уровня даты, остальное NaN."""
    calc = IntradayFeatureCalculator(str(CLEAN), use_polars=True, annualize=True)
    d = calc.calculate_all_tickers(tickers, verbose=False).to_pandas()[["date", "ticker"]]
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    dl = [c for c in prep.columns if c.startswith("low_bars") or c.startswith("source")]
    bydate = prep.drop_duplicates("date").set_index("date")[dl]
    sk = pd.DataFrame({c: pd.Series(np.nan, index=d.index, dtype=prep[c].dtype if prep[c].dtype.kind == "f" else "float64")
                       for c in prep.columns if c not in ("date", "ticker") + tuple(dl)})
    sk["date"], sk["ticker"] = d["date"], d["ticker"]
    for c in dl:
        sk[c] = d["date"].map(bydate[c]).astype(prep[c].dtype)
    return sk[prep.columns]


def build(rv28_path, cal=None, prep=None, log=print):
    cal = calendar() if cal is None else cal
    prep = load_prepared() if prep is None else prep
    L.CANDLES = CLEAN  # для 17 файлы равны dataset_final (проверено), для 11 — те же хранимые свечи пайплайна
    p28 = pd.concat([prep, skeleton(prep, EXTRA11)], ignore_index=True)
    r18 = pd.read_parquet(rv28_path)
    r18["date"] = pd.to_datetime(r18["date"])
    out, v2, NEWF, DAYS, lost, pre_nan = L.assemble(cal, p28, r18, "asof", log=log)
    # RQ: день t и лаг 1 по строкам тикера (как rv_d); новые колонки
    rq = p28[["date", "ticker"]].merge(r18[["date", "ticker", "rq"]], on=["date", "ticker"], how="left")["rq"].values
    out["rq"] = rq
    o = out.sort_values(["ticker", "date"])
    out.loc[o.index, "rq"] = o.groupby("ticker")["rq"].ffill().values  # протягивание только прошлым, как OWN в v3
    o = out.sort_values(["ticker", "date"])
    out.loc[o.index, "rq_d"] = o.groupby("ticker")["rq"].shift(1).values
    out["is_main17"] = ~out["ticker"].isin(EXTRA11)
    # n_bars = NaN, когда в окне калькулятора у дня нет баров (UTAR: 2 дня с барами только вне окна) -> 0 баров
    nb = out["n_bars"].isna()
    if nb.any():
        log(f"n_bars: {int(nb.sum())} строк без баров в окне -> 0: {out.loc[nb, ['date', 'ticker']].values.tolist()}")
        out.loc[nb, "n_bars"] = 0
    # целые колонки prepared (n_bars uint32, volume int64) стали float из-за NaN каркаса новых тикеров — вернуть тип
    for c in prep.columns:
        if prep[c].dtype.kind in "iu" and out[c].dtype != prep[c].dtype:
            if out[c].isna().any():
                log(f"{c}: есть NaN ({int(out[c].isna().sum())}), тип float оставлен")
            else:
                out[c] = out[c].astype(prep[c].dtype)
    return out, NEWF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rv28", default=str(MAIN / "data/iss_check/rv_1850_noquar_28.parquet"))
    ap.add_argument("--out", default=str(MAIN / "data/regen/features_pit_v3_28.parquet"))
    ap.add_argument("--v3", default=str(MAIN / "data/regen/features_pit_v3.parquet"))
    a = ap.parse_args()
    out, NEWF = build(a.rv28)
    out.to_parquet(a.out, index=False)
    cfg = json.load(open(a.v3.replace(".parquet", ".config.json")))
    cfg["new_cols_28"] = ["rq", "rq_d", "is_main17"]
    json.dump(cfg, open(a.out.replace(".parquet", ".config.json"), "w"), ensure_ascii=False, indent=1)
    print("записан", a.out, out.shape, flush=True)
    # сверка 17 основных с features_pit_v3 побайтно
    v3 = pd.read_parquet(a.v3)
    m = out[out.is_main17].reset_index(drop=True)
    cols = list(v3.columns)
    miss = [c for c in cols if c not in m.columns]
    bad = [c for c in cols if c not in miss and not (m[c].dtype == v3[c].dtype and
                                                     (m[c].values.tobytes() == v3[c].values.tobytes()
                                                      if m[c].dtype != object else m[c].equals(v3[c])))]
    print(f"сверка 17 с v3: строк {len(m)} vs {len(v3)}, колонок v3 {len(cols)}, нет {miss}, "
          f"не побайтно {len(bad)} {bad[:10]}; новые колонки: {[c for c in out.columns if c not in cols]}")
    print("PASS" if len(m) == len(v3) and not miss and not bad else "FAIL")


if __name__ == "__main__":
    main()
