"""P1a (Ж-217 R10, У-3): контроли K2 (XGB `x2_best_j`) и K3 (LGBM `C0_inh_F4`) на 28 бумагах. Это контроли, не trials; новой оптимизации нет.
Запуск: python3 -W ignore -m models_v2.s10.hp_k23 [--check]. Без --check: считает недостающие бумаги (SBER первой, побитно с components.XJ/LF; остальные 27 — только после сверки) и пишет выход;
с --check — проверяет записанное. Процессов счёта <= 2 (пул spawn)."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_k23_calc as KC, hp_k23_out as KO, hp_k23_world as KW, hp_paths as P

PANEL_SHA = "6e2988c6a8615ac6"
PANEL_COLS = ["ticker", "date", "t_plus", "rv_tplus", "sigma2_k1", "gap", "cross_seg"]


def read_bars(t: str) -> pd.DataFrame:
    return pd.read_parquet(P.bars_path(t))


def check_vs_panel(t: str, panel_t: pd.DataFrame, fr: pd.DataFrame) -> None:
    """Сетка (t, t⁺) и RV цели прогнозных строк обязаны совпасть с панелью побитно (иначе — другая бумага или календарь)."""
    m = fr.merge(panel_t[["date", "t_plus", "rv_tplus"]], on="date", how="left", suffixes=("", "_p"))
    if m["t_plus_p"].isna().any() or not (m["t_plus"] == m["t_plus_p"]).all() or not np.array_equal(m["rv"].to_numpy(), m["rv_tplus"].to_numpy()):
        raise ValueError("%s: сетка t⁺ или RV цели не совпали с panel_base" % t)
    if len(fr) and not (np.isfinite(fr[["sigma2_k2", "sigma2_k3"]]).all().all() and (fr[["sigma2_k2", "sigma2_k3"]] > 0).all().all()):
        raise ValueError("%s: σ̂² K2/K3 не конечна или ≤ 0" % t)


def compute_ticker(t: str, panel_t: pd.DataFrame, read, imoex: pd.DataFrame, pool: list, i2: dict | None = None, procs: int = 2) -> tuple:
    """-> (строки панели бумаги + sigma2_k2, sigma2_k3, hp_year, hp_k2, hp_k3; мета). i2 None — годовой выбор И2 бумаги по правилу; иначе заданный."""
    t0 = time.time()
    w = KW.build_world(read(t), imoex, pool)
    y_rule, t1 = KW.first_year(w), time.time()
    if i2 is None:
        i2 = KW.choose_i2(w, [Y for Y in KW.YEARS if y_rule is not None and Y >= y_rule], procs)
    y0, t2 = next((Y for Y in KW.YEARS if y_rule is not None and Y >= y_rule and Y in i2), None), time.time()
    fr = KC.forecast_frame(w, i2, KC.select_rows(w, y0, panel_t["date"]), procs)
    check_vs_panel(t, panel_t, fr)
    out = panel_t[PANEL_COLS].merge(fr.drop(columns="rv"), on=["date", "t_plus"], how="left")
    meta = {"ticker": t, "y0_rule": y_rule, "y0_used": y0, "n_panel": len(panel_t), "n_forecast": len(fr), "secs_world": t1 - t0, "secs_i2": t2 - t1, "secs_forecast": time.time() - t2,
            "i2": KW.i2_dump(i2)}
    return out, meta


def _tpath(t: str):
    return P.OUT / ("k23_%s.parquet" % t)


def _mpath(t: str):
    return P.OUT / ("k23_meta_%s.json" % t)


def run_ticker(t: str, panel: pd.DataFrame, imoex: pd.DataFrame, pool: list, i2: dict | None) -> None:
    out, meta = compute_ticker(t, panel[panel["ticker"] == t].reset_index(drop=True), read_bars, imoex, pool, i2)
    _mpath(t).write_text(json.dumps(meta, default=float))
    out.to_parquet(_tpath(t))
    print("%s: y0_rule=%s y0_used=%s строк %d прогнозов %d; мир %.0f с, И2 %.0f с, прогноз %.0f с" % (t, meta["y0_rule"], meta["y0_used"], meta["n_panel"], meta["n_forecast"],
                                                                                              meta["secs_world"], meta["secs_i2"], meta["secs_forecast"]), flush=True)


def compute_all() -> int:
    from models_v2 import run_fix
    if P.sha16(P.OUT / "panel_base.parquet") != PANEL_SHA:
        print("panel_base.parquet: SHA не совпал с P0")
        return 1
    P.OUT.mkdir(parents=True, exist_ok=True)
    panel, imoex = pd.read_parquet(P.OUT / "panel_base.parquet", columns=PANEL_COLS), read_bars("IMOEX")
    pool = KW.pool_papers(read_bars)
    if not _tpath("SBER").exists():
        run_ticker("SBER", panel, imoex, pool, run_fix.load_i2("i2_choice_g132"))
    d = KO.sber_diff(pd.read_parquet(_tpath("SBER")), pd.read_parquet(P.COMP))
    print("SBER vs components:", d, flush=True)
    if d["n"] != d["n_comp"] or d["max_abs_diff_xj"] != 0.0 or d["max_abs_diff_lf"] != 0.0:
        print("SBER не побитно — остальные бумаги не считаются (разбор до причины)")
        return 1
    for t in [x for x in P.TICKERS if x != "SBER"]:
        if not _tpath(t).exists():
            run_ticker(t, panel, imoex, pool, None)
    return assemble()


def assemble() -> int:
    df = pd.concat([pd.read_parquet(_tpath(t)) for t in P.TICKERS], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    meta = pd.DataFrame([json.loads(_mpath(t).read_text()) for t in P.TICKERS])
    df.to_parquet(KO.K23)
    tab = KO.qlike_table(df)
    KO.write_csv(tab, "P1_K23.csv", "QLIKE K1/K2/K3 по бумагам и годам цели на общих датах (K1, K2, K3 определены)")
    KO.write_csv(KO.summary(tab), "P1_K23_SUMMARY.csv", "группы и доля бумаг, где K2 < K1 и K3 < K1 (на общих датах)")
    KO.write_csv(meta.drop(columns="i2"), "P1_K23_TIME.csv", "первый год У-3 (правило и фактический), число строк, время счёта по бумагам (с)")
    KO.write_csv(pd.DataFrame([{"ticker": r.ticker, "year": Y, **h} for r in meta.itertuples() for Y, h in json.loads(r.i2).items()]), "P1_K23_I2.csv", "годовой выбор И2 каждой бумаги (HP K2/K3 = И2 + замороженные сдвиги)")
    return verify()


def verify() -> int:
    df, meta = pd.read_parquet(KO.K23), pd.DataFrame([json.loads(_mpath(t).read_text()) for t in P.TICKERS])
    bad = KO.verify(df, pd.read_parquet(P.OUT / "panel_base.parquet", columns=PANEL_COLS), meta, pd.read_parquet(P.COMP))
    print("k23.parquet: строк %d, бумаг %d, строк с прогнозом K2/K3: %d; первые годы У-3 (правило/факт): %s" % (
        len(df), df["ticker"].nunique(), int(np.isfinite(df["sigma2_k2"]).sum()), {r.ticker: (r.y0_rule, r.y0_used) for r in meta.itertuples() if r.y0_rule != r.y0_used}))
    print("\n".join(bad))
    print("ИТОГ:", "OK" if not bad else "FAIL")
    return 1 if bad else 0


def run(check: bool) -> int:
    return verify() if check else compute_all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(run(ap.parse_args().check))
