"""P1a (Ж-217 R10, У-3): контроли K2 (XGB `x2_best_j`) и K3 (LGBM `C0_inh_F4`) на 28 бумагах. Это контроли, не trials; новой оптимизации нет.
Запуск: python3 -W ignore -m models_v2.s10.hp_k23 [--tickers T ...] [--check]. Без --check: после сверки SBER (побитно с components.XJ/LF) считает недостающие бумаги из списка (по умолчанию — основные 17; E11 — отказ,
готовые `k23_<T>` не пересчитываются); с --check — проверяет 17 основных + GCHE (У-24, Ж-220 п. 6–7; `k23.parquet` из 28 не нужен). Процессов счёта <= 2 (пул spawn)."""
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


K23_17 = sorted(set(P.MAIN17) | {"GCHE"})      # У-24 (Ж-220 п. 6–7): 17 основных + GCHE (готова); E11 — перенос, K2/K3 не восстанавливаются


def todo(tickers: list) -> list:
    """Бумаги к счёту: только основные 17 (E11 — ValueError) и без готового `k23_<T>` (готовые не пересчитываются и не перезаписываются)."""
    bad = [t for t in tickers if t not in P.MAIN17]
    if bad:
        raise ValueError("K2/K3 считаются только для основных 17 (Ж-220 п. 6–7): %s" % bad)
    return [t for t in tickers if not _tpath(t).exists()]


def compute_missing(tickers: list, panel: pd.DataFrame, imoex: pd.DataFrame, pool: list, run=None) -> list:
    """Считает недостающие из `tickers` замороженной процедурой (И2 по правилу); возвращает список посчитанных."""
    run, todo_ = run or run_ticker, todo(tickers)
    for t in todo_:
        run(t, panel, imoex, pool, None)
    return todo_


def compute_all(tickers: list | None = None) -> int:
    if P.sha16(P.OUT / "panel_base.parquet") != PANEL_SHA:
        print("panel_base.parquet: SHA не совпал с P0")
        return 1
    d = KO.sber_diff(pd.read_parquet(_tpath("SBER")), pd.read_parquet(P.COMP))
    print("SBER vs components:", d, flush=True)
    if d["n"] != d["n_comp"] or d["max_abs_diff_xj"] != 0.0 or d["max_abs_diff_lf"] != 0.0:
        print("SBER не побитно — остальные бумаги не считаются (разбор до причины)")
        return 1
    panel, imoex = pd.read_parquet(P.OUT / "panel_base.parquet", columns=PANEL_COLS), read_bars("IMOEX")
    compute_missing(tickers or list(P.MAIN17), panel, imoex, KW.pool_papers(read_bars))
    return verify()


def check_k23(tickers: list, panel: pd.DataFrame, comp: pd.DataFrame, load) -> list:
    """Ошибки (пусто — ок) для `tickers`: `load(t)` -> k23_<T> или None. Нет файла основной бумаги — ошибка; сетка (t, t⁺) и RV цели — побитно панель; σ̂² K2/K3 конечны и > 0 (оба или ни одного);
    SBER — побитно components.XJ/LF. `k23.parquet` из 28 не требуется (У-24)."""
    bad = []
    for t in tickers:
        df, pt = load(t), panel[panel["ticker"] == t].reset_index(drop=True)
        if df is None:
            bad.append("%s: нет k23_%s.parquet" % (t, t))
            continue
        same = len(df) == len(pt) and all(np.array_equal(df[c].to_numpy(), pt[c].to_numpy()) for c in ("date", "t_plus", "rv_tplus"))
        bad += [] if same else ["%s: сетка (t, t⁺) или rv_tplus не равны panel_base побитно" % t]
        f = df[np.isfinite(df["sigma2_k2"]) | np.isfinite(df["sigma2_k3"])]
        ok = np.isfinite(f["sigma2_k2"]) & np.isfinite(f["sigma2_k3"]) & (f["sigma2_k2"] > 0) & (f["sigma2_k3"] > 0)
        bad += [] if len(f) and ok.all() else ["%s: σ̂² K2/K3 не конечны, ≤ 0, определён один из двух или прогнозов нет" % t]
        if t == "SBER":
            d = KO.sber_diff(df, comp)
            bad += [] if d["n"] == d["n_comp"] and d["max_abs_diff_xj"] == d["max_abs_diff_lf"] == d["max_abs_diff_rv"] == 0.0 else ["SBER не побитно: %s" % d]
    return bad


def verify() -> int:
    bad = [] if P.sha16(P.OUT / "panel_base.parquet") == PANEL_SHA else ["panel_base.parquet: SHA не совпал с P0"]
    bad += check_k23(K23_17, pd.read_parquet(P.OUT / "panel_base.parquet", columns=PANEL_COLS), pd.read_parquet(P.COMP), lambda t: pd.read_parquet(_tpath(t)) if _tpath(t).exists() else None)
    bad += ["%s: нет k23_meta_%s.json" % (t, t) for t in K23_17 if not _mpath(t).exists()]
    meta = {t: json.loads(_mpath(t).read_text()) for t in K23_17 if _mpath(t).exists()}
    print("k23 на 17 + GCHE: бумаг %d из %d; первый год У-3: %s; время счёта, с: %s" % (len(meta), len(K23_17), {t: m["y0_used"] for t, m in meta.items()},
                                                                                {t: round(m["secs_world"] + m["secs_i2"] + m["secs_forecast"]) for t, m in meta.items() if "secs_world" in m}))
    print("\n".join(bad))
    print("ИТОГ:", "OK" if not bad else "FAIL")
    return 1 if bad else 0


def run(check: bool, tickers: list | None = None) -> int:
    return verify() if check else compute_all(tickers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tickers", nargs="+", help="бумаги к счёту (только основные 17; готовые k23_<T> не пересчитываются)")
    a = ap.parse_args()
    sys.exit(run(a.check, a.tickers))
