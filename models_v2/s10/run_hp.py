"""Ж-215 / Ж-217, листы P0 и P1b (стадии k4567; k23 — лист P1a). Запуск: python3 -W ignore -m models_v2.s10.run_hp --stage inv|daily|har|shift|port|panel|k4567|k23 [--check]
Без --check стадия считает и пишет; с --check — проверяет уже записанный выход и печатает итог (код возврата 0 — гейт закрыт). Порядок: inv, daily, har, shift, port, panel. ML-обучений нет."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_daily as HD, hp_feat as F, hp_har as HH, hp_inv as HI, hp_paths as P, hp_panel as PN, hp_port as PO, hp_shift as SH

ALL = P.TICKERS + ["IMOEX"]
NTHREAD = 6      # P1b: потоков обучения на процесс (в сумме не более 2 процессов счёта)


def _daily(t: str) -> pd.DataFrame:
    return pd.read_parquet(P.OUT / ("daily_%s.parquet" % t))


def _har(t: str) -> pd.DataFrame:
    return pd.read_parquet(P.OUT / ("har_%s.parquet" % t))


def _report(ok: bool, lines: list) -> int:
    print("\n".join(lines))
    print("ИТОГ:", "OK" if ok else "FAIL")
    return 0 if ok else 1


def stage_inv(check: bool) -> int:
    if check:
        inv, cal = pd.read_csv(P.REP / "P0_INVENTORY.csv", comment="#"), pd.read_csv(P.REP / "P0_CALENDAR.csv", comment="#")
        bad = HI.check_inventory(inv, cal)
        return _report(not bad, ["бумаг %d, дат календаря %d" % (len(inv), len(cal)), *bad])
    bars = {t: HD.read_bars(t) for t in ALL}
    cores = {t: HD.core(bars[t], None if t == "IMOEX" else "auto") for t in ALL}
    inv, cal = HI.run(bars, cores)
    for name, df, note in (("P0_INVENTORY.csv", inv, "инвентарь 28 бумаг (канонические дни daily.build_daily); оборот — млн руб., медиана дня по году"),
                           ("P0_CALENDAR.csv", cal, "календарь дата × бумаги: 1 — у бумаги есть каноническая торговая строка"),
                           ("P0_SCHEDULE.csv", HI.schedule({t: bars[t] for t in P.TICKERS}), "расписание основной сессии по годам из фактических баров (regime = последний бар, мин.)"),
                           ("P0_HASHES.csv", HI.hashes(), "SHA-256[:16] входов")):
        P.write_csv(df, name, note)
    return stage_inv(True)


def stage_daily(check: bool) -> int:
    P.OUT.mkdir(parents=True, exist_ok=True)
    imo = HD.read_bars("IMOEX")
    if not check:
        HD.index_daily(imo).to_parquet(P.OUT / "daily_IMOEX.parquet")
        for t in P.TICKERS:
            HD.ticker_daily(HD.read_bars(t), imo).to_parquet(P.OUT / ("daily_%s.parquet" % t))
    rows = [{"ticker": t, **HD.identities(_daily(t))} for t in ALL]
    idn = pd.DataFrame(rows)
    sb, ref, comp = _daily("SBER"), pd.read_parquet(P.M10_SBER), pd.read_parquet(P.COMP)
    cols = [c for c in ref.columns if c in sb.columns and c != "grid_d"]
    cmp_ = HD.max_abs_diff(sb, ref, cols)
    rv_c = float(np.abs(sb["rv"].reindex(pd.DatetimeIndex(comp["target_date"])).to_numpy() - comp["rv"].to_numpy()).max())
    if not check:
        P.write_csv(idn, "P0_IDENTITIES.csv", "тождества по бумагам (макс. отн. отклонение, допуск %g)" % HD.TOL_ID)
        P.write_csv(pd.DataFrame([{"max_abs_diff_vs_Zh214_m10": cmp_["max"], "same_index": cmp_["same_index"], "n_cols": len(cols), "max_abs_diff_vs_components_rv": rv_c}]),
                    "P0_SBER_CHECK.csv", "SBER: daily_SBER против m10_daily Ж-214 и components.rv (побитно)")
    ok = bool(idn["all_ok"].all()) and cmp_["max"] == 0.0 and cmp_["same_index"] and rv_c == 0.0 and len(idn) == 29
    return _report(ok, ["файлов %d; тождества all_ok: %s (max %.2e)" % (len(idn), bool(idn["all_ok"].all()), idn.drop(columns=["ticker", "all_ok", "n_cov_skipped"], errors="ignore").max().max()),
                        "SBER vs Ж-214 m10_daily: max_abs_diff = %s по %d колонкам, индекс равен: %s" % (cmp_["max"], len(cols), cmp_["same_index"]),
                        "SBER rv vs components.rv: max_abs_diff = %s" % rv_c])


def stage_har(check: bool) -> int:
    if not check:
        for t in ALL:
            HH.series(_daily(t)).to_parquet(P.OUT / ("har_%s.parquet" % t))
    k1 = {t: _har(t) for t in ALL}
    if not check:
        P.write_csv(HH.qlike_table(k1), "P0_K1.csv", "QLIKE K1 (HAR-J, R3) по бумагам и годам цели; all_no_gap — без пар gap")
    comp = pd.read_parquet(P.COMP)
    m = k1["SBER"].merge(comp[["d", "target_date", "rv", "HAR"]], on=["d", "target_date"], suffixes=("", "_c"))
    diff = float(np.abs(m["sigma2"] - m["HAR"]).max())
    d0 = pd.Timestamp("2021-06-30")
    cut = HH.from_bars(HH.corrupt_after(HD.read_bars("SBER"), d0))
    a = k1["SBER"].set_index("d").loc[:d0, "sigma2"]
    b = cut.set_index("d").loc[:d0, "sigma2"]
    trunc = float(np.abs(a - b).max()) if a.index.equals(b.index) else float("inf")
    ok = len(m) == len(comp) and diff == 0.0 and trunc == 0.0 and len(k1) == 29
    return _report(ok, ["har-файлов %d; SBER vs components.HAR: строк %d из %d, max_abs_diff = %s" % (len(k1), len(m), len(comp), diff),
                        "обрезка будущего (SBER, d0 = %s, все бары после d0 повреждены; боевой путь from_bars): max_abs_diff = %s" % (d0.date(), trunc)])


def stage_shift(check: bool) -> int:
    imo_rv = _daily("IMOEX")["rv"]
    if not check:
        rows = [r for t in ALL for r in SH.table_one(t, _daily(t), imo_rv, _har(t))]
        P.write_csv(pd.DataFrame(rows), "P0_SHIFT.csv", "Д1: corr(x_d, ln RV_{d+k}), k=−2…+1; ok — максимум |corr| на k=0; diag — разбор несовпадений")
    s = pd.read_csv(P.REP / "P0_SHIFT.csv", comment="#")
    bad = s[~s["ok"]]
    if not check:
        P.write_csv(bad, "P0_SHIFT_MISMATCH.csv", "несовпадения Д1 (максимум не на k=0) с разбором")
    real = bad[bad["diag"].str.startswith("РАЗБОР", na=False) | bad["diag"].isna()]
    return _report(len(real) == 0 and len(s) > 0, ["строк %d (бумаг %d), несовпадений %d, из них без шумового объяснения (РАЗБОР/NaN) %d" % (len(s), s["ticker"].nunique(), len(bad), len(real)),
                                                    *["%s %s: argmax %s %s" % (r.ticker, r.series, r.argmax, r.diag) for r in real.head(20).itertuples()]])


def stage_port(check: bool) -> int:
    if not check:
        tab = PO.table({t: HD.read_bars(t) for t in P.TICKERS}, {t: _daily(t) for t in P.TICKERS})
        P.write_csv(tab, "P0_PORTABILITY.csv", "переносимость K2/K3/K6: входы замороженных спецификаций по бумагам (правило — docstring hp_port)")
    pt = pd.read_csv(P.REP / "P0_PORTABILITY.csv", comment="#")
    return _report(len(pt) == 28, ["бумаг %d; K2/K3 переносимы: %d; K6: %d" % (len(pt), pt["K2_portable"].sum(), pt["K6_portable"].sum()),
                                   *["%s: %s" % (r.ticker, r.reason_not_portable) for r in pt[~pt["K2_portable"]].itertuples()]])


def stage_panel(check: bool) -> int:
    daily, k1 = {t: _daily(t) for t in P.TICKERS}, {t: _har(t) for t in P.TICKERS}
    path = P.OUT / "panel_base.parquet"
    if not check:
        PN.build(daily, _daily("IMOEX"), k1).to_parquet(path)
        PN.write_checkpoint()
    panel = pd.read_parquet(path)
    bad = PN.check(panel, daily)
    miss = [c for c in F.X_COLS + F.M_COLS + F.C_COLS if c not in panel.columns]
    bad += ["нет признаков: %s" % miss] if miss else []
    return _report(not bad, ["panel_base.parquet: строк %d, колонок %d, sha256[:16]=%s" % (len(panel), panel.shape[1], P.sha16(path)), *bad])


def stage_k23(check: bool) -> int:
    """P1a (лист другого агента): models_v2.s10.hp_k23.run(check); модуля ещё нет — «нет модуля», rc = 1."""
    try:
        from models_v2.s10 import hp_k23
    except ModuleNotFoundError as e:
        if e.name != "models_v2.s10.hp_k23":
            raise
        print("нет модуля models_v2.s10.hp_k23")
        return 1
    return hp_k23.run(check)


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def k4567_compute() -> dict:
    """Все контроли P1b; время по стадиям. K1 для K6/K7 берётся из строк панели движка (те же даты)."""
    tm, t0 = {}, time.time()
    daily, imo = {t: _daily(t) for t in P.TICKERS}, _daily("IMOEX")
    panel = CT.build_panel_ext(daily, imo)
    panel.to_parquet(CT.ENGINE_PANEL)
    tm["panel_ext"], t0 = time.time() - t0, time.time()
    k1d = {t: g[["date"]].rename(columns={"date": "d"}) for t, g in panel.groupby("ticker")}
    k6 = {t: CT.k6_series(daily[t], imo, k1d[t]) for t in P.TICKERS}
    tm["k6"], t0 = time.time() - t0, time.time()
    k1_imo = CT.k1_ext({"IMOEX": imo})["IMOEX"]            # K1 индекса на той же расширенной сетке (har_IMOEX начинается с 03.01.2017)
    k7 = {t: CT.k7_series(daily[t], imo, k1d[t], k1_imo) for t in P.TICKERS}
    tm["k7"], t0 = time.time() - t0, time.time()
    res = {}
    for name, module in (("k4", "xgb"), ("k5", "lgbm")):
        fc, ch, secs = CT.run_k45(panel, module, nthread=NTHREAD, log=_log)
        res[name] = (fc, ch)
        tm[name], t0 = time.time() - t0, time.time()
        for dpt, s in secs.items():
            tm["%s_fit_depth%d" % (name, dpt)] = s
    return {"panel": panel, "k6": k6, "k7": k7, "k4": res["k4"], "k5": res["k5"], "time": tm}


def k4567_write(r: dict) -> None:
    k = CT.assemble(r["panel"], r["k4"][0], r["k5"][0], r["k6"], r["k7"])
    k.to_parquet(P.OUT / "k4567.parquet")
    P.write_csv(CT.qlike_tables(k), "P1_K4567.csv", "QLIKE K1, K4 (XGB без HAR-J), K5 (LGBM без HAR-J), K6 (HAR-J+E8), K7 (компонентная HAR) на общих датах бумаги, цели с 2017-01-03; группы — среднее по бумагам")
    P.write_csv(pd.concat([r["k4"][1].assign(control="k4"), r["k5"][1].assign(control="k5")]), "P1_K4567_choice.csv", "выбор (глубина, c) по годам, λ ≡ 1: по закрытым прогнозам года Y−1 (t⁺ < 01.01.Y); crit — средний по бумагам QLIKE")
    P.write_csv(pd.DataFrame([{"stage": s, "seconds": round(v, 1)} for s, v in r["time"].items()]), "P1_K4567_time.csv", "время счёта по стадиям, с (fit_depth — чистое время обучения моделей)")
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    cp.update(leaf="P1", used=0, updated=time.strftime("%Y-%m-%d %H:%M"))
    (P.REP / "CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, indent=1))


def k4567_check() -> list:
    """Ошибки выхода P1b (пусто = ок)."""
    bad, k = [], pd.read_parquet(P.OUT / "k4567.parquet")
    if k.duplicated(CT.KEY).any():
        bad.append("дубли (бумага, дата)")
    for s in CT.SIG[1:]:
        v = k[s].dropna()
        if len(v) == 0 or not (np.isfinite(v) & (v > 0)).all():
            bad.append("%s: нет значений или не конечны / не положительны" % s)
    if set(CT.HAR_BASED) & {c[2:] for c in CT.K45_FEATS}:
        bad.append("K4/K5 используют признаки из прогноза HAR")
    e8 = pd.read_parquet(P.ROOT / "data/regen/harj_hybrid_v2/MKT_series.parquet")[["d", "E8"]].rename(columns={"d": "date"})
    m = k[k["ticker"] == "SBER"].merge(e8, on="date")
    dif = float(np.abs(m["sigma2_k6"] - m["E8"]).max())
    print("K6 SBER vs E8 Ж-214: строк %d, max_abs_diff = %s" % (len(m), dif))
    if len(m) < 2271 or dif != 0.0:
        bad.append("K6 SBER ≠ E8 Ж-214 побитно")
    pb = pd.read_parquet(P.OUT / "panel_base.parquet")[CT.KEY + ["sigma2_k1"]]
    mk = k.merge(pb, on=CT.KEY, suffixes=("", "_b"))
    d1 = float(np.abs(mk["sigma2_k1"] - mk["sigma2_k1_b"]).max())
    print("K1 в k4567 vs panel_base: общих строк %d, max_abs_diff = %s" % (len(mk), d1))
    bad += ["K1 расходится с panel_base"] if d1 != 0.0 or len(mk) != len(pb) else []
    print("строк %d; K4 %d, K5 %d, K6 %d, K7 %d значений; K7 min %.4g" % (len(k), *(k[s].notna().sum() for s in CT.SIG[1:]), k["sigma2_k7"].min()))
    cp = json.loads((P.REP / "CHECKPOINT.json").read_text())
    return bad + ([] if cp.get("used") == 0 else ["CHECKPOINT used != 0"])


def stage_k4567(check: bool) -> int:
    if not check:
        r = k4567_compute()
        k4567_write(r)
        for s, v in r["time"].items():
            _log("время %s: %.1f с" % (s, v))
    bad = k4567_check()
    return _report(not bad, bad)


STAGES = {"inv": stage_inv, "daily": stage_daily, "har": stage_har, "shift": stage_shift, "port": stage_port, "panel": stage_panel, "k4567": stage_k4567, "k23": stage_k23}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["hyb", "stats", "loto"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--module", choices=("xgb", "lgbm"))
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    if a.stage in ("stats", "loto"):          # P3 (G11/G12)
        from models_v2.s10 import hp_stats_run
        return hp_stats_run.stage(a.stage, a.check, a.part, a.module)
    if a.stage == "hyb":          # P2 (G9/G10): гибриды панели; --module обязателен
        from models_v2.s10 import hp_hyb_run
        return hp_hyb_run.stage(a.module, a.check) if a.module else ap.error("--stage hyb требует --module xgb|lgbm")
    return STAGES[a.stage](a.check)


if __name__ == "__main__":
    sys.exit(main())
