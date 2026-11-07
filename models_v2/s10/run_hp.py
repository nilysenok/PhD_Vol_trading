"""Ж-215 / Ж-217, лист P0. Запуск: python3 -W ignore -m models_v2.s10.run_hp --stage inv|daily|har|shift|port|panel [--check]
Без --check стадия считает и пишет; с --check — проверяет уже записанный выход и печатает итог (код возврата 0 — гейт закрыт). Порядок: inv, daily, har, shift, port, panel. ML-обучений нет."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from models_v2.s10 import hp_daily as HD, hp_feat as F, hp_har as HH, hp_inv as HI, hp_paths as P, hp_panel as PN, hp_port as PO, hp_shift as SH

ALL = P.TICKERS + ["IMOEX"]


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


STAGES = {"inv": stage_inv, "daily": stage_daily, "har": stage_har, "shift": stage_shift, "port": stage_port, "panel": stage_panel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    return STAGES[a.stage](a.check)


if __name__ == "__main__":
    sys.exit(main())
