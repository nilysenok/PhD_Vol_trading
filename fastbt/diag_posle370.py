"""Ж-058 / Ж-069 п. 6: повтор диагностики #366 и децилей Ж-068 на движке после #370. Только вызов существующих модулей.
«До» = тот же движок с подменой в памяти `dividendy.gap_divs` / `adjust_ohlc` (как `dividendy_bylo_stalo`, вариант
bez_dividendov); перед записью «до» сверяется с журналом `S{k}_sdelki.csv` (число сделок, Σ pnl). «После» — штатный движок,
PnL сделки в определении движка: сторона·((выход + Σ D·0,87)/вход − 1) − комиссии (`diag_zh069.trade_pnl`).
Диагностика, не выбор: train 2018–2019; в ledger и реестр отбора не добавлено.
Запуск: FASTBT_MAIN=$PWD FASTBT_BARS1M=<D25/bars1m> python3 -m fastbt.diag_posle370"""
from __future__ import annotations

import csv
import subprocess
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from fastbt import dividendy, etap1, exec_e_run
from fastbt import diag_zh066 as dz
from fastbt import diag_zh066_run as run66
from fastbt import diag_zh069 as dz9
from fastbt.etap1_eval import YAKOR, Evaluator
from fastbt.metriki_e import msr

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "docs/tables/diag"
STATUS = "диагностика, train 2018–2019, после #370; в ledger и реестр отбора не добавлено"
NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]
COLS = ["data_vhoda", "data_vykhoda", "tiker", "napravlenie", "tsena_vkhoda", "tsena_vykhoda", "prichina", "pnl",
        "god_vkhoda", "dividend_D087_dolya_vkhoda"]
TOL = 2e-6                      # журналы «до» округлены до 1e-6 на сделку


@contextmanager
def before370(adjust: bool = True, gaps: bool = True):
    """Прежнее поведение (до #370): событий нет (gaps), ряд для сигналов не корректируется (adjust). Порознь — для разбора G4."""
    with ExitStack() as st:
        if gaps:
            st.enter_context(mock.patch.object(dividendy, "gap_divs", lambda t, d: np.zeros(len(d))))
        if adjust:
            st.enter_context(mock.patch.object(dividendy, "adjust_ohlc", lambda t, d, close, *b: tuple(np.array(x, float) for x in b)))
        yield


@contextmanager
def accrue_dividends():
    """На время прогона копит на сделке dcash = Σ D·0,87 (как в `diag_zh069.log_counterfactual._step`); расчёт не меняет."""
    step = exec_e_run.Run.step

    def _step(s, j, i):
        if s.tr is not None:
            s.tr.dcash = getattr(s.tr, "dcash", 0.0) + dz9.div_cash(s, i)
        step(s, j, i)

    exec_e_run.Run.step = _step
    try:
        yield
    finally:
        exec_e_run.Run.step = step


def trade_rows(trades, d, start, posle: bool) -> list:
    """Строки журнала по закрытым сделкам со входом ≥ start (индекс дня), по (дата входа, тикер)."""
    out = []
    for tr in trades:
        if not tr.reason or tr.entry_day < start:
            continue
        pnl = dz9.trade_pnl(tr) if posle else tr.side * (tr.exit_px / tr.entry_px - 1.0) - tr.fee_in - tr.fee_out
        out.append([str(d[tr.entry_day])[:10], str(d[tr.exit_day])[:10], tr.ticker, "long" if tr.side > 0 else "short",
                    f"{tr.entry_px:.6g}", f"{tr.exit_px:.6g}", tr.reason, float(f"{pnl:.6f}"), str(d[tr.entry_day])[:4],
                    float(f"{getattr(tr, 'dcash', 0.0) / tr.entry_px:.6f}")])
    return sorted(out, key=lambda r: (r[0], r[2]))


def msr_by_year(ex) -> dict:
    return {y: float(msr(ex[ex.index.year == y].to_numpy())) for y in (2018, 2019)}


def run_variant(sid: str, pt: dict, posle: bool, raw_signals: bool = False) -> dict:
    """posle=False — «до»; posle=True — штатный движок; raw_signals=True (с posle) — D считается, ряд сигналов сырой (G4)."""
    with ExitStack() as st:
        if not posle:
            st.enter_context(before370())
        elif raw_signals:
            st.enter_context(before370(adjust=True, gaps=False))
        inp = etap1.build_inputs(sid)
        evr = Evaluator(sid, inp, etap1.base_point(sid), 0.0004)
        if posle:
            st.enter_context(accrue_dividends())
        df, _, _, _, rs = evr.excess({**YAKOR, **pt}, "train")
    rows, trades = [], {}
    for run in rs:
        for tr in run.trades:
            trades.setdefault(tr.ticker, []).append(tr)
    for t, trs in trades.items():
        d = inp.meta[t]["dates"]
        rows += trade_rows(trs, d, int(np.searchsorted(d, np.datetime64("2018-01-01"))), posle)
    rows.sort(key=lambda r: (r[0], r[2]))
    return dict(rows=rows, msr=msr_by_year(df["excess"]), digest=inp.digest())


def old_journal(sid: str, suf: str) -> list:
    path = DIAG / f"{sid}_sdelki{suf}.csv"
    return [float(r["pnl"]) for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#"))]


def reconcile_before(sid: str, suf: str, rows: list) -> tuple:
    """«До» воспроизведено: число сделок равно журналу, Σ pnl — в пределах округления журнала."""
    old = old_journal(sid, suf)
    s_new, s_old = sum(r[7] for r in rows), sum(old)
    if len(rows) != len(old) or abs(s_new - s_old) > TOL * max(len(old), 1):
        raise SystemExit(f"{sid}{suf}: «до» не воспроизведено: сделок {len(rows)} против {len(old)}, Σ {s_new} против {s_old}")
    return len(old), s_old


def code() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def g4(sid: str, p: dict, after: dict, before: dict) -> None:
    """G4: «до» → (сырой ряд сигналов + D) → «после». Сделок меняет ряд сигналов, PnL — D·0,87 (печать для отчёта)."""
    mid = run_variant(sid, p, True, raw_signals=True)
    f = lambda r: (len(r["rows"]), round(100 * sum(x[7] for x in r["rows"]), 2))      # noqa: E731
    print("G4", sid, "до", f(before), "сырые сигналы+D", f(mid), "после", f(after), flush=True)


def decili() -> None:
    """Ж-069 п. 6: `diag_zh068.run_sid` на штатном (скорректированном) ряду; рядом n и доходность «до» из прежней таблицы."""
    from fastbt import diag_zh068 as z8
    old = {}
    for r in csv.DictReader(l for l in open(z8.OUT, encoding="utf-8") if not l.startswith("#")):
        old[(r["S"], r["дециль"])] = (r["n"], r["доходность_ср_%"])
    rows, dig = [], []
    for sid in NAMES:
        r = z8.run_sid(sid)
        dig.append(f"{sid}:{r['digest']}")
        for x in z8.deciles(r["obs"], r["tau"]):
            rows.append([sid, r["tau"], x[0], *old[(sid, x[0])], *x[1:]])
        print("децили", sid, "tau", r["tau"], "сигналов (голая идея)", len(r["obs"]), "с фильтрами", r["n_filtered"], flush=True)
    cols = ["S", "tau", "дециль", "n_до", "доходность_ср_%_до", "n", "сила_мин", "сила_макс", "доходность_ср_%", "доля_плюс", "t_NW", "метка"]
    meta = {"статус": z8.STATUS.replace("без дивидендов, до аудита", "после #370 (сигналы на скорректированном ряду), до аудита"),
            "коммит кода": code(), "хэш входов": "; ".join(dig), "до": "zh068_a_decili_signaly.csv (n_до, доходность_ср_%_до)"}
    dz.write_csv(DIAG / "zh068_a_decili_signaly_posle370.csv", meta, cols, rows)


def main() -> None:
    summary, base_code = [], code()
    for sid in NAMES:
        pt = run66.point(sid)
        variants = [("", pt)] + ([("_ks2", {**pt, "k_s": 2.0})] if pt.get("k_s") == "net" else [])
        for suf, p in variants:
            b, a = run_variant(sid, p, False), run_variant(sid, p, True)
            n0, s0 = reconcile_before(sid, suf, b["rows"])
            na, sa = len(a["rows"]), sum(r[7] for r in a["rows"])
            meta = {"статус": STATUS, "коммит кода": base_code, "хэш входов": a["digest"], "точка": f"{p}; стороны-сделки: {na}"}
            dz.write_csv(DIAG / f"{sid}_sdelki{suf}_posle370.csv", meta, COLS, a["rows"])
            summary.append([sid, "ks2" if suf else "база", n0, 100 * s0, b["msr"][2018], b["msr"][2019],
                            na, 100 * sa, a["msr"][2018], a["msr"][2019]])
            print(summary[-1], flush=True)
            if not suf:
                g4(sid, p, a, b)
    cols =["S", "вариант", "сделок_до", "Σ_PnL_%_до", "mSR_2018_до", "mSR_2019_до",
            "сделок_после", "Σ_PnL_%_после", "mSR_2018_после", "mSR_2019_после"]
    dz.write_csv(DIAG / "zh058_do_posle.csv", {"статус": STATUS + "; «до» — журналы #366 (сверено), mSR «до» — тот же движок с подменой adjust_ohlc/gap_divs",
                                               "коммит кода": base_code}, cols, summary)
    decili()


if __name__ == "__main__":
    main()
