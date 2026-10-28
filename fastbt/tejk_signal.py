"""Ось `tejk_signal` (Ж-084): тейк от уровня сигнала M; значения net | k_srednemu | pol_puti; только S1, S2, S5, S6.

M на день сигнала d — только по данным ≤ close d (18:50, Н6), дальше не обновляется (exec_e_run фиксирует тейк на входе):
 S1 — SMA(ma_window), S2 — SMA(bb_window): та же `indicators.calc_sma` по тому же close, что в fastbt/signals/s12.py;
 S5 — P пивотов текущего периода (по H/L/C прошлого периода, `extra["P"]` семейства s56);
 S6 — VWAP_w (`extra["vwap"]` семейства s56).
Уровень тейка и исполнение (проход на тик, стоп первым, перенос через ночь) — exec_e_rules.tejk_level, exec_e_run.Run.
Здесь же ряды отчёта по годам (Ж-084 п. 1): по закрытым сделкам; доходность сделки — без веса, с комиссией, без дивидендов.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fastbt.exec_e_rules import tejk_level
from fastbt.indicators import calc_sma, calc_std
from fastbt.signals.s56 import _gid
from fastbt.signals.s56_kern import _pivots, _vwap

NET, ZNACHENIYA = "net", ("net", "k_srednemu", "pol_puti")
RODNOY = "rodnoy_uroven"                                    # Ж-091 п. 1.1: заявка на родном уровне выхода, обновляется каждый вечер
OKNO = {"S1": "ma_window", "S2": "bb_window"}               # SMA-окно сигнала
DIAG = "diag.jsonl"
SIDS = ("S1", "S2", "S5", "S6")
NAN = float("nan")


def uroven_M(sid: str, sp: dict, dn: dict) -> np.ndarray:
    """M[тикер, d]; dn — ndarray [тикер, d]: close, high, low, volume, dates. M[:, d] использует только столбцы ≤ d.
    S5/S6 — те же `_pivots`/`_vwap`/`_gid`, что в signals/s56 (копии нет). Нет уровня — nan."""
    if sid not in SIDS:
        raise ValueError(f"{sid}: у стратегии нет оси tejk_signal (только {', '.join(SIDS)})")
    f = {k: np.asarray(dn[k], float) for k in ("close", "high", "low", "volume")}
    rows = []
    for i in range(f["close"].shape[0]):
        c, h, l, v = (f[k][i] for k in ("close", "high", "low", "volume"))
        if sid in OKNO:
            rows.append(calc_sma(c, int(sp[OKNO[sid]])))
        elif sid == "S5":
            rows.append(_pivots(_gid(dn["dates"], sp["pivot_period"]), h, l, c, sp["pivot_type"] == "fibonacci")[0])
        else:
            rows.append(_vwap(c, h, l, v, int(sp["vwap_window"]))[0])
    return np.array(rows, float)


def uroven(sid: str, sp: dict, td) -> np.ndarray:
    """M[d] одного тикера по его дневному ряду td (close/high/low/volume/dates, Н6)."""
    dn = {k: np.asarray(getattr(td, k))[None] for k in ("close", "high", "low", "volume")}
    return uroven_M(sid, sp, {**dn, "dates": td.dates})[0]


def rodnoy_uroven(sid: str, sp: dict, dn: dict, sd: int = 1) -> np.ndarray:
    """Ж-091 п. 1.1: L[тикер, d] — цена, где по close d сработал бы родной сигнальный выход стороны sd (+1 лонг, −1 шорт).
    S1 — SMA − sd·z_exit·STD (|z| < z_exit); S2 — SMA; S5 — P; S6 — VWAP − sd·0,5·dev. Те же calc_sma/calc_std/_pivots/_vwap,
    что в signals/s12 и s56 (копии нет). L[:, d] использует только столбцы ≤ d (данные ≤ 18:50 d). Нет уровня — nan."""
    if sid not in SIDS:
        raise ValueError(f"{sid}: у стратегии нет оси tejk_signal (только {', '.join(SIDS)})")
    f = {k: np.asarray(dn[k], float) for k in ("close", "high", "low", "volume")}
    rows = []
    for i in range(f["close"].shape[0]):
        c, h, l, v = (f[k][i] for k in ("close", "high", "low", "volume"))
        if sid == "S1":
            w = int(sp["ma_window"])
            rows.append(calc_sma(c, w) - sd * float(sp["z_exit"]) * calc_std(c, w))
        elif sid == "S6":
            vw, dev = _vwap(c, h, l, v, int(sp["vwap_window"]))
            rows.append(vw - sd * 0.5 * dev)
        else:
            rows.append(uroven_M(sid, sp, {k: f[k][i:i + 1] for k in f} | {"dates": dn["dates"]})[0])
    return np.array(rows, float)


def rodnoy_tikera(sid: str, sp: dict, td) -> np.ndarray:
    """[2, d] одного тикера по дневному ряду td: строка 0 — уровень для лонга, строка 1 — для шорта."""
    dn = {k: np.asarray(getattr(td, k))[None] for k in ("close", "high", "low", "volume")} | {"dates": td.dates}
    return np.vstack([rodnoy_uroven(sid, sp, dn, sd)[0] for sd in (1, -1)])


def uroven_tejka(tejk: str, sd: int, p_vh: float, M: float):
    """Уровень тейка без округления по тику (Ж-084); None — тейка нет (M не в сторону прибыли)."""
    return tejk_level(sd, p_vh, M, tejk, None)


def m_tikera(evr, sp: dict, t: str, s0: int, b: int, rodnoy: bool = False) -> np.ndarray:
    """M для строк [s0; s0+b] тикера t оценщика evr; rodnoy — [2, b+1] (лонг, шорт), родной уровень выхода."""
    if rodnoy:
        return rodnoy_tikera(evr.sid, sp, evr.sk._td[t][0])[:, s0:s0 + b + 1]
    return uroven(evr.sid, sp, evr.sk._td[t][0])[s0:s0 + b + 1]


def pole_otcheta(evr, years) -> dict:
    """Поля строки ledger: ряды по годам и n_bez_tejka — по последнему evr.excess (evr._last)."""
    rs = evr._last[4]
    pary = [(r.trades, evr.inp.meta[t]["dates"]) for t, r in zip(evr.sk.tickers, rs)]
    return {**otchet(pary, years), "n_bez_tejka": sum(r.n_bez_tejka for r in rs)}


def pisat_diag(papka, row: dict, diag) -> None:
    """Ж-087: papka — каталог потока писателя (Pisatel.pd), явно; диагностика рядом с ledger — p{NN}/diag.jsonl, по строке на строку ledger; ключ — klyuch_sha (с kod).
    diag = None у строк не ok. Формат ledger v2.0 и Chitatel не затронуты; дубль после обрыва — читатель берёт последнюю."""
    with open(Path(papka) / DIAG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"klyuch_sha": row["klyuch_sha"], "kod": row.get("kod"), "diag": diag}, sort_keys=True) + "\n")


def chitat_diag(katalog) -> dict:
    """{klyuch_sha: diag} по всем p*/diag.jsonl каталога стратегии."""
    out = {}
    for p in sorted(Path(katalog).glob("p*/" + DIAG)):
        for ln in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            out[r["klyuch_sha"]] = r["diag"]
    return out


def dohodnost(tr) -> float:
    return tr.side * math.log(tr.exit_px / tr.entry_px) - tr.fee_in - tr.fee_out


def _god(items: list) -> tuple:
    """items — [(доходность, выход по тейку)] → (доля тейков, доля прибыльных, средний выигрыш, средний проигрыш, топ-5 %)."""
    if not items:
        return (NAN,) * 5
    a, tk = np.array([r for r, _ in items], float), np.array([t for _, t in items], bool)
    win, loss = a[a > 0], a[a < 0]
    k, tot = max(1, math.ceil(0.05 * a.size)), float(a.sum())
    return (float(tk.mean()), win.size / a.size, float(win.mean()) if win.size else NAN,
            float(loss.mean()) if loss.size else NAN, float(np.sort(a)[::-1][:k].sum()) / tot if tot > 0 else NAN)


def otchet(pary: list, years) -> dict:
    """pary — [(сделки тикера, даты тикера)]. Поля строки ledger по годам (ключи — строки), год — по дню выхода:
    dolya_tejk_y (доля выходов по тейку среди закрытых), dolya_pribyl_y, vyigrysh_sredn_y / proigrysh_sredn_y
    (средняя доходность прибыльной / убыточной сделки), top5_pnl_y (сумма лучших 5 % сделок / вся сумма; nan при сумме ≤ 0)."""
    by = {int(y): [] for y in years}
    for trades, dates in pary:
        yr = np.asarray(dates, "datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
        for tr in trades:
            if tr.reason and int(yr[tr.exit_day]) in by:
                by[int(yr[tr.exit_day])].append((dohodnost(tr),tr.reason == "take"))
    cols = {y: _god(v) for y, v in by.items()}
    names = ("dolya_tejk_y", "dolya_pribyl_y", "vyigrysh_sredn_y", "proigrysh_sredn_y", "top5_pnl_y")
    return {n: {str(y): cols[y][i] for y in cols} for i, n in enumerate(names)}
