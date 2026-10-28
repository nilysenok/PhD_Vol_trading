"""Ж-069 п. 5: PnL неисполненных лимиток, если бы вход был по close d+1 (правило движка `f = zakrytie_d1`: вход только
при живом сигнале на d+1; выходы и комиссия — те же правила точки). Контрфакт считается на копии состояния Run в момент
неисполнения (боевой прогон не меняется). Основа — журнал заявок diag_zh066 (#374). Диагностика, не выбор: train 2018–2019,
комиссия 0,04 %, движок после #370; PnL — в определении движка (Д7): сторона·((выход + Σ D·0,87)/вход − 1) − комиссии.
Сборка: FASTBT_MAIN=$PWD python3 -m fastbt.diag_zh069 (S1–S5; S6 входит рынком, лимиток нет)."""
from __future__ import annotations

import copy
import csv
import hashlib
import subprocess
from contextlib import contextmanager
from pathlib import Path

from fastbt import diag_zh066 as dz
from fastbt import diag_zh066_run as run66
from fastbt import exec_e_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tables/diag/zh069_neispolnennye.csv"
STATUS = ("диагностика, train 2018–2019, после дивидендов (#370), до аудита; комиссия 0,04 %; PnL с дивидендом D·0,87 как в движке; "
          "в ledger и реестр отбора не добавлено")
NAMES = ["S1", "S2", "S3", "S4", "S5"]
COLS = ["S", "выставлено", "исп_закрыто", "исп_pnl_сумма_%", "исп_pnl_сделка_%", "исп_открыто_на_конце",
        "неисп_всего", "неисп_без_сигнала_d+1", "неисп_закрыто", "неисп_pnl_сумма_%", "неисп_pnl_сделка_%", "неисп_открыто_на_конце"]


def entry_px(run, i: int) -> float:
    """Цена контрфактического входа: close дня исполнения заявки (d+1)."""
    return run.close_of(i)


def _fork(s):
    """Копия состояния Run для контрфакта: массивы учёта и журнал сделок свои, заявка снята."""
    f = copy.copy(s)
    for k in ("gross", "cost", "pos", "used", "free"):
        setattr(f, k, getattr(s, k).copy())
    f.trades, f.order, f.tr, f._shadow = [], None, None, True
    return f


def close_entry(sh, i: int, j: int, o: dict) -> dict:
    """Вход по close d+1 (день i, строка j) и ход до выхода по правилам точки. live=False — сигнала на d+1 нет, не входим."""
    sd = o["sd"]
    if sh.dir(i) != sd:
        return dict(live=False)
    nb = len(sh.days[i].c)
    sh._enter(i, j, nb - 1, entry_px(sh, i), nb, sh.sigma[i], o["d_sig"], sd)
    for jj in range(j + 1, len(sh.ix)):
        sh.step(jj, sh.ix[jj])
        if sh.tr is None:
            break
    t = sh.trades[0]
    return dict(live=True, closed=bool(t.reason), pnl=trade_pnl(t) if t.reason else float("nan"))


def div_cash(s, i: int) -> float:
    """Дивиденд, пересёкший позицию на шаге дня i: D·nalog (как `Run.step`: dv = div[i]·nal, если позиция есть с прошлого close)."""
    return float(s.div[i] * s.nal)


def trade_pnl(t) -> float:
    """PnL сделки в определении движка после #370: сторона·((выход + Σ D·0,87)/вход − 1) − комиссии (лонг получает D, шорт платит)."""
    return t.side * ((t.exit_px + getattr(t, "dcash", 0.0)) / t.entry_px - 1.0) - t.fee_in - t.fee_out


@contextmanager
def log_counterfactual(sink: list):
    """Поверх diag_zh066_run.log_orders: к записи неисполненной заявки добавляет `cf` (результат close_entry),
    к записи исполненной — `ex` (закрытая боевая сделка с PnL в определении движка). Боевой счёт не меняется."""
    inner, step, exit_ = exec_e_run.Run._try_order, exec_e_run.Run.step, exec_e_run.Run._exit

    def _try(s, i, j, d):
        o, nf, sh = s.order, s.n_fills, _fork(s)
        inner(s, i, j, d)
        if s.n_fills == nf:
            sink[-1]["cf"] = close_entry(sh, i, j, o)

    def _step(s, j, i):
        if s.tr is not None:                       # позиция с прошлого close пересекает отсечку дня i
            s.tr.dcash = getattr(s.tr, "dcash", 0.0) + div_cash(s, i)
        step(s, j, i)

    def _exit(s, i, j, b, px, reason):
        t = s.tr
        exit_(s, i, j, b, px, reason)
        if not getattr(s, "_shadow", False):
            rec = next(r for r in reversed(sink) if r["tiker"] == s.t and r["d_sig"] == t.d_sig and r["isp"])
            rec["ex"] = dict(pnl=trade_pnl(t))

    exec_e_run.Run._try_order, exec_e_run.Run.step, exec_e_run.Run._exit = _try, _step, _exit
    try:
        yield
    finally:
        exec_e_run.Run._try_order, exec_e_run.Run.step, exec_e_run.Run._exit = inner, step, exit_


def unfilled(orders: list) -> list:
    return [o for o in orders if not o["isp"]]


def row(sid: str, executed: list, orders: list) -> tuple:
    """Строка таблицы. executed — закрытые сделки прогона (pnl долей); orders — заявки с d_sig ≥ 2018 (isp, cf у неисполненных)."""
    ex, un = [t["pnl"] for t in executed], unfilled(orders)
    fills = sum(bool(o["isp"]) for o in orders)
    live = [o["cf"] for o in un if o["cf"]["live"]]
    cl = [c["pnl"] for c in live if c["closed"]]
    assert fills + len(un) == len(orders) and len(ex) <= fills, (sid, len(ex), fills, len(un))
    e, c = dz._stat(ex), dz._stat(cl)
    return (sid, len(orders), e[0], e[1], e[2], fills - len(ex), len(un), len(un) - len(live), c[0], c[1], c[2], len(live) - len(cl))


def check_g2(rows: list) -> list:
    """G2: исполненные (закрытые + открытые) + неисполненные = выставлено; плюс сверка с zh066_g_limitki.csv (выставлено, исполнено)."""
    ref = {}
    for r in csv.reader(l for l in open(dz.OUT / "zh066_g_limitki.csv", encoding="utf-8") if not l.startswith("#")):
        if len(r) > 3 and r[1] == "все":
            ref[r[0]] = (int(r[2]), int(r[3]))
    notes = []
    for r in rows:
        assert r[2] + r[5] + r[6] == r[1], r
        if r[0] in ref and ref[r[0]] != (r[1], r[2] + r[5]):
            notes.append(f"{r[0]}: выставлено/исполнено {r[1]}/{r[2] + r[5]} против {ref[r[0]][0]}/{ref[r[0]][1]} в zh066_g_limitki.csv (до дивидендов)")
    return notes


def main() -> None:
    rows, digests = [], []
    for sid in NAMES:
        run66.log_orders = _both                      # run_point берёт журнал заявок из своего модуля
        try:
            r = run66.run_point(sid)
        finally:
            run66.log_orders = _LOG
        digests.append(f"{sid}:{r['vhody']}")
        orders = [o for o in r["orders"] if o["d_sig"] >= "2018-01-01"]
        rows.append(row(sid, [o["ex"] for o in orders if "ex" in o], orders))
        print(sid, rows[-1], flush=True)
    for n in check_g2(rows):
        print("РАСХОЖДЕНИЕ:", n)
    h = hashlib.sha256(b"".join((dz.OUT / f"{s}_sdelki.csv").read_bytes() for s in NAMES)).hexdigest()[:16]
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dg = ",".join(digests)
    dz.write_csv(OUT, {"статус": STATUS, "коммит кода": code, "хэш входов": f"журналы sha256[:16]={h}; digest входов движка {dg}"}, COLS, rows)


_LOG = run66.log_orders


@contextmanager
def _both(sink: list):
    with _LOG(sink), log_counterfactual(sink):
        yield


if __name__ == "__main__":
    main()
