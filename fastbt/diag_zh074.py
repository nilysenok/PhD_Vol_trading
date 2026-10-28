"""Ж-074 (итерация 27), диагностика: добор с плечом внутри дня по журналам P1 (ядро — чистые функции; прогон — diag_zh074_run).
Определения — docs/resheniya/tolkovaniya.md [Ж-074]. Срабатывание: за первые N баров от бара входа цена ушла в сторону позиции на x·σ_d
(бар входа — по close: порядок внутри бара неизвестен; дальше — по high/low), и ни один бар от входа до бара срабатывания не закрылся хуже входа.
Добор — лимитка по close бара срабатывания, правило движка (проход на тик, со следующего бара дня, до последнего бара живой позиции).
PnL добора = (L−1)·(сторона·((выход+Σ D·0,87)/вход_добора − 1) − 2·комиссия) − (L−1)·(КС+2 %)/365·календарные дни; стоп/тейк/выход — основной.
Диагностика, не выбор: train 2018–2019; в ledger и реестр отбора не пишется."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from fastbt.exec_e_rules import limit_mask, limit_px

ROOT = Path(__file__).resolve().parents[1]
OUT, INV = ROOT / "docs/tables/diag", ROOT / "docs/tables/inv"
STATUS = "диагностика, train 2018–2019, после #400, потолок; в ledger и реестр отбора не добавлено"
GRID_N, GRID_X, LS = (6, 12, 24), (0.25, 0.5, 1.0), (1.5, 2.0)
LEVEL, PLUS = ("stop", "trail", "bu", "take", "lim"), ("stop", "trail", "bu", "take")   # выход внутри бара; из них стоп/тейк
SIDES = (("лонг", 1), ("шорт", -1), ("все", 0))
C_DOB = ["S", "комиссия_%", "сторона", "N", "x", "сделок", "сработало", "доля_срабатывания_%", "исполнено_добора",
         "доля_исполнения_от_всех_%", "pnl_Sk_L1_%", "pnl_сработавших_L1_%", "dPnL_L1,5_сумма_%", "dPnL_L1,5_на_сработавшую_%",
         "dPnL_L2_сумма_%", "dPnL_L2_на_сработавшую_%"]
C_DAY = ["S", "комиссия_%", "сторона", "сделок", "в_плюс_внутри_дня", "доля_%", "вклад_в_pnl_%", "pnl_сумма_%", "доля_вклада_%", "выходы_по_причинам"]


@dataclass
class Tr:
    side: int
    sg: float            # σ_d, по которой считаются выходы сделки
    px: float            # цена входа
    eb: int              # бар входа
    day: object          # бары дня входа (o, h, l, c)
    last_ok: int         # последний бар дня, на котором позиция жива и добор ещё может исполниться
    xpx: float
    dcash: float         # Σ D·0,87 за время удержания
    comm: float
    tick: float | None
    cal: int             # календарные дни удержания
    rate: float          # КС + 2 % годовых на дату входа
    reason: str
    lim: bool = True     # вход лимиткой (S1–S5); S6 — рынком

    @property
    def pnl(self) -> float:
        return self.side * ((self.xpx + self.dcash) / self.px - 1.0) - 2 * self.comm


def last_ok_bar(nb: int, same_day: bool, xb: int, reason: str) -> int:
    """Выход в день входа внутри бара — добор успевает до предыдущего бара; по close — до последнего; позже — весь день."""
    if not same_day:
        return nb - 1
    return xb - 1 if reason in LEVEL else xb


def trigger(t: Tr, N: int, x: float):
    """Бар срабатывания условия добора или None."""
    d, s, need = t.day, t.side, x * t.sg
    for k in range(t.eb, min(t.eb + N, len(d.c), t.last_ok + 1)):
        if s * (d.c[k] - t.px) < 0:
            return None
        far = d.c[k] if k == t.eb else (d.h[k] if s > 0 else d.l[k])
        if s * math.log(far / t.px) >= need:
            return k
    return None


def fill(t: Tr, k: int):
    """(бар, цена) исполнения добора: лимитка по close бара k, проход на тик, со следующего бара."""
    d, L = t.day, t.day.c[k]
    for m in range(k + 1, t.last_ok + 1):
        if limit_mask(t.side, d.h[m], d.l[m], L, t.tick):
            return m, limit_px(t.side, d.o[m], L)
    return None


def add_pnl(t: Tr, fpx: float, L: float) -> float:
    """PnL добора (L−1)× основной: сделка, обе комиссии, маржа только на часть сверх 100 %."""
    w = L - 1.0
    return w * (t.side * ((t.xpx + t.dcash) / fpx - 1.0) - 2 * t.comm) - w * t.rate * t.cal / 365.0


def cell(trs: list, N: int, x: float) -> dict:
    r = {"n": len(trs), "srab": 0, "isp": 0, "base": sum(t.pnl for t in trs), "srab_pnl": 0.0, **{L: 0.0 for L in LS}}
    for t in trs:
        k = trigger(t, N, x)
        if k is None:
            continue
        r["srab"] += 1
        r["srab_pnl"] += t.pnl
        f = fill(t, k)
        if f is not None:
            r["isp"] += 1
            for L in LS:
                r[L] += add_pnl(t, f[1], L)
    return r


def pnl_at(trs: list, N: int, x: float, L: float) -> float:
    """Σ PnL сделок при плече L (G2: при L = 1 — Σ журнала)."""
    tot = 0.0
    for t in trs:
        k = trigger(t, N, x)
        f = None if k is None else fill(t, k)
        tot += t.pnl + (0.0 if f is None else add_pnl(t, f[1], L))
    return tot


def plus_intraday(trs: list) -> tuple:
    """(число, Σ PnL) сделок, закрытых стопом/тейком внутри бара (до 18:40) в плюс."""
    p = [t.pnl for t in trs if t.reason in PLUS and t.pnl > 0]
    return len(p), sum(p)


def dobor_rows(sid: str, cl: str, trs: list) -> list:
    rows = []
    for nm, sd in SIDES:
        sub = [t for t in trs if sd == 0 or t.side == sd]
        for N in GRID_N:
            for x in GRID_X:
                c, n = cell(sub, N, x), max(len(sub), 1)
                per = lambda v: 100 * v / c["srab"] if c["srab"] else float("nan")      # noqa: E731
                rows.append([sid, cl, nm, N, x, c["n"], c["srab"], 100 * c["srab"] / n, c["isp"], 100 * c["isp"] / n,
                             100 * c["base"], 100 * c["srab_pnl"], 100 * c[1.5], per(c[1.5]), 100 * c[2.0], per(c[2.0])])
    return rows


def day_rows(sid: str, cl: str, trs: list) -> list:
    rows = []
    for nm, sd in SIDES:
        sub = [t for t in trs if sd == 0 or t.side == sd]
        n, s = plus_intraday(sub)
        tot = sum(t.pnl for t in sub)
        why = ";".join(f"{r}:{sum(t.reason == r for t in sub)}" for r in sorted({t.reason for t in sub}))      # чем закрыты сделки
        rows.append([sid, cl, nm, len(sub), n, 100 * n / max(len(sub), 1), 100 * s, 100 * tot, 100 * s / tot if tot else float("nan"), why])
    return rows
