"""Относительная сила тикера к сектору — фильтр входа этапа 1, итерация 21 (Ж-051, контракт §2Б.3).

Решение на close d по данным до 18:50 d (Н6), исполнение — на d+1. Модуль только ЗАПРЕЩАЕТ вход:
`allow(pt, t, dates, ctx) → (long, short)`, как `e1_sektor.allow`.
Оси точки `pt`: rel_napravlenie ∈ {net, po_sile, protiv}, rel_k, rel_okno (w).
  rel_d = Σ_{w дней} (r_t − r_sek,t) / (σ̂_{t,d−1}·√w);  r — лог-доходности close Н6, r_sek — по
  `e1_sektor.idx_daily` (close 18:50), σ̂ — rv22 тикера: значение d−1 берётся из ряда σ̂ (индекс — день d).
  po_sile: лонг при rel ≥ k, шорт при rel ≤ −k;  protiv: наоборот.
Сектор тикера — PIT по `2026-10-28_sostav_sektorov.csv`: на день d действует последняя ребалансировка ≤ d;
«нет» (LSRG до 2021-06-21) и индекс, которого нет в `ctx.idx` (MOEXTN, MOEXRE), — фильтр не действует
(вход разрешён). Нет значения rel у тикера с индексом — вход запрещён. net — (True, True), данные не читаются.
Константы *_LAG — для тестов-мутаций (подмена в памяти): значения по умолчанию — как в контракте.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import e1_sektor as sek

CLOSE_LAG = 0       # close индекса за d; −1 = close d+1 (мутация, контракт §2Б.3)
SIGMA_LAG = 1       # σ̂ берётся за d−1; 0 = σ̂_d (мутация)
SEKTORY_CSV = "docs/issledovaniya/2026-10-28_sostav_sektorov.csv"
NO_INDEX = "нет"
MODES = ("net", "po_sile", "protiv")
load_indexes = sek.load_indexes


def sektor_pit(csv_path=SEKTORY_CSV) -> dict:
    """{тикер: [(дата ребалансировки, индекс | None)]} по возрастанию даты."""
    df = pd.read_csv(csv_path, parse_dates=["дата_ребалансировки"]).sort_values("дата_ребалансировки")
    out: dict = {}
    for r in df.itertuples(index=False):
        idx = None if r.индекс == NO_INDEX else r.индекс
        out.setdefault(r.тикер, []).append((r.дата_ребалансировки, idx))
    return out


def _names(rows, dt: pd.DatetimeIndex) -> pd.Series:
    """Имя индекса на каждый день dt (последняя ребалансировка ≤ d; до первой — None)."""
    s = pd.Series(None, index=dt, dtype=object)
    for d0, name in rows:
        s[dt >= d0] = name
    return s


class Ctx:
    """idx {индекс: close 18:50}, close {тикер: close Н6}, sigma {тикер: σ̂}, pit = sektor_pit()."""

    def __init__(self, idx: dict, close: dict, sigma: dict, pit: dict):
        self.idx, self.close, self.sigma, self.pit = idx, close, sigma, pit

    def rel(self, t: str, w: int):
        """(rel_d по дням close[t], имя индекса по дням). rel = NaN, где значения нет или индекса нет."""
        c = self.close[t]
        names = _names(self.pit.get(t, []), c.index)
        rel = pd.Series(np.nan, index=c.index)
        r = np.log(c).diff()
        sg = self.sigma[t].reindex(c.index).shift(SIGMA_LAG)
        for name in {n for n in names if n is not None and n in self.idx}:
            ix = self.idx[name].shift(CLOSE_LAG).reindex(c.index)
            z = (r - np.log(ix).diff()).rolling(w, min_periods=w).sum() / (sg * np.sqrt(w))
            rel[names == name] = z[names == name]
        return rel, names


def allow(pt, t: str, dates, ctx: Ctx | None = None):
    """(long, short): вход по close `dates[i]` разрешён."""
    mode = sek._g(pt, "rel_napravlenie", "net")
    if mode not in MODES:
        raise ValueError(f"e1_rel_sektor: режим {mode!r} вне {MODES}")
    dt = pd.DatetimeIndex(dates)
    lo, sr = np.ones(len(dt), bool), np.ones(len(dt), bool)
    if mode == "net":
        return lo, sr
    k, w = float(sek._g(pt, "rel_k", 1.0)), int(sek._g(pt, "rel_okno", 5))
    rel, names = ctx.rel(t, w)
    z = rel.reindex(dt).to_numpy(float)
    active = np.array([n is not None and n in ctx.idx for n in names.reindex(dt)], bool)
    up, dn = z >= k, z <= -k                      # NaN → False
    a, b = (up, dn) if mode == "po_sile" else (dn, up)
    return np.where(active, a, True), np.where(active, b, True)


def bez_indeksa(pit: dict, have, tickers, day) -> list:
    """Тикеры, у которых на день `day` фильтр не действует (число входов — в отчёт)."""
    d = pd.Timestamp(day)
    out = []
    for t in tickers:
        cur = [n for d0, n in pit.get(t, []) if d0 <= d]
        if not cur or cur[-1] is None or cur[-1] not in have:
            out.append(t)
    return out
