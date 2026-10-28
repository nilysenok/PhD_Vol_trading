"""Секторный тренд и ширина рынка — фильтры входа этапа 1, итерация 4 (Ж-051, §2.4 проекта).

Решение на close d по данным до 18:50 d (Н6), исполнение — на d+1 (ось `vhod` точки P1).
Модуль только ЗАПРЕЩАЕТ вход (`etap1_data.gate_trades`): `allow(pt, t, dates, ctx) → (long, short)`,
два булевых массива по `dates`; сторону задаёт знак позиции в день начала сделки.
Оси точки `pt` (dict или объект): sektor_trend, shirina ∈ {protiv, net, po_trendu}, ma_n, porog_shiriny.
  тренд: индекс сектора d против MA_n по d−n+1…d (po_trendu: лонг — выше, шорт — ниже; protiv — наоборот);
  ширина B_d — доля тикеров с close Н6 d > MA_n своего close (тикер без n дней истории — вне знаменателя);
  po_trendu: лонг — B ≥ p, шорт — B ≤ 1−p; protiv: лонг — B ≤ 1−p, шорт — B ≥ p.
Нет истории/нет значения в день — вход запрещён. AFLT и LSRG (индексов MOEXTN/MOEXRE в репо нет) —
секторный фильтр не действует (вход разрешён), ширина действует (❓6, умолчание дирижёра).
Обе оси net — (True, True) без чтения данных: путь P1 побайтно (golden).
Константы *_LAG и CUT_MIN — для тестов-мутаций (подмена в памяти): 0 — как в контракте.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import daily_1850 as d18

MA_LAG = 0               # MA индекса кончается в d; −1 = захватывает d+1 (мутация)
CLOSE_LAG = 0            # close индекса берётся за d; −1 = close d+1 (мутация)
SHIRINA_LAG = 0          # B_d; −1 = B_{d+1} (мутация «ширина с будущим»)
CUT_MIN = d18.CUT_MIN    # 18:50; 24·60 = в день входят вечерние бары (мутация)
BEZ_INDEKSA_VPUSKAT = True   # тикер без индекса: фильтр сектора не действует; False = запрет (мутация)
SEKTORY_CSV = "docs/issledovaniya/2026-10-27_sektory.csv"
IDX_DIR = "moex_discovery/data/final_v6/candles_10m/external"
END = pd.Timestamp("2022-01-01")
MODES = ("protiv", "net", "po_trendu")


def _g(pt, k, default):
    v = pt.get(k, default) if isinstance(pt, dict) else getattr(pt, k, default)
    return default if v is None else v


def idx_daily(bars: pd.DataFrame, name: str) -> pd.Series:
    """close индекса на 18:50 по дням из 10-минуток (колонка timestamp) через `daily_1850.aggregate`."""
    d = d18.aggregate(bars, name, cut_min=CUT_MIN)
    return d.set_index("date")["close"].astype(float)


def load_indexes(names, main=None, end: pd.Timestamp = END) -> dict:
    """{имя: close 18:50 по дням}; только < end (≤ 2022-01-01, фильтр на чтении). Нет файла — имени нет в ответе."""
    import pyarrow.parquet as pq
    if end > END:
        raise PermissionError("e1_sektor: даты ≥ 2022 не читаются")
    base = Path(main) if main else d18.default_main() / IDX_DIR    # main — каталог с {имя}.parquet
    out = {}
    for n in names:
        p = base / f"{n}.parquet"
        if p.exists():
            b = pq.read_table(p, filters=[("begin", "<", end.strftime("%Y-%m-%d"))]).to_pandas()
            b["timestamp"] = pd.to_datetime(b["begin"])
            out[n] = idx_daily(b, n)
    return out


def sektor_map(csv_path, have) -> dict:
    """{тикер: индекс} для индексов из `have`; прочие (AFLT/MOEXTN, LSRG/MOEXRE) — без индекса, не входят."""
    df = pd.read_csv(csv_path)
    return {r.ticker: r.indeks for r in df.itertuples() if r.indeks in have}


def _ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean().shift(MA_LAG)


class Ctx:
    """Ряды: idx {индекс: Series close 18:50}, close {тикер: Series close Н6}, smap {тикер: индекс}."""

    def __init__(self, idx: dict, close: dict, smap: dict):
        self.idx, self.close, self.smap, self._b = idx, close, smap, {}

    def trend(self, name: str, n: int):
        s = self.idx[name]
        c, m = s.shift(CLOSE_LAG), _ma(s, n)
        return (c > m).fillna(False), (c < m).fillna(False)   # NaN в сравнении → False → вход запрещён

    def breadth(self, n: int) -> pd.Series:
        if n not in self._b:
            ab = {}
            for t, s in self.close.items():
                m = s.rolling(n, min_periods=n).mean()
                ab[t] = (s > m).astype(float).where(m.notna() & s.notna())
            a = pd.DataFrame(ab)
            self._b[n] = (a.sum(axis=1, min_count=1) / a.count(axis=1).replace(0, np.nan)).shift(SHIRINA_LAG)
        return self._b[n]


def allow(pt, t: str, dates, ctx: Ctx | None = None):
    """(long, short): вход по close `dates[i]` разрешён. dates — datetime64 (даты тикера)."""
    st, sh = _g(pt, "sektor_trend", "net"), _g(pt, "shirina", "net")
    for v in (st, sh):
        if v not in MODES:
            raise ValueError(f"e1_sektor: режим {v!r} вне {MODES}")
    dt = pd.DatetimeIndex(dates)
    lo = np.ones(len(dt), bool)
    sr = lo.copy()
    if st == "net" and sh == "net":
        return lo, sr
    n, p = int(_g(pt, "ma_n", 200)), float(_g(pt, "porog_shiriny", 0.5))
    if st != "net":
        name = ctx.smap.get(t)
        if name is None:
            if not BEZ_INDEKSA_VPUSKAT:
                lo[:], sr[:] = False, False
        else:
            up, dn = (x.reindex(dt).fillna(False).to_numpy(bool) for x in ctx.trend(name, n))
            lo, sr = (up, dn) if st == "po_trendu" else (dn, up)
    if sh != "net":
        b = ctx.breadth(n).reindex(dt).to_numpy(float)
        hi, low = b >= p, b <= 1 - p          # NaN → False
        bl, bs = (hi, low) if sh == "po_trendu" else (low, hi)
        lo, sr = lo & bl, sr & bs
    return lo, sr


def bez_indeksa(smap: dict, tickers) -> list:
    """Тикеры, для которых секторный фильтр не действует (число входов — в отчёт)."""
    return [t for t in tickers if t not in smap]
