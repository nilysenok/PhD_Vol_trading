"""Относительный дневной ряд `td_rel` для оси rel_signal (V6-1a, Ж-132 п. 3, отчёт сетки v6 §2.1).

Сигнал по движению тикера сверх индекса (β = 1, без оценки): на каждом синхронном 10-минутном баре
ℓ(t) = ln P(t) − ln B(t). Бар, которого нет в одном из рядов, выпадает (внутреннее объединение по timestamp).
Бары: open_rel = O_т·(B0/O_и), close_rel = C_т·(B0/C_и); high = max(open_rel, close_rel, H_т·B0/C_и), low = min(…, L_т·B0/C_и)
(внутрибаровый экстремум пары не определён, берётся экстремум тикера в масштабе индекса на закрытии бара); volume — тикера. B0 — open первого синхронного бара индекса в переданном окне: чистый
масштаб, чтобы константный индекс давал ряд тикера побайтно; отношения не меняет, но зависит от начала окна.
Дневные OHLC — `daily_1850.aggregate` ([09:50; 18:50)), значение дня d зависит только от баров < 18:50 d.
Уровни, лимиты, стопы и σ̂ считаются по настоящему `td` тикера, не по `td_rel` (решает оценщик, V6-12).
Файлов не читает: бары подаёт вызывающий (чтение < 2022 — `daily_1850.read_bars`). Сектор PIT — V6-1b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import daily_1850 as d18

BAR_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _check(b: pd.DataFrame, name: str) -> pd.DataFrame:
    miss = [c for c in BAR_COLS if c not in b.columns]
    if miss:
        raise ValueError(f"{name}: нет колонок {miss}")
    b = b[BAR_COLS].sort_values("timestamp")
    if b["timestamp"].duplicated().any():
        raise ValueError(f"{name}: дубли timestamp")
    px = b[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(px).all() or (px <= 0).any():
        raise ValueError(f"{name}: цена не положительна или не конечна")
    return b


def sinhronnye_bary(tick: pd.DataFrame, indeks: pd.DataFrame) -> pd.DataFrame:
    """Относительные бары на пересечении timestamp (колонки BAR_COLS). Пересечения нет — пустой кадр."""
    t, i = _check(tick, "тикер"), _check(indeks, "индекс")
    m = t.merge(i[["timestamp", "open", "close"]], on="timestamp", suffixes=("", "_i"))
    if m.empty:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "timestamp" else float) for c in BAR_COLS})
    b0 = m["open_i"].iloc[0]
    o = m["open"] * (b0 / m["open_i"])
    c = m["close"] * (b0 / m["close_i"])
    k = b0 / m["close_i"]
    out = pd.DataFrame({"timestamp": m["timestamp"], "open": o, "high": np.maximum(np.maximum(o, c), m["high"] * k),
                        "low": np.minimum(np.minimum(o, c), m["low"] * k), "close": c, "volume": m["volume"]})
    return out.reset_index(drop=True)


def td_rel(tick: pd.DataFrame, indeks: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Дневные OHLCV относительного ряда (колонки как у `daily_1850.aggregate`)."""
    b = sinhronnye_bary(tick, indeks)
    if b.empty:
        kinds = {"ticker": object, "date": "datetime64[ns]"}
        return pd.DataFrame({c: pd.Series(dtype=kinds.get(c, float)) for c in d18.COLS})
    return d18.aggregate(b, ticker)
