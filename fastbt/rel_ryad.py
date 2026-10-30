"""Относительный дневной ряд `td_rel` для оси rel_signal (V6-1a, Ж-132 п. 3, отчёт сетки v6 §2.1).

Сигнал по движению тикера сверх индекса (β = 1, без оценки): на каждом синхронном 10-минутном баре
ℓ(t) = ln P(t) − ln B(t). Бар, которого нет в одном из рядов, выпадает (внутреннее объединение по timestamp).
Бары: open_rel = O_т·(B0/O_и), close_rel = C_т·(B0/C_и); high = max(open_rel, close_rel, H_т·B0/C_и), low = min(…, L_т·B0/C_и)
(внутрибаровый экстремум пары не определён, берётся экстремум тикера в масштабе индекса на закрытии бара); volume — тикера. B0 — open первого синхронного бара индекса в переданном окне: чистый
масштаб, чтобы константный индекс давал ряд тикера побайтно; отношения не меняет, но зависит от начала окна.
Дневные OHLC — `daily_1850.aggregate` ([09:50; 18:50)), значение дня d зависит только от баров < 18:50 d.
Уровни, лимиты, стопы и σ̂ считаются по настоящему `td` тикера, не по `td_rel` (решает оценщик, V6-12).
Файлов не читает: бары подаёт вызывающий (чтение < 2022 — `daily_1850.read_bars`).
V6-1b: `td_rel_rezhim` — режимы syroy / sektor / rynok; сектор PIT (`e1_rel_sektor.sektor_pit`: на день d — последняя
ребалансировка ≤ d, Д3). Нет сектора (до первой ребалансировки, «нет», индекса нет в данных) — IMOEX с флагом, молча — нет.
Смена бенчмарка склеивается без скачка: ночной переход берётся по новому бенчмарку. `proverka_d1` — проверка сдвига (Д1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import daily_1850 as d18

REZHIMY = ("syroy", "sektor", "rynok")
F_DO_PERVOJ, F_NET_SEKTORA, F_NET_INDEKSA = "do_pervoj_rebalansirovki", "net_sektora", "net_indeksa"
F_NET_BAROV = "net_barov_indeksa"      # сектор есть, но в этот день баров индекса нет (2022-03-24/25) — откат на IMOEX
RYNOK_KEY = "__rynok__"

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


def indeks_dnya(rows, day, have) -> tuple:
    """(имя индекса | None, флаг) на день: последняя ребалансировка ≤ day; rows — [(дата, имя | None)] по возрастанию."""
    cur = [n for d0, n in rows if d0 <= pd.Timestamp(day)]
    if not cur:
        return None, F_DO_PERVOJ
    if cur[-1] is None:
        return None, F_NET_SEKTORA
    if cur[-1] not in have:
        return None, F_NET_INDEKSA
    return cur[-1], ""


def _chain(parts: dict, keys: pd.Series) -> pd.DataFrame:
    """Бары дня — из бенчмарка дня; при смене бенчмарка уровень сшивается по ночному переходу нового бенчмарка."""
    out, factor, last, prev = [], 1.0, None, None
    for d, key in keys.items():
        rb = parts[key]
        day = rb[rb["_d"] == d]
        if day.empty:
            continue
        if prev is not None and key != prev:
            before = rb[rb["_d"] < d]
            factor = 1.0 if before.empty else last / before["close"].iloc[-1]
        x =day.drop(columns="_d").copy()
        if factor != 1.0:
            x[["open", "high", "low", "close"]] *= factor
        out.append(x)
        last, prev = x["close"].iloc[-1], key
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=BAR_COLS)


def td_rel_sektor(tick, imoex, indeksy: dict, rows, ticker: str):
    """(дневной td_rel по сектору PIT с откатом на IMOEX, флаги по дням: date, benchmark, flag)."""
    days = pd.Series(sorted(pd.to_datetime(tick["timestamp"]).dt.normalize().unique()))
    pairs = [indeks_dnya(rows, d, indeksy) for d in days]
    keys = pd.Series([n or RYNOK_KEY for n, _ in pairs], index=days)
    parts = {}
    for k in set(keys) | {RYNOK_KEY}:
        rb = sinhronnye_bary(tick, imoex if k == RYNOK_KEY else indeksy[k])
        parts[k] = rb.assign(_d=rb["timestamp"].dt.normalize())
    flg = [f for _, f in pairs]
    dset = {k: set(p["_d"]) for k, p in parts.items()}
    for i, (d, k) in enumerate(keys.items()):      # дыра внутри ряда сектора (после его первого дня): день — по IMOEX, не отказ всей оценки
        if k != RYNOK_KEY and dset[k] and d > min(dset[k]) and d not in dset[k]:
            keys[d], flg[i] = RYNOK_KEY, F_NET_BAROV
    flags = pd.DataFrame({"date": days, "benchmark": ["IMOEX" if k == RYNOK_KEY else k for k in keys],
                          "flag": flg})
    b = _chain(parts, keys)
    return (d18.aggregate(b, ticker) if len(b) else td_rel(tick.iloc[0:0], imoex.iloc[0:0], ticker)), flags


def td_rel_rezhim(rezhim: str, tick, ticker: str, imoex=None, indeksy=None, rows=None):
    """Ось rel_signal: syroy = обычный td (v5 побайтно), rynok = IMOEX, sektor = PIT с откатом. Возврат (td, флаги | None)."""
    if rezhim not in REZHIMY:
        raise ValueError(f"rel_signal: {rezhim!r} вне {REZHIMY}")
    if rezhim == "syroy":
        return d18.aggregate(_check(tick, "тикер"), ticker), None
    if imoex is None:
        raise ValueError(f"rel_signal={rezhim}: нет баров IMOEX (молчаливое syroy запрещено)")
    if rezhim == "rynok":
        return td_rel(tick, imoex, ticker), None
    return td_rel_sektor(tick, imoex, indeksy or {}, rows or [], ticker)


def proverka_d1(td: pd.DataFrame, ref: pd.Series, lags=(-2, -1, 0, 1)) -> dict:
    """Д1: corr(r_rel_d, ref_{d+lag}); r_rel — лог-доходность close td_rel, ref — (r_тикера − r_бенчмарка) по дням.
    Максимум обязан быть на lag 0, иначе ValueError. Возврат {lag: corr}."""
    x = pd.Series(np.log(td["close"].to_numpy(float)), index=pd.to_datetime(td["date"])).diff()
    ref = ref.copy()
    ref.index = pd.to_datetime(ref.index)
    res = {L: float(x.corr(ref.shift(-L))) for L in lags}
    if np.isnan(list(res.values())).any():
        raise ValueError(f"Д1: корреляция не определена (мало наблюдений): {res}")
    if max(res, key=res.get) != 0:
        raise ValueError(f"Д1: максимум корреляции не на lag 0: {res}")
    return res
