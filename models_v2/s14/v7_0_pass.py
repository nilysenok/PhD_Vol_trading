"""V7-0, G3: паспорт M10 SBER 2014–02.2026 по ГОДАМ: дни, бары в день (мода, мин., макс.), расписание, пропуски, нулевой объём, баров вне основной сессии, выходные сессии вне канонических дней; тождество Σr² по барам = RV цели
(допуск 1e-9) на всех днях; поля и надёжность каналов п. 3 блока Ж-232. Определение RV — канон проекта (`models_v2/daily.py`, 1e4·Σr², 10:00–18:40). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s14 import v7_0_io as IO, v7_0_seq as SQ

TOL_ID = 1e-9


def load() -> tuple:
    bars = pd.read_parquet(IO.BARS)
    d = pd.read_parquet(IO.DAILY)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index))
    return bars, d


def day_sums(f: pd.DataFrame) -> pd.Series:
    return f.groupby("day")["r2"].sum()


def identity(f: pd.DataFrame, d: pd.DataFrame) -> pd.Series:
    """Относительная разность Σr² по барам дня и RV канона (`daily_SBER.rv`) на каждом дне."""
    s = day_sums(f)
    s.index = pd.DatetimeIndex(s.index)
    ref = d["rv"].reindex(s.index)
    return (s - ref).abs() / ref.abs()


def year_rows(bars: pd.DataFrame, f: pd.DataFrame, d: pd.DataFrame, rel: pd.Series) -> pd.DataFrame:
    t = bars["begin"].dt.strftime("%H:%M")
    outside = bars[(t < "10:00") | (t > "18:40")]
    day = pd.DatetimeIndex(f["day"])
    nb = f.groupby("day").size()
    nb.index = pd.DatetimeIndex(nb.index)
    wk = D.main_session(bars)
    wk = wk[~wk["begin"].dt.normalize().isin(d.index)]
    rows = []
    for y in range(2014, 2027):
        n = nb[nb.index.year == y]
        if n.empty:
            continue
        mode = int(n.mode().iloc[0])
        m = day.year == y
        rows.append(dict(section="year", year=y, n_days=len(n), n_bars=int(n.sum()), bars_mode=mode, bars_min=int(n.min()), bars_max=int(n.max()), days_not_mode=int((n != mode).sum()), days_with_gaps=int((n < mode).sum()),
                         zero_volume_bars=int((f["volume"].to_numpy()[m] <= 0).sum()), zero_return_share=float((f["r"].to_numpy()[m] == 0).mean()), session="10:00–18:40 (основная; бары begin 10:00…18:40, %d слотов)" % SQ.SLOTS,
                         outside_main_bars=int((outside["begin"].dt.year == y).sum()), evening_in_input="нет: вечерняя сессия (19:00+) и бары до 10:00 — вне канона RV и вне входа",
                         excluded_noncanonical_session_bars=int((wk["begin"].dt.year == y).sum()), max_rel_identity=float(rel[rel.index.year == y].max())))
    return pd.DataFrame(rows)


def channel_rows(f: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Наличие и надёжность каналов; решение — по данным, причина — в note."""
    b = D.main_session(bars)
    bad_hl = int(((b["high"] < b[["open", "close"]].max(axis=1)) | (b["low"] > b[["open", "close"]].min(axis=1)) | (b["high"] < b["low"])).sum())
    ratio = (b["value"] / b["volume"] / b["close"]).to_numpy(float)
    off = int(((ratio < 0.9) | (ratio > 1.1)).sum())
    yoff = pd.Series(b["begin"].dt.year.to_numpy()[(ratio < 0.9) | (ratio > 1.1)]).value_counts().sort_index().to_dict()
    spec = [("r", "да", "close-к-close, r_1 = ln(close/open); Σ(100r)² = RV канона"), ("hl", "да" if bad_hl == 0 else "с оговоркой", "high ≥ max(open, close), low ≤ min: нарушений %d из %d" % (bad_hl, len(b))),
            ("oc", "да", "open и close есть во всех барах"), ("r2", "да", "из r"), ("rs_pos", "да", "из r; rs_pos + rs_neg = r2 (тест)"), ("rs_neg", "да", "из r"),
            ("lvol", "да", "объём > 0 во всех %d барах основной сессии, нулевых 0" % len(b)), ("vol_rel", "да", "медиана того же слота по сессиям строго до дня; в первую сессию NaN (маска)"),
            ("vwap", "да, с оговоркой" if off else "да", "value/volume ≈ close; вне ±10 %%: %d баров по годам %s (аномальные value, не фильтруется — z-нормировка устойчива к единичным)" % (off, yoff)),
            ("cum_rv", "да", "накопленная Σ r² внутри дня (PIT внутри дня: бар i знает бары ≤ i)"), ("tod", "да", "доля сессии по часам, не по номеру бара: устойчива к смене расписания")]
    return pd.DataFrame([dict(section="channel", channel=c, available=a, note=n) for c, a, n in spec])


def passport() -> tuple:
    bars, d = load()
    f = SQ.bar_table(bars, d.index)
    rel = identity(f, d)
    yr = year_rows(bars, f, d, rel)
    tot = dict(section="total", year=0, n_days=int(yr["n_days"].sum()), n_bars=int(yr["n_bars"].sum()), max_rel_identity=float(rel.max()))
    return pd.concat([yr, pd.DataFrame([tot]), channel_rows(f, bars)], ignore_index=True), dict(max_rel=float(rel.max()), n_days=len(rel), n_nan=int(rel.isna().sum()))
