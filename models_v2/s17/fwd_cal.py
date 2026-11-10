"""Л1: календарь канонических дней SBER по правилу IMOEX и список пар (d, t⁺) — ТОЛЬКО по столбцу begin (протокол, раздел 3, Р6).

Повторяет исключения `models_v2.daily.build_daily` без цен: bars<30 (баров основной сессии), nonworking_2025 (с 01.03.2025 нет бара
основной сессии IMOEX), weekend_2025, stop_2022. Правило `rv > 0` по ценам здесь не проверяется — его проверяет скрипт открытия
(полный `build_daily` после замка), расхождение — `↺`. Вечерние/утренние бары IMOEX основной сессией не считаются (`main_session`).
t⁺ — следующий канонический день (как `hp_har.target_of`), не календарный d+1.
Запуск: `python3 -m models_v2.s17.fwd_cal --write` (calendar.parquet) / `--check`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from models_v2 import daily as D
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT, Src

OUT = ROOT / "data/fwd/models/calendar.parquet"


def canonical_days(begins: pd.DataFrame, idays: set) -> pd.DataFrame:
    """begins (столбец begin SBER) и дни сессий IMOEX -> дни date, n_main (баров основной сессии)."""
    n = D.main_session(begins)["begin"].dt.normalize().value_counts()
    ex = pd.DataFrame({"date": pd.Index(begins["begin"].dt.normalize().unique()).sort_values()})
    ex["n_main"] = ex["date"].map(n).fillna(0).astype(int)
    bad = ex["n_main"] < D.MIN_BARS
    late = ex["date"] >= D.WEEKEND_FROM
    bad |= late & ~ex["date"].isin(idays)
    bad |= (ex["date"].dt.dayofweek >= 5) & late
    bad |= ex["date"].between(*D.GAP)
    return ex[~bad].reset_index(drop=True)


def build_cal(cut, src: Src) -> pd.DataFrame:
    """Календарь на момент выпуска cut: date, n_main, t_plus (следующий канонический день; у последнего NaT), new_pair."""
    cal = canonical_days(src.begins("SBER", cut), src.index_days(cut))
    cal["t_plus"] = cal["date"].shift(-1)
    cal["new_pair"] = cal["t_plus"].notna() & (cal["t_plus"] > ARCHIVE_END)
    return cal


def closed_pairs(cal: pd.DataFrame, cut) -> pd.DataFrame:
    """Закрытые к cut пары (d, t⁺): t⁺ <= cut (класс № 142: не d <= cut)."""
    return cal[cal["t_plus"].notna() & (cal["t_plus"] <= pd.Timestamp(cut))][["date", "t_plus"]].reset_index(drop=True)


def new_pairs(cal: pd.DataFrame) -> pd.DataFrame:
    """Пары нового периода: t⁺ ∈ (2026-02-27; 2026-11-10], d >= 2026-02-27."""
    return cal[cal["new_pair"] & (cal["t_plus"] <= LAST_SESSION)][["date", "t_plus"]].reset_index(drop=True)


def main(argv: list) -> int:
    cal = build_cal(LAST_SESSION, Src())
    pairs = new_pairs(cal)
    print(f"n_new={len(pairs)} first=({pairs.iloc[0,0].date()}, {pairs.iloc[0,1].date()}) last=({pairs.iloc[-1,0].date()}, {pairs.iloc[-1,1].date()})")
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(OUT) + ".tmp")
        cal.to_parquet(tmp)
        tmp.replace(OUT)
    if "--check" in argv:
        old = pd.read_parquet(OUT)
        pd.testing.assert_frame_equal(old, cal)
        print("calendar.parquet == пересчёт: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
