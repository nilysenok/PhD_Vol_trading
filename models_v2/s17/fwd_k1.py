"""Л1: K1 (HAR-J, `s10/hp_har.series`) на продлённом дневном ряду SBER: `build_k1(cut, src)` (протокол, раздел 11).

Дневной ряд — `daily.build_daily` с ЯВНЫМ `index_days` из `src` (решение 8: архивный `daily.IMOEX_PATH` не правится, `"auto"` не используется).
Строки — пары с t⁺ <= cut. Новые строки (cut > 2026-02-27) требуют замка: до него `src.bars` бросает SealedError.
Запись: `python3 -m models_v2.s17.fwd_k1 --write` — ТОЛЬКО архивные строки (cut = 2026-02-27) в data/fwd/models/k1.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from models_v2 import daily as D
from models_v2.s10 import hp_har as HH
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

OUT = ROOT / "data/fwd/models/k1.parquet"


def build_daily_sber(cut, src: Src) -> pd.DataFrame:
    """Канонический дневной ряд SBER (n_bars, rv, bv, j, seg, rv_w, rv_m) на склейке «архив + новые» до cut."""
    return D.build_daily(src.bars("SBER", cut), index_days=src.index_days(cut))[0]


def build_k1(cut, src: Src) -> pd.DataFrame:
    """K1: таблица `hp_har.series` (d, target_date, rv, sigma2, …) по парам с t⁺ <= cut."""
    return HH.series(build_daily_sber(cut, src))


def main(argv: list) -> int:
    k1 = build_k1(ARCHIVE_END, Src())
    print(f"k1 archive rows={len(k1)} d {k1['d'].min().date()}..{k1['d'].max().date()} target_max={k1['target_date'].max().date()}")
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(OUT) + ".tmp")
        k1.to_parquet(tmp)
        tmp.replace(OUT)
    if "--check" in argv:
        pd.testing.assert_frame_equal(pd.read_parquet(OUT), k1)
        print("k1.parquet == пересчёт: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
