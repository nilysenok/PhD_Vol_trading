"""Ж-212: пути и статус программы структурной переработки гибридов HAR-J × XGBoost × LightGBM (SBER h=1).
Только эти каталоги пишет программа; всё остальное — чтение."""
from __future__ import annotations

from models_v2.s6 import paths6

ROOT = paths6.ROOT
OUT = ROOT / "data/regen/hybrid_struct_v1"
REP = ROOT / "docs/otchety/hybrid_struct_v1"
PRIOR_REP = ROOT / "docs/otchety/hybrid_final_v1"
COMP = ROOT / "data/regen/xgb_lgbm_full_v1/e/components.parquet"
SERIES_A = ROOT / "data/regen/xgb_lgbm_full_v1/e/series_a.parquet"
IMOEX = ROOT / "data/iss_check/index_10m/IMOEX.parquet"

STATUS = "HISTORICAL WALK-FORWARD 2017–02.2026 (2271 дата, просмотрено; не независимый OOS); DEVELOPMENT; Ж-212; первая цифра — потолок"
FEB24 = "2022-02-24"
MIN_CLOSED = 252


def write_csv(df, name: str, note: str) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / name, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")
