"""Ж-214 (HAR-J Hybrid V2, SBER h=1): пути. Программа пишет только OUT и REP; всё остальное — чтение."""
from __future__ import annotations

from models_v2.s6 import paths6

ROOT = paths6.ROOT
OUT = ROOT / "data/regen/harj_hybrid_v2"
REP = ROOT / "docs/otchety/harj_hybrid_v2"
PRIOR_REP = ROOT / "docs/otchety/hybrid_struct_v1"
SBER = ROOT / "data/iss_check/bars_10m_full/SBER.parquet"
IMOEX = ROOT / "data/iss_check/index_10m/IMOEX.parquet"
COMP = ROOT / "data/regen/xgb_lgbm_full_v1/e/components.parquet"
M10 = OUT / "m10_daily.parquet"

STATUS = "HISTORICAL WALK-FORWARD 2017–02.2026 (2271 дата, просмотрено; не независимый OOS); DEVELOPMENT; Ж-214; первая цифра — потолок; «лет лучше» = число лет цели 2017–2026 (10; 2026 неполный, 38 дат) со средним QLIKE строки < контроля, год по target_date"
FEB24 = "2022-02-24"
TOL_ID = 1e-12   # допуск тождеств разложений: |Σ частей − RV| / RV (суммы в разном порядке, double)


def write_csv(df, name: str, note: str, sub: str = "") -> None:
    d = REP / sub if sub else REP
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "w", encoding="utf-8") as f:
        f.write("# %s; %s\n" % (STATUS, note))
        df.to_csv(f, index=False, lineterminator="\n")
