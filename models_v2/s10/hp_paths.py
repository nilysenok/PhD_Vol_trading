"""Ж-215 / Ж-217, лист P0: пути, вселенная (R1), справочник секторов. Программа пишет только OUT и REP; всё остальное — чтение."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BARS_DIR = ROOT / "data/iss_check/bars_10m_full"
IMOEX = ROOT / "data/iss_check/index_10m/IMOEX.parquet"
COMP = ROOT / "data/regen/xgb_lgbm_full_v1/e/components.parquet"
M10_SBER = ROOT / "data/regen/harj_hybrid_v2/m10_daily.parquet"
OUT = ROOT / "data/regen/harj_panel_hybrid"
REP = ROOT / "docs/otchety/harj_panel_hybrid"

MAIN17 = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX", "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"]
E11 = ["GCHE", "MRKC", "MRKP", "MSNG", "MSRS", "PIKK", "RGSS", "SELG", "TGKA", "TRMK", "UTAR"]
M16 = [t for t in MAIN17 if t != "SBER"]
TICKERS = sorted(MAIN17 + E11)
FIRST_TARGET = "2017-01-03"
MIN_PAIRS = 252
GAP_DAYS = 5          # R2: разрыв t -> t⁺ > 5 календарных дней

# Справочник сектора — вручную, по общеизвестной отраслевой принадлежности эмитента (без сверки с внешним источником; не признак, R1).
SECTOR = {"LKOH": "Нефть и газ", "NVTK": "Нефть и газ", "TATN": "Нефть и газ",
          "SBER": "Финансы", "VTBR": "Финансы", "MOEX": "Финансы", "RGSS": "Финансы",
          "ALRS": "Металлы и добыча", "MTLR": "Металлы и добыча", "SELG": "Металлы и добыча", "TRMK": "Металлы и добыча",
          "HYDR": "Генерация", "IRAO": "Генерация", "MSNG": "Генерация", "OGKB": "Генерация", "TGKA": "Генерация",
          "MRKC": "Электросети", "MRKP": "Электросети", "MSRS": "Электросети",
          "MTSS": "Телеком", "RTKM": "Телеком", "MGNT": "Потребительский", "GCHE": "Потребительский",
          "LSRG": "Строительство", "PIKK": "Строительство", "AFLT": "Транспорт", "UTAR": "Транспорт", "PHOR": "Химия"}


def group(t: str) -> str:
    return "SBER" if t == "SBER" else ("M16" if t in M16 else "E11")


def bars_path(t: str) -> Path:
    return IMOEX if t == "IMOEX" else BARS_DIR / (t + ".parquet")


def sha16(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_csv(df, name: str, note: str) -> Path:
    REP.mkdir(parents=True, exist_ok=True)
    p = REP / name
    with open(p, "w", encoding="utf-8") as f:
        f.write("# DEVELOPMENT; Ж-215/Ж-217 P0; %s\n" % note)
        df.to_csv(f, index=False, lineterminator="\n")
    return p
