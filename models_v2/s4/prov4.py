"""Шапка таблиц Ж-200: «до аудита», sha кода s4, этапа 1 и входов (SBER, RVI, 16 бумаг, при наличии — IMOEX ISS)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from models_v2 import prov

S4 = Path(__file__).resolve().parent
PAPERS = ("AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX", "MTLR", "MTSS", "NVTK", "OGKB", "PHOR",
          "RTKM", "TATN", "VTBR")
BARS_DIR = prov.ROOT / "data/iss_check/bars_10m_full"
OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"


def code_sha4() -> str:
    h = hashlib.sha256()
    for f in sorted(S4.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=None)
def sha_papers() -> str:
    h = hashlib.sha256()
    for n in PAPERS:
        h.update(n.encode() + prov.sha_file(BARS_DIR / (n + ".parquet")).encode())
    return h.hexdigest()[:16]


def header4(extra: str = "", imoex: str = "") -> str:
    return ("# разработка; до аудита; оценка 2017–02.2026 по Ж-200 (2022+ — просмотр контроля гл. 3 по SBER); "
            "код models_v2/s4 sha256[:16]=%s, этап 1 sha256[:16]=%s; sha входов: bars=%s rvi=%s papers16=%s imoex=%s %s\n"
            % (code_sha4(), prov.code_sha(), prov.sha_file(prov.BARS)[:16], prov.sha_file(prov.RVI)[:16],
               sha_papers(), imoex or "нет", extra))


def write_csv(df, path, extra: str = "", imoex: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header4(extra, imoex))
        df.to_csv(f, index=False, lineterminator="\n")
