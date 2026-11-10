"""Л1: источник «архив + новые» с обрезкой по моменту выпуска и замком (протокол HARJ_HYBRID_SBER_FWD, раздел 11).

`cut` — день выпуска прогноза d: берутся бары с begin <= d 18:40 (вечерняя сессия и дни после d не входят).
До замка (`OPENED_HARJ_FWD.json`, создаёт лист Л6) новым файлам отдаётся ТОЛЬКО столбец begin: `bars` и `sealed_rv` без замка —
исключение SealedError, даже если запрошенный cut лежит в новом периоде. Архивные строки (begin <= 2026-02-27) отдаются всегда.
Файлы только читаются; новые файлы без замка читаются с columns=["begin"].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_END = pd.Timestamp("2026-02-27")      # последний день архива
LAST_SESSION = pd.Timestamp("2026-11-10")     # последняя полностью завершённая сессия (Р5)
CUT_TIME = pd.Timedelta(hours=18, minutes=40)  # последний бар основной сессии стартует в 18:40


class SealedError(RuntimeError):
    """Чтение цен, объёмов, value новых файлов или запечатанной RV до замка."""


@dataclass(frozen=True)
class Paths:
    bars_dir: Path = ROOT / "data/iss_check/bars_10m_full"
    imoex_archive: Path = ROOT / "data/iss_check/index_10m/IMOEX.parquet"
    new_sber: Path = ROOT / "data/fwd/sber_m10/SBER_m10_fwd.parquet"
    new_mkt_dir: Path = ROOT / "data/fwd/mkt_m10"
    sealed_rv: Path = ROOT / "data/fwd/sber_m10/rv_fwd_SEALED.parquet"
    lock: Path = ROOT / "data/fwd/models/OPENED_HARJ_FWD.json"


def limit_of(cut) -> pd.Timestamp:
    """Момент выпуска: день cut 18:40 (включительно)."""
    return pd.Timestamp(cut).normalize() + CUT_TIME


class Src:
    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()

    def archive_path(self, inst: str) -> Path:
        return self.p.imoex_archive if inst == "IMOEX" else self.p.bars_dir / f"{inst}.parquet"

    def new_path(self, inst: str) -> Path:
        return self.p.new_sber if inst == "SBER" else self.p.new_mkt_dir / f"{inst}_m10_fwd.parquet"

    def is_open(self) -> bool:
        return self.p.lock.exists()

    def _read(self, path: Path, limit: pd.Timestamp, columns=None) -> pd.DataFrame:
        df = pd.read_parquet(path, columns=columns)
        return df[df["begin"] <= limit].reset_index(drop=True)

    def begins(self, inst: str, cut) -> pd.DataFrame:
        """Столбец begin (архив + новые) до cut 18:40. Новые файлы — только begin; замок не нужен."""
        lim = limit_of(cut)
        a = self._read(self.archive_path(inst), lim, ["begin"])
        n = self._new_begins(inst, lim)
        if len(a) and len(n) and not n["begin"].min() > a["begin"].max():
            raise ValueError(f"{inst}: новые бары пересекаются с архивом (begin {n['begin'].min()} <= {a['begin'].max()})")
        return pd.concat([a, n], ignore_index=True)

    def _new_begins(self, inst: str, lim: pd.Timestamp) -> pd.DataFrame:
        path = self.new_path(inst)
        if not path.exists():
            return pd.DataFrame({"begin": pd.Series([], dtype="datetime64[ms]")})
        return self._read(path, lim, ["begin"])

    def bars(self, inst: str, cut) -> pd.DataFrame:
        """Бары (архив + новые) до cut 18:40. Если среди них есть строки новых файлов, а замка нет — SealedError."""
        lim = limit_of(cut)
        a = self._read(self.archive_path(inst), lim)
        if len(self._new_begins(inst, lim)) == 0:
            return a
        if not self.is_open():
            raise SealedError(f"{inst}: цены новых файлов до замка не читаются (cut={pd.Timestamp(cut).date()})")
        return pd.concat([a, self._read(self.new_path(inst), lim)], ignore_index=True)

    def index_days(self, cut) -> set:
        """Даты, где у IMOEX есть бар основной сессии (begin 10:00…18:40), день <= cut. Только begin."""
        from models_v2 import daily as D
        return set(D.main_session(self.begins("IMOEX", cut))["begin"].dt.normalize())

    def sealed_rv(self) -> pd.DataFrame:
        if not self.is_open():
            raise SealedError("rv_fwd_SEALED.parquet до замка не читается")
        return pd.read_parquet(self.p.sealed_rv)
