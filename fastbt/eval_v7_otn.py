"""Л2.3 (Ж-146 п. 2, Ж-153): ось `otnositelno ∈ {net, sektor, imoex}` в EvaluatorV7 для S1–S4 (миксин; ряд — fastbt.otnositelno_v7).

Сигнал точки строится по относительному ряду (доходность тикера за вычетом равновзвешенного сектора из наших тикеров без него / IMOEX, дневные
r[d] = close[d]/close[d−1] − 1, решение дня d видит ≤ d, вход d+1 — Д2); исполнение, уровни, комиссия — по реальным ценам тикера (Unit строится на
реальных td, подменяется только вход семьи). `net` — путь v6 побайтно (оси в точке нет, ключ кэша без добавки).
Правило двойного фона (#909, canon_v7.dvojnoj_fon): otnositelno ≠ net вместе с rel_signal ≠ syroy — ValueError, не тихий пересчёт.
Источники — `OtnIst` (в `Ist7.otn`): IMOEX (дневной ряд) и карта секторов (dict тикер→метка или CSV; None — docs/issledovaniya/2026-10-27_sektory.csv).
Ключ кэша (Д5): ("otn", режим, sha источника) поверх ключа v6/v7. Данных не читает, кроме карты секторов.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import otnositelno_v7 as ot

SEMYI_OTN = ("S1", "S2", "S3", "S4")
SYROY = "syroy"


@dataclass
class OtnIst:
    """imoex — дневной ряд IMOEX (≤ 2021 на разработке); karta — {тикер: метка сектора}, путь к CSV или None (карта проекта)."""
    imoex: pd.Series
    karta: dict | Path | None = None

    def metki(self, dates: pd.DatetimeIndex, tickers: list) -> np.ndarray:
        if isinstance(self.karta, dict):
            return np.array([[self.karta.get(t, "") for t in tickers]] * len(dates), dtype=object)
        return ot.metki_iz_csv(self.karta or ot.KARTA_CSV, dates, tickers)

    @property
    def sha(self) -> str:
        h = hashlib.sha256(pd.util.hash_pandas_object(self.imoex, index=True).to_numpy().tobytes())
        if isinstance(self.karta, dict):
            h.update(repr(sorted(self.karta.items())).encode())
        else:
            h.update(Path(self.karta or ot.KARTA_CSV).read_bytes())
        return h.hexdigest()


class OtnMixin:
    otn = "net"

    def _otn_razbor(self, pt: dict) -> dict:
        """Выделяет ось из точки (в v6 её нет — оценщик v6 получает точку без неё); проверки: S1–S4, источник, двойной фон."""
        r = pt.get("otnositelno", "net")
        ot._proverit(r)
        if r != "net":
            if self.sid not in SEMYI_OTN:
                raise ValueError(f"{self.sid}: ось otnositelno — только S1–S4")
            if pt.get("rel_signal", SYROY) != SYROY:
                raise ValueError(f"{self.sid}: otnositelno={r} вместе с rel_signal={pt['rel_signal']!r} — двойной фон (канон v7), rel_signal должна быть {SYROY}")
            if getattr(self.ist7, "otn", None) is None:
                raise ValueError(f"{self.sid}: ось otnositelno включена, а в Ist7 нет otn (IMOEX и карта секторов)")
        self.otn = r
        return {k: v for k, v in pt.items() if k != "otnositelno"}

    def _otn_key(self) -> tuple:
        return (("otn", self.otn, self.ist7.otn.sha),) if self.otn != "net" else ()

    def _otn_fon(self) -> tuple:
        """(fon, flag) по панели закрытий всех тикеров оценщика; фон дня d — доходность за день d (≤ d), кэш на режим."""
        kesh = self.__dict__.setdefault("_otn_kesh", {})
        if self.otn not in kesh:
            o = self.ist7.otn
            c = pd.DataFrame({t: pd.Series(td.close, index=pd.DatetimeIndex(td.dates)) for t, (td, _) in sorted(self.sk._td.items())})
            kesh[self.otn] = ot.fon_ryad(c, o.imoex, o.metki(c.index, list(c.columns)), self.otn)
        return kesh[self.otn]

    def _sig_src(self, fam, ts, is_, p):
        if self.otn != "net":
            fam = ot.semya(fam, self.otn, *self._otn_fon())
        return super()._sig_src(fam, ts, is_, p)
