"""Базы сравнения validation A′ (толкование Ж-039 п. 1, 27.10 20:30): боевая A через `s9.Engine.port`, а не S1–S6 в центрах сеток.

База A′ — A (процедура A: пара тикер × семья, топ-10, голосование ≥ 6/10; позиции `Engine.A_row`, равные веса 1/cnt).
База Sk — «A только Sk» (`shag0.keep_sids`), как в паспорте шага 0. В заморозке база — точная ссылка: функция, конфиг, sha конфига.
Данные читает только под замком (`evaluate`); validation до `open_validation` недоступна.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SIDS = tuple(f"S{i}" for i in range(1, 7))
FUNKTSIYA = "fastbt.sweep_val_baza.S9Baza"
YEARS = {"setup": (2018,), "control": (2019,), "train": (2018, 2019), "validation": (2020, 2021)}
COMM_PCT = 0.04
_ENGINE: dict = {}


def konfig(bid: str) -> dict:
    """Конфиг базы: «A» — вся A; «Sk» — A только на семье Sk."""
    if bid != "A" and bid not in SIDS:
        raise ValueError(f"база {bid}: ожидается A или S1…S6")
    return dict(protsedura="A: пара тикер × семья, топ-10, голосование ≥ 6/10", dvizhok="s9.Engine.port", sigma_set="v2",
                scope="dev", comm_pct=COMM_PCT, semya=None if bid == "A" else bid, vesa="1/cnt (все пары A)")


def sha_konfiga(k: dict) -> str:
    return hashlib.sha256(json.dumps(k, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def baza_ref(bid: str) -> dict:
    k = konfig(bid)
    return dict(id=bid, funktsiya=FUNKTSIYA, konfig=k, sha_konfiga=sha_konfiga(k))


def ref_ok(ref: dict, kind: str, sid: str | None) -> bool:
    """Ссылка корректна: sha конфига пересчитывается, функция — s9-база, id = A для сборок и семья для стратегии."""
    want = "A" if kind == "A" else sid
    return bool(ref.get("id") == want and ref.get("funktsiya") == FUNKTSIYA and ref.get("konfig") == konfig(want)
                and ref.get("sha_konfiga") == sha_konfiga(ref["konfig"]))


def baza_ok(m: dict, bazy: dict) -> bool:
    """Кандидат m заморозки: ссылка на базу есть, корректна и равна записи в bazy."""
    return "baza" in m and m["baza"] == bazy.get(m["baza"].get("id")) and ref_ok(m["baza"], m["kind"], m.get("sid"))


def s9_engine():
    """Движок разработки s9 (σ v2, SHA сверены) — один на процесс."""
    if "e" not in _ENGINE:
        from fastbt import s9
        s9.set_scope("dev")
        s9.use_v2()
        _ENGINE["e"] = s9.Engine(s9.load_cache())
    return _ENGINE["e"]


class S9Baza:
    """Интерфейс `Sweep.evaluate` для базы: params игнорируются (база — не сборка). eng — движок (в тестах синтетический)."""

    def __init__(self, ref: dict, eng=None):
        self.ref, self._eng, self._val_open = ref, eng, False

    def open_validation(self) -> dict:
        self._val_open = True
        return {}

    def n_mult(self) -> int:
        return 1

    def frame(self, comm_pct: float | None = None):
        from fastbt import shag0
        eng = self._eng if self._eng is not None else s9_engine()
        pos = np.asarray(eng.A_row, float)
        if self.ref["konfig"]["semya"]:
            pos = shag0.keep_sids(pos, shag0.sid_row(eng), [self.ref["konfig"]["semya"]])
        return eng.port(pos, comm=(COMM_PCT if comm_pct is None else comm_pct) / 100.0)

    def evaluate(self, params: dict | None = None, period: str = "train", comm_pct: float | None = None) -> dict:
        if period == "validation" and not self._val_open:
            raise PermissionError("validation: база открывается вместе со всеми (open_validation)")
        from fastbt import sweep_stat as ss
        years, df = YEARS[period], self.frame(comm_pct)
        row, q = ss.metrics_row(df, years), ss.year_slice(df, years)
        row["excess_d"] = [float(x) for x in q["excess"].values]
        row["daty"] = [d.strftime("%Y-%m-%d") for d in q.index]
        return {**row, "period": period, "comm_pct": COMM_PCT if comm_pct is None else comm_pct, "baza": self.ref["id"]}
