"""Оценщик движка перебора (Ж-031, Э1-а): evaluate(params, period) -> одна строка стандартной таблицы.

Контракт драйверов (`fastbt/sweep_drivers.py`): dict минимум с `sharpe` и `by_year`; здесь же mSR, Сортино, MDD,
оборот, доля дней, сделок/год, T, skew, kurt (для DSR), N проб накопленное, comm_pct.
Режимы (OAT, факторный, Моррис, Optuna), цель, плато, DSR, PBO — в `sweep_drivers.py`; здесь их нет.
Validation: период `validation` доступен только после `open_validation()`; он пишет флаг «validation открыта: <время>»,
повторное открытие — только с `reopen_note`, и N в DSR удваивается (`n_mult`). Validation в пробы (ledger) не пишется.
Evaluator: `sig_names`, `series_digest()`, `code_id()`, `signals(sig_params)`, `pnl(sigs, params, comm, end)`
-> DataFrame [excess, exposure, turnover, sides] по дням < end.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from fastbt import sweep_stat as ss
from fastbt.sweep_spec import SignalCache, Spec, pkey


class Sweep:
    def __init__(self, spec: Spec, evaluator, cache_dir=None):
        self.spec, self.ev, self.cache = spec, evaluator, SignalCache(cache_dir)
        self.out = Path(spec.out) if spec.out else None
        self.comm = [c / 100.0 for c in spec.comm_pct]
        self.ledger, self._mem, self._lock, self._val_open = {}, {}, threading.Lock(), False
        if self.out and (self.out / "ledger.json").exists():
            self.ledger = json.loads((self.out / "ledger.json").read_text("utf-8"))

    def _df(self, p: dict, comm: float, years) -> pd.DataFrame:
        years = tuple(years)
        if any(y in self.spec.periods.validation for y in years) and not self._val_open:
            raise PermissionError("validation закрыта: годы 2020–2021 только после open_validation()")
        if max(years) > ss.END_ALLOWED:
            raise PermissionError("данные ≥ 2022 запрещены")
        mk = (pkey(p), comm, years)
        if mk not in self._mem:
            sp = {k: v for k, v in p.items() if k in self.ev.sig_names}
            k = SignalCache.key(self.spec.obj, sp, self.ev.series_digest(), self.ev.code_id())
            sigs = self.cache.get(k, lambda: self.ev.signals(sp))
            end = pd.Timestamp(f"{max(years) + 1}-01-01")
            self._mem[mk] = ss.year_slice(self.ev.pnl(sigs, p, comm, end), years)
        return self._mem[mk]

    def evaluate(self, params: dict, period: str = "train", comm_pct: float | None = None) -> dict:
        """Строка таблицы для точки: base + params, период setup|control|train|validation."""
        years = self.spec.periods.years(period)
        comm = self.comm[0] if comm_pct is None else comm_pct / 100.0
        p = {**self.spec.base, **params}
        row = ss.metrics_row(self._df(p, comm, years), years)
        if period == "setup" and comm == self.comm[0]:           # проба = точка на периоде отбора (настройка 2018) при комиссии отбора (первая в comm_pct)
            with self._lock:
                self.ledger[pkey(p)] = row["sr_d"]
        mult = self.n_mult() if period == "validation" else 1
        return {**p, "comm_pct": comm * 100, "period": period, **row, "N": len(self.trials_sr()) * mult, "n_mult": mult}

    def trials_sr(self) -> list:
        """Дневные SR накопленных проб (настройка/контроль/train) при комиссии отбора — для sweep_drivers.dsr.
        Пересчёт при другой комиссии (справочный) пробой не считается и SR пробы не перезаписывает."""
        return list(self.ledger.values())

    def _opens(self) -> list:
        f = self.out / "validation_flag.json" if self.out else None
        return json.loads(f.read_text("utf-8"))["opens"] if f and f.exists() else []

    def n_mult(self) -> int:
        return 2 ** max(len(self._opens()) - 1, 0)

    def open_validation(self) -> dict:
        if self.out is None:
            raise ValueError("validation: нужен out (флаг открытия пишется на диск)")
        opens = self._opens()
        if opens and not self.spec.reopen_note:
            raise PermissionError("validation уже открыта; повторно — только с записанным изменением (reopen_note)")
        t = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        opens.append(dict(text=f"validation открыта: {t}", note=self.spec.reopen_note, n_probes=len(self.ledger)))
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "validation_flag.json").write_text(
            json.dumps(dict(opens=opens, n_mult=2 ** (len(opens) - 1)), ensure_ascii=False), "utf-8")
        self._val_open = True
        return dict(opens=opens, n_mult=self.n_mult())

    def save(self):
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            (self.out / "ledger.json").write_text(json.dumps(self.ledger), "utf-8")


def make_evaluator(spec: Spec):
    kind = spec.obj.get("kind")
    if kind == "Sk":
        from fastbt.sweep_sk import SkEvaluator
        return SkEvaluator(spec.obj["sid"], spec.obj.get("tickers"))
    raise NotImplementedError(f"объект {kind!r}: в v1 только Sk (S1–S6); сборка A и семьи C/E/B/D — позже")
