"""Вход оценщика (Ж-031, Э1-а): Spec из yaml и кэш сигналов.

Кэш сигналов отдельно от PnL: ключ = хэш (объект, параметры сигнала, ряд, код).
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from fastbt import sweep_stat as ss


@dataclass
class Spec:
    obj: dict                      # {kind: Sk, sid: S1, tickers: [...]}; сборка A и C/E/B/D — позже
    base: dict                     # исходные значения параметров
    comm_pct: tuple = (0.035, 0.04)  # % на сторону; первая — основная
    periods: ss.Periods = field(default_factory=ss.Periods)
    out: str | None = None         # папка: ledger.json (пробы), validation_flag.json
    reopen_note: str | None = None  # запись об изменении для повторного открытия validation

    @classmethod
    def from_yaml(cls, path_or_dict):
        import yaml
        d = path_or_dict if isinstance(path_or_dict, dict) else yaml.safe_load(Path(path_or_dict).read_text("utf-8"))
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "periods" in d:
            d["periods"] = ss.Periods(**{k: tuple(v) for k, v in d["periods"].items()})
        if "comm_pct" in d:
            d["comm_pct"] = tuple(d["comm_pct"])
        return cls(**d)


class SignalCache:
    def __init__(self, d=None):
        self.mem, self.dir, self.hits, self.misses = {}, Path(d) if d else None, 0, 0

    @staticmethod
    def key(obj, sig_params, series_digest, code_id) -> str:
        raw = json.dumps([obj, sorted(sig_params.items()), series_digest, code_id], default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def get(self, k, make):
        if k in self.mem:
            self.hits += 1
            return self.mem[k]
        f = self.dir / f"{k}.pkl" if self.dir else None
        if f is not None and f.exists():
            self.mem[k] = pickle.loads(f.read_bytes())
            self.hits += 1
            return self.mem[k]
        self.misses += 1
        self.mem[k] = make()
        if f is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(self.mem[k]))
        return self.mem[k]


def pkey(p: dict) -> str:
    return json.dumps(sorted(p.items()), default=str)
