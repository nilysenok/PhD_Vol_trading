"""Интерфейс семейства сигналов fastbt (контракт между ядром и семействами S1..S6).

Семейство = модуль `fastbt/signals/s<N>.py` (или общий на два), который регистрирует
объект `Family` через `register(...)`. Ядро вызывает только то, что описано здесь.

Источник истины семантики: docs/fastbt_spec.md (код c22a4b3: V4 для S1–S4, V41 для S5/S6).

Входы семейства
  td  : TickerData — массивы ОДНОГО тикера, daily, float64, без NaN-заполнения
        (dates datetime64[ns], open/high/low/close/volume, log_ret вперёд).
  ind : dict — базовые индикаторы ядра, ровно как `compute_base` c22a4b3
        (fastbt/indicators.py): atr14, adx14, vol_sma20, vol_sma5, vol_regime,
        red_count5, green_count5, rh_{5,10,15,20}, rl_{5,10,15,20}, vf_pass,
        _volume, bb_width, bw_p30, sma20_slope10. Семейство НЕ меняет ind.
  grid: list[dict] — сигнальные параметры в порядке expand_grid (сортировка ключей,
        последний ключ быстрее всех) — ровно как SIGNAL_GRIDS / S5_SIG_DAILY / S6D_SIG.

Выход семейства — SignalSet (K конфигураций × T баров):
  sig      int8[K,T]   +1/-1/0, sig[:, :WARMUP] = 0
  exit_l   bool[K,T]   сигнальный выход длинной позиции
  exit_s   bool[K,T]   сигнальный выход короткой (если в старом коде один exit_arr —
                       exit_s = exit_l, и split=False)
  split    bool        True, если старый код передавал кортеж (exit_l, exit_s)
  strength f64[K,T]    сила сигнала для слоя исполнения (V4 dispatch_strength / V41)
  extra    dict[str, f64[K,T]]  уровни для «уровневых» RM, только S5/S6:
           S5: "P","S1","S2","R1","R2" (уровни pivot конфигурации, как в V41);
           S6: "vwap","dev" (окно конфигурации, как precompute_vwap_cache_ext).
           Остальные семейства: {}.
  params   list[dict]  = grid (тот же порядок)

Кэш: ядро кэширует SignalSet на диск по ключу cache_key(family, td, grid) —
  sha1(имя семейства, family.code_version, sha1 исходника модуля семейства,
  sha1 входных массивов тикера, json(grid)). Меняешь логику — меняется хэш исходника,
  кэш инвалидируется сам; code_version — ручной «рубильник».

Эквивалентность: family.legacy(L, td, params) вызывает СТАРУЮ функцию сигналов
  c22a4b3 (L = load_legacy(): модули sw, v4, v41, s5r, s6r) для ОДНОЙ конфигурации и
  возвращает (sig, exit_l, exit_s, strength) — как их получил бы старый конвейер.
  check_family(family, td) сравнивает побитово (сигналы, выходы) и до 1e-12 (сила).
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional

import numpy as np

WARMUP = 200


@dataclass
class TickerData:
    ticker: str
    dates: np.ndarray          # datetime64[ns], по возрастанию
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    log_ret: np.ndarray        # log(close[t+1]/max(close[t],1e-12)), последний = 0
    sigma: Dict[str, np.ndarray] = field(default_factory=dict)  # h1/h5/h22, NaN до 2017

    @property
    def n(self) -> int:
        return len(self.close)

    def digest(self) -> str:
        h = hashlib.sha1(self.ticker.encode())
        for a in (self.dates.astype("int64"), self.open, self.high, self.low,
                  self.close, self.volume):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


@dataclass
class SignalSet:
    sig: np.ndarray
    exit_l: np.ndarray
    exit_s: np.ndarray
    strength: np.ndarray
    params: List[dict]
    split: bool = False
    extra: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.sig.shape[0]

    def exit_info(self, i: int):
        """exit_info одной конфигурации в форме старого кода."""
        return (self.exit_l[i], self.exit_s[i]) if self.split else self.exit_l[i]


@dataclass
class Family:
    sid: str                    # "S1".."S6"
    category: str               # "Contrarian" | "Trend" | "Range"
    grid: Dict[str, list]       # сигнальная сетка daily (как в коде c22a4b3)
    generate: Callable[[TickerData, dict, List[dict]], SignalSet]
    legacy: Callable[[object, TickerData, dict], tuple]
    code_version: str = "1"
    module: Optional[object] = None

    def params(self) -> List[dict]:
        return expand_grid(self.grid)

    def source_hash(self) -> str:
        src = inspect.getsource(self.module) if self.module is not None else ""
        return hashlib.sha1(src.encode()).hexdigest()


_REGISTRY: Dict[str, Family] = {}


def register(fam: Family) -> Family:
    _REGISTRY[fam.sid] = fam
    return fam


_REGISTRY_V5: Dict[str, object] = {}   # сетка v5 (Ж-127): семьи регистрируются отдельно от v4
_REGISTRY_V7: Dict[str, object] = {}   # сетка v7 (Ж-153): S8–S12 — фабрики `fabrika(ist7) -> fn` (eval_v7); S1–S6 — из v5; S7 нет
_SETKI = {"v4": _REGISTRY, "v5": _REGISTRY_V5, "v7": _REGISTRY_V7}


def register_family(sid: str, fn, setka: str = "v4"):
    """Зарегистрировать семью сетки `setka` ("v4" — как `register`, но без Family-обёртки; "v5" — отдельный реестр)."""
    if setka not in _SETKI:
        raise KeyError(f"setka: {setka!r}")
    _SETKI[setka][sid] = fn
    return fn


def get_family(sid: str, setka: str = "v4"):
    if setka == "v7":                       # S8–S12 регистрирует eval_v7; S1–S6 в v7 — семьи v5 (путь v6 побайтно); KeyError — S7 и прочее
        import importlib
        importlib.import_module("fastbt.eval_v7")
        return _REGISTRY_V7[sid] if sid in _REGISTRY_V7 else _REGISTRY_V5[sid]
    if setka != "v4":
        if setka not in _SETKI:
            raise KeyError(f"setka: {setka!r}")
        return _SETKI[setka][sid]   # v5 импортируется владельцем модуля заранее; KeyError — если не зарегистрирована
    if sid not in _REGISTRY:
        import importlib
        importlib.import_module(f"fastbt.signals.{sid.lower()}")
    return _REGISTRY[sid]


def families() -> Dict[str, Family]:
    return dict(_REGISTRY)


def expand_grid(d: Dict[str, list]) -> List[dict]:
    """Ровно SW.expand_grid: декартово произведение по отсортированным ключам."""
    if not d:
        return [{}]
    keys = sorted(d.keys())
    return [dict(zip(keys, c)) for c in product(*[d[k] for k in keys])]


def params_json(grid: List[dict]) -> str:
    return json.dumps(grid, sort_keys=True, default=str)


def cache_key(fam: Family, td: TickerData, grid: List[dict]) -> str:
    h = hashlib.sha1()
    for part in (fam.sid, fam.code_version, fam.source_hash(), td.digest(),
                 params_json(grid)):
        h.update(part.encode())
    return h.hexdigest()


def check_family(fam: Family, td: TickerData, ind: dict, grid: Optional[List[dict]] = None,
                 tol: float = 1e-12) -> dict:
    """Сравнить сигналы семейства со старым кодом c22a4b3 по всем конфигурациям grid."""
    from fastbt.legacy import load_legacy
    L = load_legacy()
    grid = grid if grid is not None else fam.params()
    ss = fam.generate(td, ind, grid)
    bad = []
    for i, p in enumerate(grid):
        sig, el, es, st = fam.legacy(L, td, p)
        es = el if es is None else es
        ok = (np.array_equal(ss.sig[i], np.asarray(sig, dtype=np.int8))
              and np.array_equal(ss.exit_l[i], np.asarray(el, dtype=bool))
              and np.array_equal(ss.exit_s[i], np.asarray(es, dtype=bool))
              and np.allclose(ss.strength[i], st, rtol=0, atol=tol, equal_nan=True))
        if not ok:
            bad.append(p)
    return {"sid": fam.sid, "ticker": td.ticker, "n": len(grid), "bad": bad}
