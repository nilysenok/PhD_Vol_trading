"""E1 куба v7 (Ж-153): `EvaluatorV7(EvaluatorV6)` + реестр семей `v7` (interface._SETKI["v7"]).

S1–S6 в v7 — путь v6 побайтно: семья берётся из реестра v5 (get_family(sid, "v7") → v5), ключ кэша `_fkey` без суффикса.
S8–S12 (S7 не входит, Ж-153 п. 2) — фабрики `fabrika(ist7) -> fn(td, ind, grid) -> SignalSet`; источники — `Ist7`: b10 — 10-минутки
по тикерам (S10–S12, как после bars_agg.prep_10m), panel — панель сечения (S8, S9, v7_kross.Panel). Сигнал идёт в слой исполнения как у S1–S6
(Unit под sid-прокси S3: категория и таблицы RM — тренд; общие оси выхода и риска S10–S12 — куб S3, plan §1.1).
Контракт SignalSet, закреплённый здесь (замечания тестировщиков волны 2):
  S10–S12: exit — СОСТОЯНИЕ цепочки взятых сигналов, не событие каждого сигнала: флаг ставится только на выход позиции, которую сигнал
  открыл (сигналы внутри удержания пропущены — как в golaya_v7_isp.ispolnit), иначе выход пропущенного сигнала закрыл бы позицию нового;
  S8, S9: strength в прогреве = 0, как sig[:, :WARMUP] (контракт SignalSet S1–S6).
Полный evaluate_v2 для S8–S12 — после канона v7 (V7-1b) и оси k_okup (V7-3): здесь ValueError, не молчаливый v6.
Ключ кэша S8–S12 (Д5, №7): sid + sha источников (`Ist7.sha`) поверх сигнальных осей точки (tau, vhod, T).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fastbt import exec_e_minuderzh as minuderzh
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7_okup import OkupMixin
from fastbt.eval_v7_otn import OtnMixin
from fastbt.signals import v7_impuls, v7_kross
from fastbt.signals.interface import WARMUP, SignalSet, register_family
from fastbt.unit import Unit

SEMYI_V7 = ("S8", "S9", "S10", "S11", "S12")
PROXY = {s: "S3" for s in SEMYI_V7}


@dataclass
class Ist7:
    """Источники семей v7: b10 {тикер: 10-минутки} — S10–S12; panel (v7_kross.Panel) — S8, S9."""
    b10: dict | None = None
    panel: object | None = None
    otn: object | None = None          # eval_v7_otn.OtnIst: IMOEX и карта секторов оси otnositelno (S1–S4); в sha не входит — ключ ("otn", …)

    @property
    def sha(self) -> str:
        h = hashlib.sha256()
        for t in sorted(self.b10 or {}):
            h.update(t.encode() + pd.util.hash_pandas_object(self.b10[t], index=True).to_numpy().tobytes())
        pn = self.panel
        if pn is not None and hasattr(pn, "c"):
            h.update(pd.util.hash_pandas_object(pn.c, index=True).to_numpy().tobytes())
            h.update(pd.util.hash_pandas_object(pn.imoex, index=True).to_numpy().tobytes())
            for a in (pn.metki, pn.sigma, pn.ok):
                h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def _cepochka(sig: np.ndarray, tau: int, vhod: str) -> np.ndarray:
    """bool[T]: выход позиции, открытой взятым сигналом, — close дня i+τ. Взятые: d1 — сигнал дня i ≥ выхода прошлой позиции
    (вход i+1 > выхода), T — строго после (вход в T дня i, прошлая позиция закрывается close). Выход за концом ленты — нет."""
    n, ex, i = len(sig), np.zeros(len(sig), bool), 0
    while i < n:
        if sig[i] == 0:
            i += 1
            continue
        x = i + tau
        if x < n:
            ex[x] = True
        i = x if vhod == "d1" else x + 1
    return ex


def _kontrakt_impuls(fn):
    def g(td, ind, grid) -> SignalSet:
        ss = fn(td, ind, grid)
        ex = np.stack([_cepochka(ss.sig[k], p["tau"], p["vhod"]) for k, p in enumerate(grid)])
        return SignalSet(ss.sig, ex, ex.copy(), ss.strength, ss.params, split=True, extra=ss.extra)
    return g


def _obnulit_progrev(ss: SignalSet) -> SignalSet:
    st = ss.strength.copy()
    st[:, :WARMUP] = 0.0
    return SignalSet(ss.sig, ss.exit_l, ss.exit_s, st, ss.params, split=ss.split, extra=ss.extra)


def _fabrika_impuls(sid: str):
    def fabrika(ist7: Ist7):
        if not ist7.b10:
            raise ValueError(f"{sid}: в Ist7 нет b10 (10-минутки по тикерам)")
        return _kontrakt_impuls(v7_impuls.semya(sid, ist7.b10))
    return fabrika


def _fabrika_sech(sid: str):
    def fabrika(ist7: Ist7):
        if ist7.panel is None:
            raise ValueError(f"{sid}: в Ist7 нет panel (панель сечения)")
        fn = v7_kross.semya(sid, ist7.panel)
        return lambda td, ind, grid: _obnulit_progrev(fn(td, ind, grid))
    return fabrika


for _s in ("S10", "S11", "S12"):
    register_family(_s, _fabrika_impuls(_s), "v7")
for _s in ("S8", "S9"):
    register_family(_s, _fabrika_sech(_s), "v7")


class EvaluatorV7(OkupMixin, OtnMixin, EvaluatorV6):
    min_uder = 0                                                    # V7-4: умолчание на уровне класса (объекты без __init__ в тестах)

    def __init__(self, sid, inp, base_p, commission: float = 0.0004, fdr=None, ist=None, rol: str | None = None, ist7: Ist7 | None = None, min_uderzhanie: int = 0):
        super().__init__(PROXY.get(sid, sid), inp, base_p, commission, fdr, ist, rol)
        self.sid, self.proxy, self.ist7 = sid, PROXY.get(sid, sid), ist7 or Ist7()
        self.min_uder = minuderzh.dney(sid, min_uderzhanie)         # V7-4: константа куба, только контртренд; 0 — путь v6

    def _k7(self) -> tuple:
        return ((("v7", self.sid, self.ist7.sha),) if self.sid in SEMYI_V7 else ()) + self._otn_key() + self._okup_key() + (((minuderzh.KLYUCH, self.min_uder),) if self.min_uder else ())

    def _params(self, *a):
        p = super()._params(*a)
        return replace(p, min_uderzhanie=self.min_uder) if self.min_uder else p

    @property
    def _fkey(self) -> tuple:
        return self._fk + self._k7()

    @_fkey.setter
    def _fkey(self, v: tuple) -> None:
        self._fk = v

    def _signals5(self, sp: dict) -> dict:
        if self.sid not in SEMYI_V7:
            return super()._signals5(sp)
        from fastbt.signals.interface import get_family
        fam = get_family(self.sid, setka="v7")(self.ist7)
        return {t: Unit(t, self.proxy, td, ind, fam(td, ind, [dict(sp)])) for t, (td, ind) in self.sk._td.items()}

    def evaluate_v2(self, pt: dict, period: str, vozm: dict | None = None):
        if self.sid in SEMYI_V7:
            raise ValueError(f"{self.sid}: полный путь точки S8–S12 — после канона v7 (V7-1b) и оси k_okup (V7-3); здесь только сигнал")
        return super().evaluate_v2(self._otn_razbor(pt), period, vozm)
