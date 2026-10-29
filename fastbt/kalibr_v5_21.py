"""К-1 (волна 17, `docs/otchety/2026-10-29_gotovnost_servera.md` §3, вариант (б)): фазы 1–2 калибратора v5 на ledger 2.1 (суммы
сегментов по кварталам) вместо рядов 2.0. Данных не читает. Определения те же, что в `kalibr_v5`: окно [начало; 31.12(t−1)], ни одного бара года t;
фаза 1 — ворота v4 → top-50 по Шарпу окна; фаза 2 — соседи ±1 по осям с эффектом, плато, край (< MIN_NB) не берётся → top-20.
Окно — `stats.window` на сегментах 0…j (j = Q4 года t−1; Шарп ddof = 1·√252, Д7; выход на конце окна оплачен, № 6).
Фаза 3 и голос (позиции top-10) — лист К-2: здесь отдаются индексы и ключи членов.

↺ класс B: просадка окна — по `stats.window` (накопленная сумма сегмента 0 не поднимается до нуля), в 2.0 — с полом 0 (`kalibr_v5.metriki`);
различие только если equity с первых дней ниже нуля; из сумм не восстановить. Откат: фазы 1–2 по ledger 2.0.
↺ без |позиции| (poz_konec = false) экспозиция из сумм неизвестна — пул не строится (отказ), а не «0 %» с пустыми воротами."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from fastbt import grids, kalibr_v5 as K, ledger_v2 as lm, stats, sweep_plato as SP
from fastbt.approach_a import _desc
from fastbt.sweep_pts import MIN_NB


def segment_okna(t: int, god0: int, nseg: int) -> int:
    """Индекс последнего сегмента окна года t: Q4 года t−1. Окно, заходящее в год t, невозможно по построению."""
    j = (t - 1 - god0) * 4 + 3
    if j < 3 or j >= nseg:
        raise ValueError(f"окно года {t}: сегмент {j} вне 0…{nseg - 1} (начало {god0})")
    return j


def metriki_okna(sm: np.ndarray, nbars: np.ndarray, t: int, god0: int, kom: float) -> tuple:
    """(sh, mdd, экспозиция %) окна по строкам sm[..., 16, NF]."""
    sh, mdd, ex, _ = stats.window(np.asarray(sm, float), np.asarray(nbars, float), segment_okna(t, god0, len(nbars)), kom)
    return np.asarray(sh), np.asarray(mdd), np.asarray(ex)


@dataclass
class Pul21:
    """Точки одной стратегии: sm[P, тикеры, 16, NF] — суммы сегментов (`ledger_v2.summy`), nbars[16] — баров в кварталах."""
    sid: str
    osi: dict
    tochki: list
    sm: np.ndarray
    nbars: np.ndarray
    god0: int = 2016
    komissiya: float = 0.0
    dosschitat: Optional[Callable] = None     # pt -> sm[тикеры, 16, NF]; None — соседей вне ledger нет
    novye: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.sm.ndim != 4 or self.sm.shape[0] != len(self.tochki) or self.sm.shape[2:] != (len(self.nbars), stats.NF):
            raise ValueError("sm: [точки, тикеры, сегменты, NF]")
        self.idx, self.kol, self._sh = {}, {}, {}
        for i, pt in enumerate(self.tochki):
            self.idx.setdefault(K.klyuch(self.sid, pt), i)
        for a in self.osi:
            self.kol[a] = np.array([lm.tok(pt[a]) if a in pt else "" for pt in self.tochki])

    def n_proby(self) -> int:
        return len(self.idx) + len(self.novye)

    def okno(self, t: int) -> tuple:
        """(sh, mdd, ex) всех точек ledger: [P, тикеры]; считается один раз на год."""
        if t not in self._sh:
            self._sh[t] = metriki_okna(self.sm, self.nbars, t, self.god0, self.komissiya)
        return self._sh[t]

    def sh_soseda(self, kl: str, q: dict, k: int, t: int) -> Optional[float]:
        """Шарп окна тикера k у соседа: из ledger либо досчитанного (проба в N); нет — None."""
        if kl in self.idx:
            return float(self.okno(t)[0][self.idx[kl], k])
        if kl not in self.novye:
            if self.dosschitat is None:
                return None
            self.novye[kl] = np.asarray(self.dosschitat(q), float)
        return float(metriki_okna(self.novye[kl][None], self.nbars, t, self.god0, self.komissiya)[0][0, k])


def faza1(pul: Pul21, k: int, t: int) -> tuple:
    """(sh [P] тикера k, top-50 индексов после ворот; ворота не прошёл никто — top-10 по sh, как `kalibr_v5.faza1`)."""
    sh, mdd, ex = (a[:, k] for a in pul.okno(t))
    ok = (sh > 0) & (mdd > K.MDD_MIN) & (ex > K.EXP_MIN)
    vse = list(range(len(pul.tochki)))
    prohod = [i for i in vse if ok[i]] or _desc(vse, lambda i: sh[i])[:grids.ENS_K]
    return sh, _desc(prohod, lambda i: sh[i])[:K.TOP_F1]


def faza2(pul: Pul21, k: int, t: int, sh: np.ndarray, top50: list) -> list:
    """top-20 по плато: [{i, kl, sh, plato, n_nb}]; край (соседей < MIN_NB) не берётся."""
    osi, exp, out = K.osi_s_effektom(pul, sh), {"grid": pul.osi}, []
    fix = lambda q: lm.canon_v2(pul.sid, q)        # noqa: E731 — сосед по канону = точка ledger
    for i in top50:
        pt = pul.tochki[i]
        nb = [pul.sh_soseda(K.klyuch(pul.sid, q), q, k, t) for q in SP.neighbours(exp, fix(pt), osi, fix)]
        nb = [v for v in nb if v is not None and v == v]
        if len(nb) >= MIN_NB:
            out.append(dict(i=i, kl=K.klyuch(pul.sid, pt), sh=float(sh[i]), plato=SP.plateau_value(float(sh[i]), nb), n_nb=len(nb)))
    return _desc(out, lambda c: c["plato"])[:K.TOP_F2]


def god(pul: Pul21, t: int) -> dict:
    """Фазы 1–2 года t по всем тикерам: {god, okno, tikery: {k: {top20, chleny (top-10 по sh), plato} | None}, n_proby}."""
    tik = {}
    for k in range(pul.sm.shape[1]):
        sh, top50 = faza1(pul, k, t)
        top20 = faza2(pul, k, t, sh, top50)
        chleny = _desc(top20, lambda c: c["sh"])[:grids.ENS_K]
        tik[k] = None if not chleny else {"top20": top20, "chleny": [c["kl"] for c in chleny], "i": [c["i"] for c in chleny],
                                          "plato": [c["plato"] for c in chleny]}
    return {"god": t, "okno": (f"{pul.god0}-01-01", f"{t - 1}-12-31"), "tikery": tik, "n_proby": pul.n_proby()}


def kalibrovat(pul: Pul21, period: str = "train4") -> dict:
    """{t: god(...)} по годам walk-forward периода (2017–2019)."""
    nachalo, gody = K.PERIODY[period]
    if int(nachalo[:4]) != pul.god0:
        raise ValueError(f"период {period} начинается в {nachalo[:4]}, пул — в {pul.god0}")
    return {t: god(pul, t) for t in gody}


def pul_iz_ledger(root: Path, sid: str, osi: dict, god0: int = 2016) -> Pul21:
    """Пул из ledger 2.1 с |позицией| (poz_konec): комиссия строк едина, иначе отказ. Ряды не нужны — только суммы."""
    ch = lm.Chitatel(root, sid)
    if ch.v != lm.V21 or not ch.manifest.get("poz_konec"):
        raise ValueError(f"{sid}: нужен ledger 2.1 с poz_konec (экспозиция и выход на конце окна из сумм)")
    rows = ch.rows()
    kom = {float(r["klyuch"]["komissiya"]) for r in rows}
    if len(kom) != 1:
        raise ValueError(f"{sid}: комиссия строк ledger не единая: {sorted(kom)}")
    tochki = [{a: v for a, v in r["klyuch"].items() if a not in K.SLUZH} for r in rows]
    return Pul21(sid, {a: [lm.tok(v) for v in vals] for a, vals in osi.items()}, tochki, ch.summy(rows, "tik"),
                 np.asarray(ch.manifest["nbars"], float), god0, kom.pop())
