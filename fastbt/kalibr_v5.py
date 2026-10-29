"""П7 (Ж-127 п. 3; контракт `docs/otchety/2026-10-29_setka_v5.md` §3): калибратор v5 поверх ledger Соболя. Данных не читает.

Год t walk-forward: окно отбора расширяющееся, [начало периода; 31.12(t−1)], ни одного бара года t (маска [0], не [1]).
Тройка (стратегия, подход, тикер): фаза 1 — ворота v4 (sh > 0, mdd > −30 %, экспозиция > 5 %) → top-50 по Шарпу окна;
фаза 2 — соседи ±1 по осям с эффектом (размах средних Шарпа по значениям оси ≥ TOR_MIN), `sweep_plato.neighbours`,
плато = `sweep_plato.plateau_value` (точка и посчитанные соседи), соседей < MIN_NB — край, не берётся → top-20 по плато;
фаза 3 — до П2 тождество: top-10 (`grids.ENS_K`) по Шарпу окна из top-20; голосование — `execlayer.vote`, `grids.VOTE_FRAC`,
как `ApproachA.ensemble`. Шарп — ddof = 1 · √252 (`metriki_e.sr_dnevnoy`, Д7); выход позиции на конце окна оплачивается.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from fastbt import grids, ledger_v2 as lm, period_v2 as PV, sweep_plato as SP
from fastbt.approach_a import _desc
from fastbt.data import year_bounds
from fastbt.execlayer import vote
from fastbt.kalibr_v5_golos import golos_v2
from fastbt.sweep_pts import MIN_NB, TOR_MIN

PERIODY = {"train4": ("2016-01-01", (2017, 2018, 2019)),     # §3: окно 2017 на train4 — один 2016
           "v2": ("2015-01-01", (2017, 2018, 2019)),         # Ж-113 п. 1, 3: train 2015–2019, первый год 2017
           "wf": ("2015-01-01", tuple(range(2017, 2027)))}   # Ж-136: 10 окон; 2026 — янв.–фев., окно 2015–2025 (44 кв.)
TOP_F1, TOP_F2, MDD_MIN, EXP_MIN = 50, 20, -0.30, 5.0        # approach_a.py:67, 77, 84
SLUZH = ("period", "komissiya", "fdr")


def klyuch(sid: str, pt: dict) -> str:
    """Канон точки (`canon_v2`): псевдонимы — одна точка и одна проба в N."""
    return json.dumps({k: lm.tok(v) for k, v in lm.canon_v2(sid, pt).items()}, sort_keys=True, ensure_ascii=False)


@dataclass
class Pul:
    """Точки одной стратегии: ex[P, тикеры, T] — дневной избыточный ряд, poz — позиции (или None), osi — порядок значений."""
    sid: str
    daty: np.ndarray
    osi: dict
    tochki: list
    ex: np.ndarray
    poz: Optional[np.ndarray] = None
    komissiya: float = 0.0
    dosschitat: Optional[Callable] = None     # pt -> (ex[тикеры, T], poz | None): новая проба; None — соседей вне ledger нет
    novye: dict = field(default_factory=dict)
    period: str = "train4"                    # охрана дат: wf — до конца периода и только при снятом замке (`period_v2`)

    def __post_init__(self):
        self.daty = np.asarray(self.daty, "datetime64[D]")
        self._ohrana()
        if self.ex.shape[0] != len(self.tochki) or self.ex.shape[2] != self.daty.size:
            raise ValueError("ex: [точки, тикеры, дни]")
        self.idx, self.kol = {}, {}
        for i, pt in enumerate(self.tochki):
            self.idx.setdefault(klyuch(self.sid, pt), i)
        for a in self.osi:
            self.kol[a] = np.array([lm.tok(pt[a]) if a in pt else "" for pt in self.tochki])

    def _ohrana(self):
        if not self.daty.size or self.daty.max() < np.datetime64("2020-01-01"):
            return
        if self.period != "wf":
            raise ValueError("даты ≥ 2020 в пуле калибровки train/В2 — отказ (validation — П8)")
        PV.proverit_chtenie("wf", otmetka=PV.OTMETKA)   # без отметки владельца — ZamokVladeltsa
        if self.daty.max() > np.datetime64(PV.PERIODY["wf"].konec.date()):
            raise ValueError("даты пула wf позже конца периода — отказ")

    def n_proby(self) -> int:
        """N для DSR фаз 1–2: различные ключи ledger + досчитанные соседи."""
        return len(self.idx) + len(self.novye)

    def ryad(self, kl: str, pt: dict, k: int):
        """(ex[T], poz[T] | None) по ключу; вне ledger — dosschitat (проба в N) или None."""
        if kl in self.idx:
            i = self.idx[kl]
            return self.ex[i, k], None if self.poz is None else self.poz[i, k]
        if kl not in self.novye:
            if self.dosschitat is None:
                return None
            self.novye[kl] = self.dosschitat(pt)
        e, p = self.novye[kl]
        return e[k], None if p is None else p[k]


def granica(daty: np.ndarray, t: int) -> int:
    """Конец окна отбора года t (не включая): индекс первого бара года t."""
    return int(np.searchsorted(daty, np.datetime64(f"{t}-01-01"), side="left"))


def sharp_okna(x: np.ndarray) -> np.ndarray:
    """По строкам: mean/sd (ddof = 1)·√252; sd ≤ 1e-12 → 0 — определение `sr_dnevnoy` (Д7)."""
    x = np.asarray(x, float)
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, x.mean(axis=-1) / sd, 0.0) * np.sqrt(252.0)


def metriki(x: np.ndarray, p: Optional[np.ndarray], kom: float) -> tuple:
    """(sh, mdd, экспозиция %) окна по строкам. Открытая на конце окна позиция платит выход (как stats.window, № 6)."""
    x = np.array(x, float, ndmin=2)
    if p is not None:
        p = np.array(p, float, ndmin=2)
        x[:, -1] -= kom * np.abs(p[:, -1])
    c = np.cumsum(x, axis=-1)
    mdd = (c - np.maximum.accumulate(np.maximum(c, 0.0), axis=-1)).min(axis=-1)
    ex = ((x if p is None else p) != 0).mean(axis=-1) * 100.0       # ↺ класс B: без позиций — доля дней с ненулевым рядом
    return sharp_okna(x), mdd, ex


def osi_s_effektom(pul: Pul, sh: np.ndarray) -> list:
    """Оси, у которых размах средних Шарпа окна по значениям ≥ TOR_MIN (Ж-031: параметр с эффектом)."""
    out = []
    for a, vals in pul.osi.items():
        sr = [float(sh[m].mean()) for m in (pul.kol[a] == lm.tok(v) for v in vals) if m.any()]
        if len(sr) > 1 and max(sr) - min(sr) >= TOR_MIN:
            out.append(a)
    return out


def faza1(pul: Pul, k: int, s: int, e: int) -> tuple:
    """(sh всех точек, top-50 индексов после ворот; ворот не прошёл никто — top-10 по sh, как approach_a.py:75-76)."""
    sh, mdd, ex = metriki(pul.ex[:, k, s:e], None if pul.poz is None else pul.poz[:, k, s:e], pul.komissiya)
    ok = (sh > 0) & (mdd > MDD_MIN) & (ex > EXP_MIN)
    vse = list(range(len(pul.tochki)))
    prohod = [i for i in vse if ok[i]] or _desc(vse, lambda i: sh[i])[:grids.ENS_K]
    return sh, _desc(prohod, lambda i: sh[i])[:TOP_F1]


def sh_soseda(pul: Pul, kl: str, q: dict, k: int, s: int, e: int, sh: np.ndarray):
    if kl in pul.idx:
        return float(sh[pul.idx[kl]])
    r = pul.ryad(kl, q, k)
    if r is None:
        return None
    return float(metriki(r[0][s:e], None if r[1] is None else r[1][s:e], pul.komissiya)[0][0])


def faza2(pul: Pul, k: int, s: int, e: int, sh: np.ndarray, top50: list) -> list:
    """top-20 по плато: [{i, kl, sh, plato, n_nb}]; край (соседей < MIN_NB) не берётся."""
    osi, exp, out = osi_s_effektom(pul, sh), {"grid": pul.osi}, []
    fix = lambda q: lm.canon_v2(pul.sid, q)        # noqa: E731 — сосед по канону = точка ledger
    for i in top50:
        pt = pul.tochki[i]
        nb = [sh_soseda(pul, klyuch(pul.sid, q), q, k, s, e, sh) for q in SP.neighbours(exp, fix(pt), osi, fix)]
        nb = [v for v in nb if v is not None and v == v]
        if len(nb) >= MIN_NB:
            out.append(dict(i=i, kl=klyuch(pul.sid, pt), sh=float(sh[i]), plato=SP.plateau_value(float(sh[i]), nb), n_nb=len(nb)))
    return _desc(out, lambda c: c["plato"])[:TOP_F2]


def ansambl(pul: Pul, k: int, chleny: list, s_t: int, e_t: int):
    """Позиции ансамбля на годе t: `vote` по позициям членов (как ApproachA.ensemble); позиций нет — None."""
    if pul.poz is None or not chleny:
        return None
    stack = np.ascontiguousarray(np.stack([pul.poz[c["i"], k] for c in chleny]), dtype=float)
    out = np.zeros(pul.daty.size)
    vote(stack, len(chleny), s_t, e_t, out, grids.VOTE_FRAC)
    return out[s_t:e_t]


def god(pul: Pul, t: int, nachalo: str, poz_fn=None) -> dict:
    """Отбор года t по всем тикерам пула: {okno, tikery: {k: {chleny, poz, ryad} | None}, n_proby}.
    poz_fn(kl, k) -> pos[T] автомата v2 (П6, kalibr_v5_golos) — голос по ним; None — по позициям пула (`ansambl`, v4)."""
    s, e = int(np.searchsorted(pul.daty, np.datetime64(nachalo))), granica(pul.daty, t)
    s_t, e_t = year_bounds(pul.daty, t)
    if e - s < 2:
        return {"god": t, "okno": None, "tikery": {}, "n_proby": pul.n_proby()}
    tik = {}
    for k in range(pul.ex.shape[1]):
        sh, top50 = faza1(pul, k, s, e)
        top20 = faza2(pul, k, s, e, sh, top50)
        chleny = _desc(top20, lambda c: c["sh"])[:grids.ENS_K]          # фаза 3 до П2 — тождество (§3)
        tik[k] = None if not chleny else {"chleny": [c["kl"] for c in chleny], "plato": [c["plato"] for c in chleny],
                                          "poz": ansambl(pul, k, chleny, s_t, e_t) if poz_fn is None
                                          else golos_v2(poz_fn, chleny, k, pul.daty.size, s_t, e_t),
                                          "ryad": np.stack([pul.ex[c["i"], k, s_t:e_t] for c in chleny])}
    return {"god": t, "okno": (str(pul.daty[s]), str(pul.daty[e - 1])), "tikery": tik, "n_proby": pul.n_proby()}


def kalibrovat(pul: Pul, period: str = "train4", poz_fn=None) -> dict:
    """{t: god(...)} по годам walk-forward периода (2017–2019)."""
    nachalo, gody = PERIODY[period]
    return {t: god(pul, t, nachalo, poz_fn) for t in gody}


def pul_iz_ledger(root: Path, sid: str, osi: dict) -> Pul:
    """Пул из ledger v2.0 (train4): точки — `klyuch` строк без служебных полей, ex — `tik`; позиций в v2.0 нет."""
    ch = lm.Chitatel(root, sid)
    rows = ch.rows()
    tochki = [{a: v for a, v in r["klyuch"].items() if a not in SLUZH} for r in rows]
    daty = np.load(Path(root) / sid / "daty.npy")
    return Pul(sid, daty, {a: [lm.tok(v) for v in vals] for a, vals in osi.items()}, tochki, ch.tik(rows))
