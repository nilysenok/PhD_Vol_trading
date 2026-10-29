"""Контекст сигнала v6 (Ж-132 п. 2; инвентаризация v6 §1.1, оси К9–К16, К20–К24) → (long_ok, short_ok).

Тонкий слой над готовыми модулями: `e1_sektor` (К9–К12), `e1_rel_sektor` (К13–К16), `filtr_e1` (К20–К23),
`rezhim_e1` (К24). Ничего не читает с диска: ряды приходят в `Dannye` (один тикер на сетке `dates`).
Модуль только ЗАПРЕЩАЕТ вход: результат — два булевых массива по `dates`, оси объединяются по «И».
Оси точки `pt` (dict или объект), значение «выкл.» — первое, при всех «выкл.» → (True, True) без чтения рядов
(побайтно путь v5, Ж-132 п. 2):
  sektor_trend, shirina ∈ {net, protiv, po_trendu}; ma_n (200); porog_shiriny (0,5)   — К9–К12
  relf_napr ∈ {net, po_sile, protiv}; relf_k (1); relf_okno (5); relf_bench ∈ {sektor, rynok} — К13–К16
  post_shok ∈ {net, "z0:m0"} (3:1, 2:0,5) — шок ≥ z0 σ̂ в [d−4; d] при |Z_рынка| ≤ m0   — К20
  szhatie, rv_perevorot, obyom_anom ∈ {net, da}                                          — К21–К23
  rezhim ∈ {net, imoex_ma200, rank08}                                                    — К24
К21 в v6 действует только у тренда (у контра — дубль `squeeze_pct`): включает ли ось, решает куб, не модуль.
Всё по данным ≤ close d (решение на d, исполнение d+1); нет истории → вход запрещён.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fastbt import e1_rel_sektor as rel
from fastbt import e1_sektor as sek
from fastbt import filtr_e1 as f

REZHIM = {"net": None, "imoex_ma200": "ma200", "rank08": "rank08"}
BENCH = ("sektor", "rynok")
DA = ("net", "da")
IMOEX = "IMOEX"
DAWN = pd.Timestamp("1900-01-01")


@dataclass
class Dannye:
    """Ряды одного тикера `t` на сетке `dates` (datetime64). Поля нужны только включённым осям."""
    t: str
    dates: np.ndarray
    close: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None
    volume: np.ndarray | None = None
    sig5: np.ndarray | None = None       # σ̂ rv5 / rv22 / rv66 по дням (известна на 18:50 своего дня)
    sig22: np.ndarray | None = None
    sig66: np.ndarray | None = None
    Z: np.ndarray | None = None          # Z_d рынка = R_d / σ_рынка,d−1 на dates (К20)
    sektor: sek.Ctx | None = None        # К9–К12
    rel_sektor: rel.Ctx | None = None    # К13–К16, bench = sektor
    rel_rynok: rel.Ctx | None = None     # К13–К16, bench = rynok (см. `rynok_ctx`)
    rezhim: dict = field(default_factory=dict)   # {"ma200": bool[], "rank08": bool[]} на dates


def rynok_ctx(imoex: pd.Series, close: dict, sigma: dict) -> rel.Ctx:
    """Контекст К16 `rynok`: IMOEX как индекс каждого тикера с 1900 года (сектор — не нужен)."""
    pit = {t: [(DAWN, IMOEX)] for t in close}
    return rel.Ctx({IMOEX: imoex}, close, sigma, pit)


def parse_shok(v) -> tuple[float, float]:
    """'3:1' → (3, 1); '2:0,5' → (2, 0,5). Порог шока z0 > 0, m0 ≥ 0."""
    try:
        a, b = (float(x.replace(",", ".")) for x in str(v).split(":"))
    except ValueError as e:
        raise ValueError(f"kontekst_v6: post_shok {v!r}, ждали 'net' или 'z0:m0'") from e
    if a <= 0 or b < 0:
        raise ValueError(f"kontekst_v6: post_shok {v!r}: z0 > 0, m0 ≥ 0")
    return a, b


def _need(x, name: str, ось: str):
    if x is None:
        raise ValueError(f"kontekst_v6: ось {ось} включена, а в Dannye нет {name}")
    return x


def _flag(v, name: str) -> bool:
    if v not in DA:
        raise ValueError(f"kontekst_v6: {name}={v!r} вне {DA}")
    return v == "da"


def _sektor(pt, d: Dannye):
    if sek._g(pt, "sektor_trend", "net") == "net" and sek._g(pt, "shirina", "net") == "net":
        return None
    return sek.allow(pt, d.t, d.dates, _need(d.sektor, "sektor", "sektor_trend/shirina"))


def _rel(pt, d: Dannye):
    pt2 = {"rel_napravlenie": sek._g(pt, "relf_napr", "net"), "rel_k": sek._g(pt, "relf_k", 1.0),
           "rel_okno": sek._g(pt, "relf_okno", 5)}
    if pt2["rel_napravlenie"] == "net":
        return None
    b = sek._g(pt, "relf_bench", "sektor")
    if b not in BENCH:
        raise ValueError(f"kontekst_v6: relf_bench={b!r} вне {BENCH}")
    ctx = d.rel_sektor if b == "sektor" else d.rel_rynok
    return rel.allow(pt2, d.t, d.dates, _need(ctx, f"rel_{b}", "relf_napr"))


def _shok(pt, d: Dannye):
    v = sek._g(pt, "post_shok", "net")
    if v == "net":
        return None
    z0, m0 = parse_shok(v)
    z = f.z_series(f.ret(_need(d.close, "close", "post_shok")), _need(d.sig22, "sig22", "post_shok"))
    with np.errstate(invalid="ignore"):
        calm = np.abs(_need(d.Z, "Z", "post_shok")) <= m0
        return f.win_any((-z >= z0) & calm, 5), f.win_any((z >= z0) & calm, 5)


def _symm(pt, d: Dannye):
    """К21–К23: условие не зависит от стороны."""
    out = []
    if _flag(sek._g(pt, "szhatie", "net"), "szhatie"):
        out.append(f.szhatie(f.rg_series(_need(d.high, "high", "szhatie"), _need(d.low, "low", "szhatie"),
                                         _need(d.close, "close", "szhatie"))))
    if _flag(sek._g(pt, "rv_perevorot", "net"), "rv_perevorot"):
        s5 = _need(d.sig5, "sig5", "rv_perevorot")
        out.append(f.rv_perevorot(f.u_series(s5, _need(d.sig66, "sig66", "rv_perevorot")), s5))
    if _flag(sek._g(pt, "obyom_anom", "net"), "obyom_anom"):
        z = f.z_series(f.ret(_need(d.close, "close", "obyom_anom")), _need(d.sig22, "sig22", "obyom_anom"))
        out.append(f.obyom(f.v_series(_need(d.volume, "volume", "obyom_anom")), z))
    return [(x, x) for x in out]


def _rezhim(pt, d: Dannye):
    v = sek._g(pt, "rezhim", "net")
    if v not in REZHIM:
        raise ValueError(f"kontekst_v6: rezhim={v!r} вне {tuple(REZHIM)}")
    if v == "net":
        return None
    x = np.asarray(_need(d.rezhim.get(REZHIM[v]), f"rezhim[{REZHIM[v]!r}]", "rezhim"), bool)
    return x, x


def long_short(pt, d: Dannye) -> tuple[np.ndarray, np.ndarray]:
    """(long_ok, short_ok) по `d.dates`: вход по close `dates[i]` разрешён. Оси — по «И»."""
    n = len(d.dates)
    lo, sr = np.ones(n, bool), np.ones(n, bool)
    parts = [_sektor(pt, d), _rel(pt, d), _shok(pt, d), *_symm(pt, d), _rezhim(pt, d)]
    for p in parts:
        if p is None:
            continue
        a, b = np.asarray(p[0], bool), np.asarray(p[1], bool)
        if len(a) != n or len(b) != n:
            raise ValueError(f"kontekst_v6: длина флага {len(a)} ≠ {n} дат")
        lo, sr = lo & a, sr & b
    return lo, sr
